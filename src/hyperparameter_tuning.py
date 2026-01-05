"""
Hyperparameter Tuning Module - Optuna-based optimization for trading models.

This module provides:
- Bayesian optimization using Optuna
- Time-series cross-validation
- XGBoost hyperparameter tuning
- Ensemble weight optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed. Run: pip install optuna")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna with time-series CV.
    
    Features:
    - Bayesian optimization with TPE sampler
    - Time-series cross-validation (no data leakage)
    - XGBoost parameter tuning
    - Ensemble weight optimization
    - Early stopping and pruning
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            config: Configuration dictionary with optimization parameters
        """
        self.config = config or {}
        self.opt_config = self.config.get('optimization', {
            'optuna_trials': 100,
            'cv_folds': 5,
            'early_stopping': 20
        })
        
        self.best_params = {}
        self.study = None
        self.optimization_history = []
        
    def create_xgboost_objective(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Callable:
        """
        Create Optuna objective function for XGBoost.
        
        Args:
            X: Feature matrix
            y: Target labels
            n_splits: Number of CV folds
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial: optuna.Trial) -> float:
            # Define parameter search space
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'verbosity': 0,
                'random_state': 42,
                
                # Tunable parameters
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.8, 1.2),
            }
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Handle any remaining NaN values
                X_train = X_train.fillna(0)
                X_val = X_val.fillna(0)
                
                # Train model
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Evaluate
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                scores.append(score)
                
                # Early pruning
                trial.report(np.mean(scores), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        return objective
    
    def optimize_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ticker: str = "default",
        n_trials: int = None,
        n_splits: int = None
    ) -> Dict:
        """
        Run Optuna optimization for XGBoost.
        
        Args:
            X: Feature matrix
            y: Target labels
            ticker: Ticker name for logging
            n_trials: Number of optimization trials
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with best parameters and study results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        n_trials = n_trials or self.opt_config.get('optuna_trials', 100)
        n_splits = n_splits or self.opt_config.get('cv_folds', 5)
        
        logger.info(f"Starting Optuna optimization for {ticker}")
        logger.info(f"  Trials: {n_trials}, CV folds: {n_splits}")
        logger.info(f"  Data shape: {X.shape}")
        
        # Create study with TPE sampler
        sampler = TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f'xgboost_{ticker}'
        )
        
        # Create objective
        objective = self.create_xgboost_objective(X, y, n_splits)
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,
            n_jobs=1  # Sequential for reproducibility
        )
        
        # Store results
        self.study = study
        self.best_params[ticker] = study.best_params
        
        # Compile results
        results = {
            'ticker': ticker,
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time': datetime.now().isoformat(),
            'trials_dataframe': study.trials_dataframe().to_dict()
        }
        
        self.optimization_history.append(results)
        
        logger.info(f"✅ Optimization complete for {ticker}")
        logger.info(f"   Best accuracy: {study.best_value:.4f}")
        logger.info(f"   Best params: {study.best_params}")
        
        return results
    
    def optimize_ensemble_weights(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        n_trials: int = 50
    ) -> Dict:
        """
        Optimize ensemble weights using Optuna.
        
        Args:
            predictions: Dictionary of model_name -> predictions
            y_true: True labels
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with optimal weights
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available")
        
        model_names = list(predictions.keys())
        n_models = len(model_names)
        
        def objective(trial: optuna.Trial) -> float:
            # Suggest weights (will be normalized to sum to 1)
            weights = []
            for i, name in enumerate(model_names):
                w = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
                weights.append(w)
            
            # Normalize weights
            total = sum(weights)
            if total == 0:
                weights = [1.0 / n_models] * n_models
            else:
                weights = [w / total for w in weights]
            
            # Combine predictions
            combined = np.zeros_like(y_true, dtype=float)
            for w, name in zip(weights, model_names):
                combined += w * predictions[name]
            
            # Round to get binary predictions
            y_pred = (combined >= 0.5).astype(int)
            
            # Calculate accuracy
            return accuracy_score(y_true, y_pred)
        
        # Run optimization
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Extract best weights
        best_weights = {}
        total = 0
        for name in model_names:
            w = study.best_params.get(f'weight_{name}', 0)
            best_weights[name] = w
            total += w
        
        # Normalize
        for name in model_names:
            best_weights[name] /= total if total > 0 else 1
        
        results = {
            'best_weights': best_weights,
            'best_accuracy': study.best_value,
            'n_trials': len(study.trials)
        }
        
        logger.info(f"✅ Ensemble weight optimization complete")
        logger.info(f"   Best weights: {best_weights}")
        logger.info(f"   Best accuracy: {study.best_value:.4f}")
        
        return results
    
    def cross_validate_params(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict,
        n_splits: int = 5
    ) -> Dict:
        """
        Cross-validate a specific parameter set.
        
        Args:
            X: Feature matrix
            y: Target labels
            params: XGBoost parameters to evaluate
            n_splits: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Handle NaN
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)
            
            # Train
            model = xgb.XGBClassifier(**params, use_label_encoder=False, verbosity=0)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Evaluate
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            fold_results.append({
                'fold': fold,
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
        
        # Aggregate results
        results = {
            'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
            'mean_precision': np.mean([r['precision'] for r in fold_results]),
            'mean_recall': np.mean([r['recall'] for r in fold_results]),
            'mean_f1': np.mean([r['f1'] for r in fold_results]),
            'fold_results': fold_results
        }
        
        return results
    
    def train_with_best_params(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        ticker: str = "default"
    ) -> Tuple[xgb.XGBClassifier, Dict]:
        """
        Train model with best parameters found during optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            ticker: Ticker name
            
        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        if ticker not in self.best_params:
            raise ValueError(f"No optimized params for {ticker}. Run optimize_xgboost first.")
        
        params = self.best_params[ticker].copy()
        params.update({
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 0,
            'random_state': 42
        })
        
        # Handle NaN
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Train
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'params': params
        }
        
        logger.info(f"✅ Trained {ticker} with optimized params")
        logger.info(f"   Test accuracy: {metrics['accuracy']:.4f}")
        
        return model, metrics
    
    def save_optimization_results(self, output_dir: str):
        """
        Save optimization results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best params
        params_path = output_path / 'best_params.json'
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save optimization history
        history_path = output_path / 'optimization_history.json'
        # Convert to JSON-serializable format
        serializable_history = []
        for h in self.optimization_history:
            h_copy = h.copy()
            if 'trials_dataframe' in h_copy:
                del h_copy['trials_dataframe']  # Too large
            serializable_history.append(h_copy)
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"✅ Saved optimization results to {output_dir}")
    
    def load_best_params(self, params_path: str):
        """
        Load previously saved best parameters.
        
        Args:
            params_path: Path to best_params.json
        """
        with open(params_path, 'r') as f:
            self.best_params = json.load(f)
        logger.info(f"Loaded best params for: {list(self.best_params.keys())}")
    
    def get_param_importance(self, ticker: str = None) -> pd.DataFrame:
        """
        Get parameter importance from Optuna study.
        
        Args:
            ticker: Ticker to get importance for (uses last study if None)
            
        Returns:
            DataFrame with parameter importance
        """
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            df = pd.DataFrame([
                {'parameter': k, 'importance': v}
                for k, v in importance.items()
            ]).sort_values('importance', ascending=False)
            return df
        except Exception as e:
            logger.warning(f"Could not compute param importance: {e}")
            return pd.DataFrame()


def main():
    """Main function to run hyperparameter tuning."""
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load features
    features_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features_v2.parquet'
    if not features_path.exists():
        features_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features.parquet'
    
    df = pd.read_parquet(features_path)
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(config)
    
    # Get feature columns and target
    feature_cols = [c for c in df.columns if not any(x in c.lower() for x in ['target', 'date', 'ticker'])]
    target_col = 'GBPUSD_X_target_next_day'  # Example ticker
    
    if target_col not in df.columns:
        # Find any target column
        target_cols = [c for c in df.columns if 'target' in c.lower()]
        if target_cols:
            target_col = target_cols[0]
        else:
            print("No target column found!")
            return
    
    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]
    
    # Remove any remaining NaN in target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"Training data shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Run optimization (reduced trials for demo)
    results = optimizer.optimize_xgboost(X, y, ticker='GBPUSD_X', n_trials=20)
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'models' / 'optimized'
    optimizer.save_optimization_results(str(output_dir))
    
    print("\n✅ Hyperparameter Tuning Complete!")
    print(f"   Best accuracy: {results['best_value']:.4f}")
    print(f"   Best params: {results['best_params']}")


if __name__ == "__main__":
    main()
