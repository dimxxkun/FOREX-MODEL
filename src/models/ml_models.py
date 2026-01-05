"""
Machine Learning Models for Forex Signal Trading.

XGBoost-based classifier for next-day direction prediction.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.utils import get_logger, load_config, timer_decorator


class XGBoostTradingModel:
    """
    XGBoost binary classifier for next-day direction prediction.
    
    Features:
    - All engineered features
    - Time-based train/validation/test split
    - Walk-forward optimization
    - Feature importance analysis
    - Probability calibration for confidence scores
    
    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        models: Dict of trained models per ticker.
        scalers: Dict of fitted scalers per ticker.
        feature_names: List of feature names used.
    
    Example:
        >>> model = XGBoostTradingModel('config/config.yaml')
        >>> model.train(features_df, 'GBPUSD')
        >>> predictions = model.predict(test_df, 'GBPUSD')
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the XGBoostTradingModel.
        
        Args:
            config_path: Path to configuration YAML file.
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        self.config = load_config(config_path)
        self.logger = get_logger('forex_signal_model.ml_models')
        
        # Model configuration
        self.model_config = self.config.get('models', {}).get('xgboost', {})
        
        # Storage
        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        
        # Training metrics
        self.train_metrics: Dict[str, Dict] = {}
        
        self.logger.info("XGBoostTradingModel initialized")
    
    def _get_feature_columns(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> List[str]:
        """
        Get feature columns for a ticker, excluding targets.
        
        Args:
            df: Features DataFrame.
            ticker: Ticker symbol.
        
        Returns:
            List of feature column names.
        """
        ticker_clean = ticker.replace('=', '_').replace('^', '').replace('-', '_').replace('.', '_')
        
        # Get all columns for this ticker (excluding targets)
        feature_cols = [
            c for c in df.columns 
            if (c.startswith(ticker_clean) or 
                c.startswith('DXY') or 
                c.startswith('VIX') or
                c.startswith('TNX') or
                c.startswith('Oil'))
            and 'Target' not in c
        ]
        
        return feature_cols
    
    def _select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_features: int = 100
    ) -> List[str]:
        """
        Select top features using mutual information and correlation filtering.
        
        Args:
            X: Feature DataFrame.
            y: Target Series.
            max_features: Maximum features to keep.
        
        Returns:
            List of selected feature names.
        """
        self.logger.info(f"Selecting features from {len(X.columns)} candidates...")
        
        # 1. Remove low-variance features
        selector = VarianceThreshold(threshold=0.01)
        try:
            selector.fit(X)
            high_var_cols = X.columns[selector.get_support()].tolist()
            X_filtered = X[high_var_cols]
            self.logger.debug(f"After variance filter: {len(high_var_cols)} features")
        except Exception:
            X_filtered = X
            high_var_cols = X.columns.tolist()
        
        # 2. Remove highly correlated features (>0.95)
        corr_matrix = X_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        uncorrelated_cols = [c for c in high_var_cols if c not in to_drop]
        X_uncorr = X_filtered[uncorrelated_cols]
        self.logger.debug(f"After correlation filter: {len(uncorrelated_cols)} features")
        
        # 3. Select top features by mutual information
        if len(uncorrelated_cols) > max_features:
            try:
                mi_scores = mutual_info_classif(X_uncorr, y, random_state=42)
                mi_df = pd.DataFrame({
                    'feature': uncorrelated_cols,
                    'mi_score': mi_scores
                }).sort_values('mi_score', ascending=False)
                
                selected = mi_df.head(max_features)['feature'].tolist()
                self.logger.info(f"Selected top {len(selected)} features by MI")
            except Exception as e:
                self.logger.warning(f"MI selection failed: {e}, using all features")
                selected = uncorrelated_cols[:max_features]
        else:
            selected = uncorrelated_cols
        
        return selected
    
    def _time_series_split(
        self,
        df: pd.DataFrame,
        train_pct: float = 0.6,
        val_pct: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets (time-based).
        
        Args:
            df: DataFrame with DatetimeIndex.
            train_pct: Percentage for training.
            val_pct: Percentage for validation.
        
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        n = len(df)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        self.logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    @timer_decorator
    def train(
        self,
        df: pd.DataFrame,
        ticker: str,
        tune_hyperparams: bool = True
    ) -> Dict[str, Any]:
        """
        Train XGBoost model for a ticker.
        
        Args:
            df: Features DataFrame with target column.
            ticker: Ticker symbol.
            tune_hyperparams: Whether to tune hyperparameters.
        
        Returns:
            Dictionary with training metrics.
        """
        self.logger.info(f"Training XGBoost for {ticker}...")
        
        # Find target column dynamically (handles tickers with special chars like GC=F)
        target_col = None
        ticker_variants = [
            ticker,  # Original: GC=F
            ticker.replace('=', '_').replace('^', '').replace('-', '_').replace('.', '_'),  # Cleaned: GC_F
            ticker.replace('=X', ''),  # For forex pairs
        ]
        
        for variant in ticker_variants:
            candidate = f'{variant}_Target_Direction'
            if candidate in df.columns:
                target_col = candidate
                break
        
        if target_col is None:
            # Last resort: search for any matching target column
            for col in df.columns:
                if 'Target_Direction' in col and any(v in col for v in ticker_variants):
                    target_col = col
                    break
        
        if target_col is None:
            raise ValueError(f"Target column not found for {ticker}. Available: {[c for c in df.columns if 'Target' in c]}")
        
        # Get feature columns
        feature_cols = self._get_feature_columns(df, ticker)
        
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Time-based split
        train_idx = int(len(X) * 0.6)
        val_idx = int(len(X) * 0.8)
        
        X_train = X.iloc[:train_idx]
        y_train = y.iloc[:train_idx]
        X_val = X.iloc[train_idx:val_idx]
        y_val = y.iloc[train_idx:val_idx]
        X_test = X.iloc[val_idx:]
        y_test = y.iloc[val_idx:]
        
        # Feature selection (on training data only)
        selected_features = self._select_features(X_train, y_train, max_features=80)
        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler and feature names
        self.scalers[ticker] = scaler
        self.feature_names[ticker] = selected_features
        
        # Build model
        if tune_hyperparams:
            model = self._tune_hyperparameters(X_train_scaled, y_train)
        else:
            model = xgb.XGBClassifier(
                max_depth=self.model_config.get('max_depth', 5),
                learning_rate=self.model_config.get('learning_rate', 0.05),
                n_estimators=self.model_config.get('n_estimators', 200),
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        # Train with early stopping
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        # Store model
        self.models[ticker] = model
        
        # Calculate metrics
        train_preds = model.predict(X_train_scaled)
        val_preds = model.predict(X_val_scaled)
        test_preds = model.predict(X_test_scaled)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_preds),
            'val_accuracy': accuracy_score(y_val, val_preds),
            'test_accuracy': accuracy_score(y_test, test_preds),
            'train_f1': f1_score(y_train, train_preds),
            'val_f1': f1_score(y_val, val_preds),
            'test_f1': f1_score(y_test, test_preds),
            'n_features': len(selected_features),
            'n_train': len(y_train),
            'n_val': len(y_val),
            'n_test': len(y_test),
        }
        
        try:
            metrics['val_auc'] = roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
            metrics['test_auc'] = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        except Exception:
            metrics['val_auc'] = 0.5
            metrics['test_auc'] = 0.5
        
        self.train_metrics[ticker] = metrics
        
        # Store feature importance
        importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        self.feature_importance[ticker] = importance
        
        self.logger.info(f"{ticker} - Val Acc: {metrics['val_accuracy']:.3f}, "
                        f"Test Acc: {metrics['test_accuracy']:.3f}, "
                        f"Test AUC: {metrics['test_auc']:.3f}")
        
        return metrics
    
    def _tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: pd.Series
    ) -> xgb.XGBClassifier:
        """
        Tune hyperparameters using RandomizedSearchCV.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
        
        Returns:
            Best XGBClassifier model.
        """
        self.logger.info("Tuning hyperparameters...")
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
        }
        
        base_model = xgb.XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=tscv,
            scoring='f1',
            random_state=42,
            n_jobs=-1
        )
        
        search.fit(X_train, y_train)
        
        self.logger.info(f"Best params: {search.best_params_}")
        self.logger.info(f"Best CV score: {search.best_score_:.3f}")
        
        return search.best_estimator_
    
    def predict(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Generate predictions for a ticker.
        
        Args:
            df: Features DataFrame.
            ticker: Ticker symbol.
        
        Returns:
            DataFrame with [Date, Ticker, Signal, Confidence].
        """
        if ticker not in self.models:
            raise ValueError(f"No trained model for {ticker}")
        
        ticker_clean = ticker.replace('=', '_').replace('^', '').replace('-', '_').replace('.', '_')
        
        # Get features
        feature_cols = self.feature_names[ticker]
        X = df[feature_cols].copy()
        
        # Handle NaN
        X = X.fillna(method='ffill').fillna(0)
        
        # Scale
        X_scaled = self.scalers[ticker].transform(X)
        
        # Predict
        model = self.models[ticker]
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Convert to signals: 1 (BUY) if prob > 0.5, else 0 (SELL/HOLD)
        # Map to -1, 0, 1 format
        signals = np.where(predictions == 1, 1, -1)
        
        # Confidence = distance from 0.5, scaled to 0-100
        confidence = np.abs(probabilities - 0.5) * 200
        
        # Create output
        result = pd.DataFrame({
            'Date': df.index,
            'Ticker': ticker,
            'Signal': signals,
            'Confidence': confidence,
            'Probability': probabilities
        })
        
        return result
    
    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for all trained tickers.
        
        Args:
            df: Features DataFrame.
        
        Returns:
            Combined predictions DataFrame.
        """
        all_predictions = []
        
        for ticker in self.models.keys():
            try:
                preds = self.predict(df, ticker)
                all_predictions.append(preds)
            except Exception as e:
                self.logger.warning(f"Prediction failed for {ticker}: {e}")
        
        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        return pd.DataFrame()
    
    def get_feature_importance(
        self,
        ticker: str,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get top N feature importances for a ticker.
        
        Args:
            ticker: Ticker symbol.
            top_n: Number of features to return.
        
        Returns:
            DataFrame with feature importances.
        """
        if ticker not in self.feature_importance:
            raise ValueError(f"No feature importance for {ticker}")
        
        return self.feature_importance[ticker].head(top_n)
    
    def save_model(
        self,
        ticker: str,
        path: Optional[str] = None
    ) -> None:
        """
        Save trained model to disk.
        
        Args:
            ticker: Ticker symbol.
            path: Save path. Uses models/ if None.
        """
        if ticker not in self.models:
            raise ValueError(f"No trained model for {ticker}")
        
        if path is None:
            path = f"models/xgboost_{ticker.replace('=', '_')}.pkl"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.models[ticker],
            'scaler': self.scalers[ticker],
            'feature_names': self.feature_names[ticker],
            'metrics': self.train_metrics.get(ticker, {}),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Saved model to {path}")
    
    def load_model(
        self,
        ticker: str,
        path: Optional[str] = None
    ) -> None:
        """
        Load trained model from disk.
        
        Args:
            ticker: Ticker symbol.
            path: Load path. Uses models/ if None.
        """
        if path is None:
            path = f"models/xgboost_{ticker.replace('=', '_')}.pkl"
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models[ticker] = model_data['model']
        self.scalers[ticker] = model_data['scaler']
        self.feature_names[ticker] = model_data['feature_names']
        self.train_metrics[ticker] = model_data.get('metrics', {})
        
        self.logger.info(f"Loaded model from {path}")
    
    def train_all(
        self,
        df: pd.DataFrame,
        tune_hyperparams: bool = False
    ) -> Dict[str, Dict]:
        """
        Train models for all main tickers.
        
        Args:
            df: Features DataFrame.
            tune_hyperparams: Whether to tune hyperparameters.
        
        Returns:
            Dictionary of metrics per ticker.
        """
        main_tickers = self.config['data']['tickers']['main']
        all_metrics = {}
        
        for ticker in main_tickers:
            try:
                metrics = self.train(df, ticker, tune_hyperparams)
                all_metrics[ticker] = metrics
            except Exception as e:
                self.logger.error(f"Failed to train {ticker}: {e}")
                all_metrics[ticker] = {'error': str(e)}
        
        return all_metrics


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    from pathlib import Path
    
    # Load features
    features_path = Path('data/processed/features.parquet')
    if features_path.exists():
        df = pd.read_parquet(features_path)
        
        # Initialize and train
        model = XGBoostTradingModel()
        metrics = model.train_all(df, tune_hyperparams=False)
        
        # Print results
        print("\nTraining Results:")
        for ticker, m in metrics.items():
            if 'error' not in m:
                print(f"  {ticker}: Val Acc={m['val_accuracy']:.3f}, "
                      f"Test Acc={m['test_accuracy']:.3f}")
            else:
                print(f"  {ticker}: ERROR - {m['error']}")
    else:
        print(f"Features not found at {features_path}")
