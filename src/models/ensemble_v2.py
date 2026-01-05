"""
Ensemble V2 - Advanced ensemble with meta-learning and regime adaptation.

This module provides:
- Meta-learner approach (LightGBM on base model outputs)
- Confidence calibration (Platt scaling)
- Regime-weighted voting
- Dynamic weighting based on recent performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

logger = logging.getLogger(__name__)


class AdvancedEnsemble:
    """
    Advanced ensemble with meta-learning capabilities.
    
    Improvements over simple voting:
    - Regime-weighted voting (different weights per regime)
    - Confidence calibration (Platt scaling)
    - Disagreement handling (when models conflict)
    - Performance-weighted voting (recent accuracy)
    - Meta-learner to combine base model outputs
    
    Features:
    - Learns optimal combination from data
    - Adapts weights based on market conditions
    - Calibrated probability outputs
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize advanced ensemble.
        
        Args:
            config: Configuration dictionary with ensemble parameters
        """
        self.config = config or {}
        self.ensemble_config = self.config.get('ensemble_v2', {
            'meta_learner': 'lightgbm',
            'calibration': 'platt',
            'dynamic_weight_window': 20
        })
        
        self.base_models = {}
        self.meta_learner = None
        self.calibrator = None
        self.regime_weights = {}
        self.performance_history = []
        self.is_trained = False
        
    def _extract_model(self, model: Any) -> Any:
        """
        Extract actual model object from wrapper (dict, etc.).
        
        Args:
            model: Model object or dictionary containing model
            
        Returns:
            Actual model object with predict/predict_proba methods
        """
        if isinstance(model, dict):
            # Try common keys for wrapped models
            for key in ['model', 'classifier', 'estimator', 'booster', 'xgb_model', 'lgb_model']:
                if key in model:
                    return self._extract_model(model[key])  # Recursive in case nested
            # If no known key, log warning
            logger.warning(f"Model is a dict with keys: {list(model.keys())}. Could not extract model object.")
        return model
    
    def add_base_model(self, name: str, model: Any):
        """
        Add a base model to the ensemble.
        
        Args:
            name: Model name/identifier
            model: Trained model with predict and predict_proba methods
        """
        # Extract actual model if wrapped in a dictionary
        actual_model = self._extract_model(model)
        self.base_models[name] = actual_model
        logger.info(f"Added base model: {name} (type: {type(actual_model).__name__})")
    
    def get_base_predictions(
        self,
        X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get predictions from all base models.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions DataFrame, probabilities DataFrame)
        """
        predictions = {}
        probabilities = {}
        
        X_filled = X.fillna(0)
        
        for name, model_or_dict in self.base_models.items():
            try:
                # Try to extract model if it's still a dict (e.g., loaded from pickle)
                model = self._extract_model(model_or_dict)
                
                pred = model.predict(X_filled)
                prob = model.predict_proba(X_filled)[:, 1]
                predictions[name] = pred
                probabilities[name] = prob
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {e}")
                predictions[name] = np.zeros(len(X))
                probabilities[name] = np.ones(len(X)) * 0.5
        
        return pd.DataFrame(predictions), pd.DataFrame(probabilities)
    
    def train_meta_learner(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_col: Optional[str] = None
    ) -> Dict:
        """
        Train meta-learner on base model outputs.
        
        Args:
            X: Feature matrix for base models
            y: Target labels
            regime_col: Optional regime column in X for regime-aware training
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training meta-learner...")
        
        # Get base model predictions
        pred_df, prob_df = self.get_base_predictions(X)
        
        # Create meta-features
        meta_features = prob_df.copy()
        
        # Add prediction agreement features
        meta_features['agreement'] = (pred_df.nunique(axis=1) == 1).astype(int)
        meta_features['mean_prob'] = prob_df.mean(axis=1)
        meta_features['std_prob'] = prob_df.std(axis=1)
        meta_features['max_prob'] = prob_df.max(axis=1)
        meta_features['min_prob'] = prob_df.min(axis=1)
        meta_features['prob_range'] = meta_features['max_prob'] - meta_features['min_prob']
        
        # Add regime features if available
        if regime_col and regime_col in X.columns:
            meta_features['regime'] = X[regime_col].values
            # One-hot encode regime
            regime_dummies = pd.get_dummies(meta_features['regime'], prefix='regime')
            meta_features = pd.concat([meta_features.drop('regime', axis=1), regime_dummies], axis=1)
        
        # Handle NaN
        meta_features = meta_features.fillna(0)
        y_clean = y.fillna(0).astype(int)
        
        # Reset indices to align properly (meta_features may have fresh indices)
        meta_features = meta_features.reset_index(drop=True)
        y_clean = y_clean.reset_index(drop=True)
        
        # Ensure same length
        min_len = min(len(meta_features), len(y_clean))
        meta_features = meta_features.iloc[:min_len]
        y_clean = y_clean.iloc[:min_len]
        
        # Validate we have data
        if len(meta_features) == 0:
            raise ValueError(
                f"No valid samples for meta-learner training. "
                f"Original X shape: {X.shape}, y shape: {y.shape}"
            )
        
        logger.info(f"  Training on {len(meta_features)} samples with {len(meta_features.columns)} features")
        
        # Train meta-learner
        meta_learner_type = self.ensemble_config.get('meta_learner', 'lightgbm')
        
        if meta_learner_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.meta_learner = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
        else:
            # Fallback to logistic regression
            self.meta_learner = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        
        # Train
        self.meta_learner.fit(meta_features, y_clean)
        
        # Evaluate
        meta_pred = self.meta_learner.predict(meta_features)
        meta_prob = self.meta_learner.predict_proba(meta_features)[:, 1]
        
        train_accuracy = accuracy_score(y_clean, meta_pred)
        
        results = {
            'train_accuracy': train_accuracy,
            'meta_features': list(meta_features.columns),
            'n_samples': len(y_clean),
            'meta_learner_type': type(self.meta_learner).__name__
        }
        
        self.is_trained = True
        
        logger.info(f"✅ Meta-learner trained")
        logger.info(f"   Train accuracy: {train_accuracy:.4f}")
        logger.info(f"   Meta-features: {len(meta_features.columns)}")
        
        return results
    
    def calibrate_confidence(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """
        Calibrate model probabilities using Platt scaling.
        
        Args:
            X: Feature matrix
            y: True labels
        """
        logger.info("Calibrating confidence scores...")
        
        # Get ensemble probabilities
        _, prob_df = self.get_base_predictions(X)
        ensemble_probs = prob_df.mean(axis=1).values.reshape(-1, 1)
        
        # Clean data
        y_clean = y.fillna(0).astype(int)
        valid_idx = ~np.isnan(ensemble_probs).flatten()
        ensemble_probs = ensemble_probs[valid_idx]
        y_clean = y_clean.iloc[valid_idx] if hasattr(y_clean, 'iloc') else y_clean[valid_idx]
        
        # Train calibrator (Platt scaling via logistic regression)
        self.calibrator = LogisticRegression(C=1.0, max_iter=1000)
        self.calibrator.fit(ensemble_probs, y_clean)
        
        logger.info("✅ Confidence calibration complete")
    
    def calibrate_probability(self, prob: np.ndarray) -> np.ndarray:
        """
        Apply calibration to probability scores.
        
        Args:
            prob: Raw probability scores
            
        Returns:
            Calibrated probabilities
        """
        if self.calibrator is None:
            return prob
        
        prob_reshaped = prob.reshape(-1, 1) if prob.ndim == 1 else prob
        return self.calibrator.predict_proba(prob_reshaped)[:, 1]
    
    def train_regime_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_col: str
    ) -> Dict:
        """
        Learn optimal weights for each regime.
        
        Args:
            X: Feature matrix
            y: True labels
            regime_col: Column name for regime classification
            
        Returns:
            Dictionary with regime-specific weights
        """
        logger.info("Learning regime-specific weights...")
        
        if regime_col not in X.columns:
            logger.warning(f"Regime column {regime_col} not found")
            return {}
        
        pred_df, prob_df = self.get_base_predictions(X)
        
        self.regime_weights = {}
        
        for regime in X[regime_col].dropna().unique():
            regime_mask = X[regime_col] == regime
            
            if regime_mask.sum() < 50:
                continue
            
            regime_y = y[regime_mask]
            
            # Find best weights for this regime
            best_weights = {}
            best_accuracy = 0
            
            # Grid search over weight combinations
            for w1 in np.arange(0.0, 1.1, 0.1):
                for w2 in np.arange(0.0, 1.1 - w1, 0.1):
                    w3 = 1.0 - w1 - w2
                    weights = [w1, w2, w3][:len(self.base_models)]
                    
                    if len(weights) < len(self.base_models):
                        weights.extend([0] * (len(self.base_models) - len(weights)))
                    
                    # Normalize
                    total = sum(weights)
                    if total > 0:
                        weights = [w / total for w in weights]
                    
                    # Combine predictions
                    combined_prob = np.zeros(regime_mask.sum())
                    for i, (name, _) in enumerate(self.base_models.items()):
                        combined_prob += weights[i] * prob_df.loc[regime_mask, name].values
                    
                    combined_pred = (combined_prob >= 0.5).astype(int)
                    accuracy = accuracy_score(regime_y.fillna(0), combined_pred)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_weights = dict(zip(self.base_models.keys(), weights))
            
            self.regime_weights[str(regime)] = {
                'weights': best_weights,
                'accuracy': best_accuracy,
                'n_samples': regime_mask.sum()
            }
            
            logger.info(f"  {regime}: accuracy={best_accuracy:.4f}, weights={best_weights}")
        
        return self.regime_weights
    
    def dynamic_weighting(
        self,
        recent_predictions: List[Tuple[Dict[str, int], int]],
        window: int = None
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights based on recent performance.
        
        Args:
            recent_predictions: List of (model_predictions, actual) tuples
            window: Number of recent predictions to consider
            
        Returns:
            Dictionary of model weights
        """
        window = window or self.ensemble_config.get('dynamic_weight_window', 20)
        
        if not recent_predictions:
            # Equal weights if no history
            n_models = len(self.base_models)
            return {name: 1.0 / n_models for name in self.base_models.keys()}
        
        # Use last N predictions
        recent = recent_predictions[-window:]
        
        # Calculate accuracy for each model
        model_correct = {name: 0 for name in self.base_models.keys()}
        
        for model_preds, actual in recent:
            for name, pred in model_preds.items():
                if pred == actual:
                    model_correct[name] += 1
        
        # Convert to weights
        total_correct = sum(model_correct.values())
        
        if total_correct == 0:
            n_models = len(self.base_models)
            return {name: 1.0 / n_models for name in self.base_models.keys()}
        
        weights = {
            name: correct / total_correct
            for name, correct in model_correct.items()
        }
        
        return weights
    
    def combine_with_regime(
        self,
        X: pd.DataFrame,
        regime_col: str = None,
        use_meta: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine predictions using regime-specific weights.
        
        Args:
            X: Feature matrix
            regime_col: Column with regime classification
            use_meta: Whether to use meta-learner if available
            
        Returns:
            Tuple of (predictions, confidence scores)
        """
        pred_df, prob_df = self.get_base_predictions(X)
        
        # If meta-learner available and trained, use it
        if use_meta and self.meta_learner is not None and self.is_trained:
            return self._predict_with_meta(X, regime_col)
        
        # Otherwise use weighted voting
        final_probs = np.zeros(len(X))
        
        if regime_col and regime_col in X.columns and self.regime_weights:
            # Regime-specific weights
            for i, regime in enumerate(X[regime_col]):
                regime_str = str(regime)
                if regime_str in self.regime_weights:
                    weights = self.regime_weights[regime_str]['weights']
                else:
                    # Default equal weights
                    weights = {name: 1.0 / len(self.base_models) for name in self.base_models.keys()}
                
                for name, w in weights.items():
                    if name in prob_df.columns:
                        final_probs[i] += w * prob_df.iloc[i][name]
        else:
            # Equal weights
            final_probs = prob_df.mean(axis=1).values
        
        # Apply calibration
        if self.calibrator is not None:
            final_probs = self.calibrate_probability(final_probs)
        
        # Convert to predictions
        predictions = (final_probs >= 0.5).astype(int)
        confidence = np.abs(final_probs - 0.5) * 2 * 100  # Scale to 0-100
        
        return predictions, confidence
    
    def _predict_with_meta(
        self,
        X: pd.DataFrame,
        regime_col: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using meta-learner.
        
        Args:
            X: Feature matrix
            regime_col: Optional regime column
            
        Returns:
            Tuple of (predictions, confidence)
        """
        _, prob_df = self.get_base_predictions(X)
        
        # Create meta-features (same as training)
        meta_features = prob_df.copy()
        pred_df, _ = self.get_base_predictions(X)
        
        meta_features['agreement'] = (pred_df.nunique(axis=1) == 1).astype(int)
        meta_features['mean_prob'] = prob_df.mean(axis=1)
        meta_features['std_prob'] = prob_df.std(axis=1)
        meta_features['max_prob'] = prob_df.max(axis=1)
        meta_features['min_prob'] = prob_df.min(axis=1)
        meta_features['prob_range'] = meta_features['max_prob'] - meta_features['min_prob']
        
        if regime_col and regime_col in X.columns:
            meta_features['regime'] = X[regime_col].values
            regime_dummies = pd.get_dummies(meta_features['regime'], prefix='regime')
            meta_features = pd.concat([meta_features.drop('regime', axis=1), regime_dummies], axis=1)
        
        meta_features = meta_features.fillna(0)
        
        # Align columns with training
        # (This is a simplification - in production, save and load column names)
        
        # Predict
        predictions = self.meta_learner.predict(meta_features)
        probabilities = self.meta_learner.predict_proba(meta_features)[:, 1]
        
        confidence = np.abs(probabilities - 0.5) * 2 * 100
        
        return predictions, confidence
    
    def resolve_conflicts(
        self,
        pred_df: pd.DataFrame,
        prob_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resolve conflicts when models disagree.
        
        Strategy:
        - If models agree: high confidence
        - If models disagree: take higher confidence prediction, reduce overall confidence
        
        Args:
            pred_df: DataFrame of predictions
            prob_df: DataFrame of probabilities
            
        Returns:
            Tuple of (final predictions, adjusted confidence)
        """
        n_samples = len(pred_df)
        final_preds = np.zeros(n_samples)
        final_conf = np.zeros(n_samples)
        
        conf_penalty = self.config.get('models', {}).get('ensemble', {}).get('disagreement_penalty', 20)
        conf_bonus = self.config.get('models', {}).get('ensemble', {}).get('agreement_bonus', 10)
        
        for i in range(n_samples):
            preds = pred_df.iloc[i].values
            probs = prob_df.iloc[i].values
            
            # Check agreement
            if len(set(preds)) == 1:
                # All agree
                final_preds[i] = preds[0]
                final_conf[i] = min(100, np.mean(np.abs(probs - 0.5) * 200) + conf_bonus)
            else:
                # Disagreement - take highest confidence
                confidences = np.abs(probs - 0.5)
                best_idx = np.argmax(confidences)
                final_preds[i] = preds[best_idx]
                final_conf[i] = max(0, np.abs(probs[best_idx] - 0.5) * 200 - conf_penalty)
        
        return final_preds.astype(int), final_conf
    
    def predict(
        self,
        X: pd.DataFrame,
        regime_col: str = None,
        return_confidence: bool = True
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature matrix
            regime_col: Optional regime column for regime-aware prediction
            return_confidence: Whether to return confidence scores
            
        Returns:
            Predictions (and optionally confidence scores)
        """
        predictions, confidence = self.combine_with_regime(X, regime_col)
        
        if return_confidence:
            return predictions, confidence
        return predictions
    
    def save(self, output_dir: str):
        """
        Save ensemble to disk.
        
        Args:
            output_dir: Directory to save ensemble components
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save base models
        for name, model in self.base_models.items():
            with open(output_path / f'base_model_{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save meta-learner
        if self.meta_learner is not None:
            with open(output_path / 'meta_learner.pkl', 'wb') as f:
                pickle.dump(self.meta_learner, f)
        
        # Save calibrator
        if self.calibrator is not None:
            with open(output_path / 'calibrator.pkl', 'wb') as f:
                pickle.dump(self.calibrator, f)
        
        # Save regime weights
        import json
        with open(output_path / 'regime_weights.json', 'w') as f:
            json.dump(self.regime_weights, f, indent=2)
        
        logger.info(f"✅ Saved ensemble to {output_dir}")
    
    def load(self, input_dir: str):
        """
        Load ensemble from disk.
        
        Args:
            input_dir: Directory containing ensemble components
        """
        input_path = Path(input_dir)
        
        # Load base models
        for model_file in input_path.glob('base_model_*.pkl'):
            name = model_file.stem.replace('base_model_', '')
            with open(model_file, 'rb') as f:
                self.base_models[name] = pickle.load(f)
        
        # Load meta-learner
        meta_path = input_path / 'meta_learner.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                self.meta_learner = pickle.load(f)
            self.is_trained = True
        
        # Load calibrator
        calib_path = input_path / 'calibrator.pkl'
        if calib_path.exists():
            with open(calib_path, 'rb') as f:
                self.calibrator = pickle.load(f)
        
        # Load regime weights
        import json
        weights_path = input_path / 'regime_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.regime_weights = json.load(f)
        
        logger.info(f"✅ Loaded ensemble from {input_dir}")
        logger.info(f"   Base models: {list(self.base_models.keys())}")


def main():
    """Main function to train advanced ensemble."""
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load features
    features_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'features.parquet'
    df = pd.read_parquet(features_path)
    
    # Load existing models
    models_dir = Path(__file__).parent.parent.parent / 'models'
    
    # Initialize ensemble
    ensemble = AdvancedEnsemble(config)
    
    # Load XGBoost models
    for ticker in ['GBPUSD_X', 'EURUSD_X', 'GC_F']:
        model_path = models_dir / f'xgboost_{ticker}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            ensemble.add_base_model(f'xgboost_{ticker}', model)
    
    if not ensemble.base_models:
        print("No base models found! Train XGBoost models first.")
        return
    
    # Get features and target
    feature_cols = [c for c in df.columns if not any(x in c.lower() for x in ['target', 'date', 'ticker', 'regime'])]
    target_col = 'GBPUSD_X_target_next_day'
    
    if target_col not in df.columns:
        target_cols = [c for c in df.columns if 'target' in c.lower()]
        if target_cols:
            target_col = target_cols[0]
        else:
            print("No target column found!")
            return
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Train meta-learner
    results = ensemble.train_meta_learner(X, y)
    print(f"\n✅ Meta-learner trained: {results}")
    
    # Calibrate confidence
    ensemble.calibrate_confidence(X, y)
    
    # Save ensemble
    output_dir = models_dir / 'ensemble_v2'
    ensemble.save(str(output_dir))
    
    print(f"\n✅ Advanced Ensemble Complete!")
    print(f"   Saved to: {output_dir}")


if __name__ == "__main__":
    main()
