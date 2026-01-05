"""
Walk-Forward Validation Module - Robust out-of-sample testing.

This module provides:
- Rolling window train/test splits
- Performance tracking across periods
- Stability analysis
- Regime-specific performance breakdown
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for trading models.
    
    Logic:
    - Train on N days, test on next M days
    - Roll forward by M days, retrain, test
    - Track performance stability across periods
    - Identify regimes where model fails
    
    Features:
    - No look-ahead bias (strict time ordering)
    - Multiple metrics tracked per fold
    - Automatic model retraining
    - Performance degradation detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize walk-forward validator.
        
        Args:
            config: Configuration dictionary with walk-forward parameters
        """
        self.config = config or {}
        self.wf_config = self.config.get('walk_forward', {
            'train_days': 500,
            'test_days': 63,
            'step_days': 63,
            'min_folds': 4
        })
        
        self.results = []
        self.fold_models = []
        self.equity_curves = []
        
    def run_walk_forward(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model_params: Optional[Dict] = None,
        ticker: str = "default"
    ) -> Dict:
        """
        Execute full walk-forward analysis.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            model_params: XGBoost parameters (uses defaults if None)
            ticker: Ticker name for logging
            
        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Starting walk-forward validation for {ticker}")
        
        train_days = self.wf_config.get('train_days', 500)
        test_days = self.wf_config.get('test_days', 63)
        step_days = self.wf_config.get('step_days', 63)
        min_folds = self.wf_config.get('min_folds', 4)
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
        
        df = df.sort_index()
        
        # Default model params
        if model_params is None:
            model_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 5,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'use_label_encoder': False,
                'verbosity': 0,
                'random_state': 42
            }
        
        # Calculate number of folds
        total_days = len(df)
        n_folds = max(1, (total_days - train_days) // step_days)
        
        if n_folds < min_folds:
            logger.warning(f"Only {n_folds} folds possible (minimum: {min_folds})")
        
        logger.info(f"  Train window: {train_days} days")
        logger.info(f"  Test window: {test_days} days")
        logger.info(f"  Step: {step_days} days")
        logger.info(f"  Number of folds: {n_folds}")
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        for fold in range(n_folds):
            # Calculate indices
            train_start = fold * step_days
            train_end = train_start + train_days
            test_start = train_end
            test_end = min(test_start + test_days, total_days)
            
            if test_end <= test_start:
                break
            
            # Split data
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]
            
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data[target_col]
            X_test = test_data[feature_cols].fillna(0)
            y_test = test_data[target_col]
            
            # Skip if insufficient data
            if len(X_train) < 100 or len(X_test) < 10:
                continue
            
            # Remove NaN targets
            valid_train = ~y_train.isna()
            valid_test = ~y_test.isna()
            
            X_train = X_train[valid_train]
            y_train = y_train[valid_train]
            X_test = X_test[valid_test]
            y_test = y_test[valid_test]
            
            if len(y_train) < 50 or len(y_test) < 5:
                continue
            
            # Train model
            model = xgb.XGBClassifier(**model_params)
            model.fit(X_train, y_train, verbose=False)
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            fold_metrics = {
                'fold': fold,
                'train_start': train_data.index[0].strftime('%Y-%m-%d') if hasattr(train_data.index[0], 'strftime') else str(train_data.index[0]),
                'train_end': train_data.index[-1].strftime('%Y-%m-%d') if hasattr(train_data.index[-1], 'strftime') else str(train_data.index[-1]),
                'test_start': test_data.index[0].strftime('%Y-%m-%d') if hasattr(test_data.index[0], 'strftime') else str(test_data.index[0]),
                'test_end': test_data.index[-1].strftime('%Y-%m-%d') if hasattr(test_data.index[-1], 'strftime') else str(test_data.index[-1]),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'win_rate': np.mean(y_pred == y_test),
                'pred_positive_rate': np.mean(y_pred),
                'actual_positive_rate': np.mean(y_test)
            }
            
            fold_results.append(fold_metrics)
            self.fold_models.append(model)
            
            # Store predictions for later analysis
            all_predictions.extend(y_pred.tolist())
            all_actuals.extend(y_test.tolist())
            all_dates.extend([str(d) for d in test_data.index[valid_test]])
            
            logger.info(f"  Fold {fold}: Accuracy={fold_metrics['accuracy']:.4f}, "
                       f"Period: {fold_metrics['test_start']} to {fold_metrics['test_end']}")
        
        # Aggregate results
        if not fold_results:
            logger.error("No valid folds completed!")
            return {'error': 'No valid folds'}
        
        results_df = pd.DataFrame(fold_results)
        
        aggregated = {
            'ticker': ticker,
            'n_folds': len(fold_results),
            'mean_accuracy': results_df['accuracy'].mean(),
            'std_accuracy': results_df['accuracy'].std(),
            'min_accuracy': results_df['accuracy'].min(),
            'max_accuracy': results_df['accuracy'].max(),
            'mean_precision': results_df['precision'].mean(),
            'mean_recall': results_df['recall'].mean(),
            'mean_f1': results_df['f1'].mean(),
            'mean_win_rate': results_df['win_rate'].mean(),
            'total_test_samples': results_df['test_size'].sum(),
            'fold_results': fold_results,
            'all_predictions': all_predictions,
            'all_actuals': all_actuals,
            'all_dates': all_dates
        }
        
        self.results.append(aggregated)
        
        logger.info(f"✅ Walk-forward complete for {ticker}")
        logger.info(f"   Mean accuracy: {aggregated['mean_accuracy']:.4f} ± {aggregated['std_accuracy']:.4f}")
        logger.info(f"   Accuracy range: [{aggregated['min_accuracy']:.4f}, {aggregated['max_accuracy']:.4f}]")
        
        return aggregated
    
    def analyze_stability(self, results: Dict = None) -> Dict:
        """
        Analyze performance stability across folds.
        
        Args:
            results: Walk-forward results (uses last if None)
            
        Returns:
            Dictionary with stability metrics
        """
        if results is None:
            if not self.results:
                raise ValueError("No results available. Run walk_forward first.")
            results = self.results[-1]
        
        fold_results = results.get('fold_results', [])
        if not fold_results:
            return {'error': 'No fold results'}
        
        accuracies = [f['accuracy'] for f in fold_results]
        
        # Calculate stability metrics
        stability = {
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'cv': np.std(accuracies) / np.mean(accuracies) if np.mean(accuracies) > 0 else 0,  # Coefficient of variation
            'min': np.min(accuracies),
            'max': np.max(accuracies),
            'range': np.max(accuracies) - np.min(accuracies),
            'trend': np.polyfit(range(len(accuracies)), accuracies, 1)[0],  # Linear trend
            'n_folds_above_50pct': sum(1 for a in accuracies if a > 0.5),
            'pct_folds_above_50pct': sum(1 for a in accuracies if a > 0.5) / len(accuracies) * 100
        }
        
        # Determine if model is degrading
        trend_slope = stability['trend']
        if trend_slope < -0.01:  # Significant negative trend
            stability['status'] = 'DEGRADING'
            stability['warning'] = 'Model accuracy declining over time'
        elif stability['cv'] > 0.2:  # High variability
            stability['status'] = 'UNSTABLE'
            stability['warning'] = 'High variability in performance'
        elif stability['mean'] < 0.5:
            stability['status'] = 'POOR'
            stability['warning'] = 'Average accuracy below 50%'
        else:
            stability['status'] = 'STABLE'
            stability['warning'] = None
        
        logger.info(f"Stability Analysis:")
        logger.info(f"  Status: {stability['status']}")
        logger.info(f"  CV: {stability['cv']:.4f}")
        logger.info(f"  Trend: {stability['trend']:.6f}")
        if stability['warning']:
            logger.warning(f"  ⚠️ {stability['warning']}")
        
        return stability
    
    def detect_regime_failures(
        self,
        results: Dict,
        regime_col: str = None,
        df: pd.DataFrame = None
    ) -> Dict:
        """
        Identify which market regimes cause model failures.
        
        Args:
            results: Walk-forward results
            regime_col: Column name for regime classification
            df: Original DataFrame with regime data
            
        Returns:
            Dictionary with regime-specific performance
        """
        fold_results = results.get('fold_results', [])
        
        # Find failing folds (accuracy < 45%)
        failing_folds = [f for f in fold_results if f['accuracy'] < 0.45]
        good_folds = [f for f in fold_results if f['accuracy'] >= 0.55]
        
        analysis = {
            'n_failing_folds': len(failing_folds),
            'n_good_folds': len(good_folds),
            'failing_periods': [(f['test_start'], f['test_end'], f['accuracy']) for f in failing_folds],
            'good_periods': [(f['test_start'], f['test_end'], f['accuracy']) for f in good_folds]
        }
        
        # If regime data available, analyze by regime
        if regime_col and df is not None and regime_col in df.columns:
            regime_performance = {}
            
            all_dates = results.get('all_dates', [])
            all_predictions = results.get('all_predictions', [])
            all_actuals = results.get('all_actuals', [])
            
            if all_dates and all_predictions and all_actuals:
                # Create prediction DataFrame
                pred_df = pd.DataFrame({
                    'date': pd.to_datetime(all_dates),
                    'prediction': all_predictions,
                    'actual': all_actuals
                })
                pred_df = pred_df.set_index('date')
                
                # Merge with regime data
                if isinstance(df.index, pd.DatetimeIndex):
                    pred_df = pred_df.join(df[[regime_col]], how='left')
                    
                    for regime in pred_df[regime_col].dropna().unique():
                        regime_mask = pred_df[regime_col] == regime
                        if regime_mask.sum() > 10:
                            regime_preds = pred_df.loc[regime_mask, 'prediction']
                            regime_actuals = pred_df.loc[regime_mask, 'actual']
                            regime_performance[str(regime)] = {
                                'accuracy': accuracy_score(regime_actuals, regime_preds),
                                'n_samples': len(regime_preds)
                            }
                    
                    analysis['regime_performance'] = regime_performance
        
        logger.info(f"Regime Failure Analysis:")
        logger.info(f"  Failing folds: {analysis['n_failing_folds']}")
        logger.info(f"  Good folds: {analysis['n_good_folds']}")
        
        return analysis
    
    def plot_rolling_metrics(
        self,
        results: Dict = None,
        save_path: str = None
    ):
        """
        Plot rolling performance metrics across folds.
        
        Args:
            results: Walk-forward results
            save_path: Path to save the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return
        
        if results is None:
            if not self.results:
                raise ValueError("No results available")
            results = self.results[-1]
        
        fold_results = results.get('fold_results', [])
        if not fold_results:
            return
        
        # Extract metrics
        folds = [f['fold'] for f in fold_results]
        accuracies = [f['accuracy'] for f in fold_results]
        precisions = [f['precision'] for f in fold_results]
        recalls = [f['recall'] for f in fold_results]
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy over folds
        ax1 = axes[0, 0]
        ax1.plot(folds, accuracies, 'b-o', linewidth=2, markersize=8)
        ax1.axhline(y=0.5, color='r', linestyle='--', label='50% baseline')
        ax1.axhline(y=np.mean(accuracies), color='g', linestyle=':', label=f'Mean: {np.mean(accuracies):.3f}')
        ax1.fill_between(folds, 
                         [np.mean(accuracies) - np.std(accuracies)] * len(folds),
                         [np.mean(accuracies) + np.std(accuracies)] * len(folds),
                         alpha=0.2, color='green')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Across Walk-Forward Folds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision and Recall
        ax2 = axes[0, 1]
        ax2.plot(folds, precisions, 'g-o', label='Precision', linewidth=2)
        ax2.plot(folds, recalls, 'orange', marker='s', label='Recall', linewidth=2)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Score')
        ax2.set_title('Precision & Recall Across Folds')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Accuracy histogram
        ax3 = axes[1, 0]
        ax3.hist(accuracies, bins=10, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0.5, color='r', linestyle='--', label='50% baseline')
        ax3.axvline(x=np.mean(accuracies), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Fold Accuracies')
        ax3.legend()
        
        # Cumulative accuracy trend
        ax4 = axes[1, 1]
        cumulative_acc = np.cumsum(accuracies) / np.arange(1, len(accuracies) + 1)
        ax4.plot(folds, cumulative_acc, 'purple', linewidth=2)
        ax4.axhline(y=0.5, color='r', linestyle='--')
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('Cumulative Mean Accuracy')
        ax4.set_title('Cumulative Performance Trend')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        plt.close()
    
    def get_best_fold_model(self) -> Optional[xgb.XGBClassifier]:
        """
        Get the model from the best performing fold.
        
        Returns:
            XGBoost model from best fold
        """
        if not self.results or not self.fold_models:
            return None
        
        # Find best fold
        last_results = self.results[-1]
        fold_results = last_results.get('fold_results', [])
        
        if not fold_results:
            return None
        
        best_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]['accuracy'])
        
        if best_idx < len(self.fold_models):
            return self.fold_models[best_idx]
        
        return None
    
    def save_results(self, output_dir: str):
        """
        Save walk-forward results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results (without models)
        results_copy = []
        for r in self.results:
            r_copy = {k: v for k, v in r.items() if k not in ['all_predictions', 'all_actuals', 'all_dates']}
            results_copy.append(r_copy)
        
        import json
        with open(output_path / 'walk_forward_results.json', 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        # Save best model
        best_model = self.get_best_fold_model()
        if best_model:
            with open(output_path / 'best_wf_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
        
        logger.info(f"✅ Saved walk-forward results to {output_dir}")


def main():
    """Main function to run walk-forward validation."""
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
    
    # Initialize validator
    validator = WalkForwardValidator(config)
    
    # Get feature columns
    feature_cols = [c for c in df.columns if not any(x in c.lower() for x in ['target', 'date', 'ticker'])]
    
    # Run for each ticker
    for ticker in ['GBPUSD_X', 'EURUSD_X', 'GC_F']:
        target_col = f'{ticker}_target_next_day'
        
        if target_col not in df.columns:
            continue
        
        results = validator.run_walk_forward(
            df, feature_cols, target_col, ticker=ticker
        )
        
        # Analyze stability
        stability = validator.analyze_stability(results)
        
        # Plot metrics
        plot_path = Path(__file__).parent.parent / 'results' / 'charts' / f'wf_{ticker}.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        validator.plot_rolling_metrics(results, str(plot_path))
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results' / 'walk_forward'
    validator.save_results(str(output_dir))
    
    print("\n✅ Walk-Forward Validation Complete!")


if __name__ == "__main__":
    main()
