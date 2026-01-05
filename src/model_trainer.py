"""
Model Trainer for Forex Signal Model.

Orchestrates training of all models and generates comparison reports.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils import get_logger, load_config, timer_decorator
from src.models.technical_rules import TechnicalRulesSystem
from src.models.ml_models import XGBoostTradingModel
from src.models.ensemble import EnsembleModel
from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import calculate_performance_metrics, format_metrics_report
from src.backtesting.visualizations import generate_performance_report


class ModelTrainer:
    """
    Train all models and save artifacts.
    
    Orchestrates:
    1. Technical rules signal generation
    2. XGBoost training and prediction
    3. Ensemble combination
    4. Backtesting of all models
    5. Performance comparison
    
    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        technical_system: TechnicalRulesSystem instance.
        ml_model: XGBoostTradingModel instance.
        ensemble: EnsembleModel instance.
        backtest_engine: BacktestEngine instance.
    
    Example:
        >>> trainer = ModelTrainer('config/config.yaml')
        >>> results = trainer.train_and_evaluate('data/processed/features.parquet')
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the ModelTrainer.
        
        Args:
            config_path: Path to configuration YAML file.
        """
        self.config = load_config(config_path)
        self.config_path = config_path
        self.logger = get_logger('forex_signal_model.model_trainer')
        
        # Initialize models
        self.technical_system = TechnicalRulesSystem(config_path)
        self.ml_model = XGBoostTradingModel(config_path)
        self.ensemble = EnsembleModel(config_path)
        self.backtest_engine = BacktestEngine(config_path)
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
        self.logger.info("ModelTrainer initialized")
    
    @timer_decorator
    def load_data(self, data_path: str = 'data/processed/features.parquet') -> pd.DataFrame:
        """
        Load feature data.
        
        Args:
            data_path: Path to features parquet file.
        
        Returns:
            Features DataFrame.
        """
        self.logger.info(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    @timer_decorator
    def train_technical_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate signals using technical rules.
        
        Args:
            df: Features DataFrame.
        
        Returns:
            Results dictionary with signals.
        """
        self.logger.info("Generating technical rules signals...")
        
        signals = self.technical_system.generate_all_signals(df)
        summary = self.technical_system.get_signal_summary(signals)
        
        self.results['technical_signals'] = signals
        self.results['technical_summary'] = summary
        
        self.logger.info(f"Technical rules: {summary['buy_signals']} BUY, "
                        f"{summary['sell_signals']} SELL signals")
        
        return {'signals': signals, 'summary': summary}
    
    @timer_decorator
    def train_xgboost(
        self,
        df: pd.DataFrame,
        tune_hyperparams: bool = False
    ) -> Dict[str, Any]:
        """
        Train XGBoost models.
        
        Args:
            df: Features DataFrame.
            tune_hyperparams: Whether to tune hyperparameters.
        
        Returns:
            Training metrics per ticker.
        """
        self.logger.info("Training XGBoost models...")
        
        metrics = self.ml_model.train_all(df, tune_hyperparams)
        
        # Generate predictions
        signals = self.ml_model.predict_all(df)
        
        self.results['xgboost_metrics'] = metrics
        self.results['xgboost_signals'] = signals
        
        for ticker, m in metrics.items():
            if 'error' not in m:
                self.logger.info(f"XGBoost {ticker}: Val Acc={m['val_accuracy']:.3f}, "
                               f"Test Acc={m['test_accuracy']:.3f}")
        
        return metrics
    
    @timer_decorator
    def train_ensemble(
        self,
        df: pd.DataFrame,
        tune_hyperparams: bool = False
    ) -> Dict[str, Any]:
        """
        Train ensemble model.
        
        Args:
            df: Features DataFrame.
            tune_hyperparams: Whether to tune XGBoost hyperparameters.
        
        Returns:
            Ensemble training results.
        """
        self.logger.info("Training ensemble model...")
        
        # This trains XGBoost internally
        ensemble_results = self.ensemble.train(df, tune_hyperparams)
        
        # Generate predictions
        signals = self.ensemble.predict(df)
        analysis = self.ensemble.get_signal_analysis(signals)
        
        self.results['ensemble_signals'] = signals
        self.results['ensemble_analysis'] = analysis
        
        self.logger.info(f"Ensemble: {analysis['buy_count']} BUY, {analysis['sell_count']} SELL, "
                        f"Agreement rate: {analysis['agreement_rate']:.1f}%")
        
        return {'signals': signals, 'analysis': analysis}
    
    @timer_decorator
    def backtest_model(
        self,
        signals: pd.DataFrame,
        features: pd.DataFrame,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Backtest a model's signals.
        
        Args:
            signals: Signals DataFrame.
            features: Features DataFrame.
            model_name: Name for the model.
        
        Returns:
            Backtest results.
        """
        self.logger.info(f"Backtesting {model_name}...")
        
        self.backtest_engine.reset()
        results = self.backtest_engine.run_backtest(signals, features)
        
        # Calculate detailed metrics
        trades_df = results['trades_df']
        equity_df = results['equity_df']
        
        if not trades_df.empty and not equity_df.empty:
            metrics = calculate_performance_metrics(trades_df, equity_df)
            results['metrics'] = metrics
            
            self.logger.info(
                f"{model_name}: Return={metrics.get('total_return_pct', 0):.2f}%, "
                f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
                f"Win Rate={metrics.get('win_rate', 0):.1f}%"
            )
        else:
            results['metrics'] = {}
            self.logger.warning(f"{model_name}: No trades executed")
        
        return results
    
    def save_models(self, save_dir: str = 'models') -> None:
        """
        Save all trained models.
        
        Args:
            save_dir: Directory to save models.
        """
        self.logger.info(f"Saving models to {save_dir}/...")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost models
        for ticker in self.ml_model.models.keys():
            self.ml_model.save_model(ticker, f"{save_dir}/xgboost_{ticker.replace('=', '_')}.pkl")
        
        # Save ensemble
        self.ensemble.save(f"{save_dir}/ensemble")
        
        self.logger.info("Models saved successfully")
    
    def generate_reports(
        self,
        results: Dict[str, Dict],
        save_dir: str = 'results'
    ) -> None:
        """
        Generate comparison reports for all models.
        
        Args:
            results: Dict of model_name -> backtest results.
            save_dir: Directory to save reports.
        """
        self.logger.info(f"Generating reports to {save_dir}/...")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate report for each model
        for model_name, result in results.items():
            if 'metrics' in result and result['metrics']:
                report_path = f"{save_dir}/{model_name.lower().replace(' ', '_')}_report.html"
                generate_performance_report(
                    result['metrics'],
                    result['trades_df'],
                    result['equity_df'],
                    report_path,
                    model_name
                )
                self.logger.info(f"Generated report: {report_path}")
        
        # Generate comparison report
        self._generate_comparison_report(results, save_dir)
    
    def _generate_comparison_report(
        self,
        results: Dict[str, Dict],
        save_dir: str
    ) -> None:
        """Generate model comparison report."""
        comparison_data = []
        
        for model_name, result in results.items():
            metrics = result.get('metrics', {})
            if metrics:
                comparison_data.append({
                    'Model': model_name,
                    'Return (%)': metrics.get('total_return_pct', 0),
                    'Sharpe': metrics.get('sharpe_ratio', 0),
                    'Win Rate (%)': metrics.get('win_rate', 0),
                    'Profit Factor': metrics.get('profit_factor', 0),
                    'Max DD (%)': metrics.get('max_drawdown_pct', 0),
                    'Total Trades': metrics.get('total_trades', 0),
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            df.to_csv(f"{save_dir}/model_comparison.csv", index=False)
            
            # Print comparison
            print("\n" + "=" * 80)
            print("MODEL COMPARISON")
            print("=" * 80)
            print(df.to_string(index=False))
            print("=" * 80 + "\n")
    
    @timer_decorator
    def train_and_evaluate(
        self,
        data_path: str = 'data/processed/features.parquet',
        tune_hyperparams: bool = False,
        save_models: bool = True,
        generate_reports: bool = True
    ) -> Dict[str, Any]:
        """
        Full training and evaluation pipeline.
        
        Args:
            data_path: Path to features file.
            tune_hyperparams: Whether to tune hyperparameters.
            save_models: Whether to save trained models.
            generate_reports: Whether to generate reports.
        
        Returns:
            Complete results dictionary.
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING MODEL TRAINING AND EVALUATION")
        self.logger.info("=" * 60)
        
        # Load data
        df = self.load_data(data_path)
        
        # Split data for backtesting (use last 20% as test)
        test_start_idx = int(len(df) * 0.8)
        test_df = df.iloc[test_start_idx:]
        
        self.logger.info(f"Test period: {test_df.index.min()} to {test_df.index.max()}")
        
        # Train models
        self.train_technical_rules(df)
        self.train_xgboost(df, tune_hyperparams)
        self.train_ensemble(df, tune_hyperparams)
        
        # Filter signals to test period
        all_results = {}
        
        # Backtest Technical Rules
        tech_signals = self.results['technical_signals'].copy()
        tech_signals['Date'] = pd.to_datetime(tech_signals['Date'])
        tech_signals_test = tech_signals[tech_signals['Date'] >= test_df.index.min()]
        all_results['Technical Rules'] = self.backtest_model(
            tech_signals_test, test_df, 'Technical Rules'
        )
        
        # Backtest XGBoost
        ml_signals = self.results.get('xgboost_signals', pd.DataFrame())
        if not ml_signals.empty:
            ml_signals['Date'] = pd.to_datetime(ml_signals['Date'])
            ml_signals_test = ml_signals[ml_signals['Date'] >= test_df.index.min()]
            all_results['XGBoost'] = self.backtest_model(
                ml_signals_test, test_df, 'XGBoost'
            )
        
        # Backtest Ensemble
        ensemble_signals = self.results.get('ensemble_signals', pd.DataFrame())
        if not ensemble_signals.empty:
            ensemble_signals['Date'] = pd.to_datetime(ensemble_signals['Date'])
            ensemble_signals_test = ensemble_signals[ensemble_signals['Date'] >= test_df.index.min()]
            all_results['Ensemble'] = self.backtest_model(
                ensemble_signals_test, test_df, 'Ensemble'
            )
        
        # Save models
        if save_models:
            self.save_models()
        
        # Generate reports
        if generate_reports:
            self.generate_reports(all_results)
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING AND EVALUATION COMPLETE")
        self.logger.info("=" * 60)
        
        return all_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate trading models')
    parser.add_argument('--data', type=str, default='data/processed/features.parquet',
                       help='Path to features file')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving models')
    parser.add_argument('--no-reports', action='store_true',
                       help='Skip generating reports')
    
    args = parser.parse_args()
    
    # Check data exists
    from pathlib import Path
    if not Path(args.data).exists():
        print(f"Error: Data file not found at {args.data}")
        print("Run 'python main.py --mode all' first to generate features.")
        exit(1)
    
    # Train and evaluate
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate(
        data_path=args.data,
        tune_hyperparams=args.tune,
        save_models=not args.no_save,
        generate_reports=not args.no_reports
    )
    
    print("\nTraining complete! Check the results/ directory for reports.")
