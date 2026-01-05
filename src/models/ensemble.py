"""
Ensemble Model for Forex Signal Trading.

Combines Technical Rules and XGBoost predictions.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config
from src.models.technical_rules import TechnicalRulesSystem
from src.models.ml_models import XGBoostTradingModel


class EnsembleModel:
    """
    Weighted ensemble combining technical rules and XGBoost.
    
    Combination strategies:
    1. Simple voting: Both agree → high confidence
    2. Weighted average: technical_weight × rules + ml_weight × xgboost
    3. Conflict resolution: When models disagree
    
    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        technical_system: TechnicalRulesSystem instance.
        ml_model: XGBoostTradingModel instance.
        weights: Model weights dictionary.
    
    Example:
        >>> ensemble = EnsembleModel('config/config.yaml')
        >>> ensemble.train(features_df)
        >>> signals = ensemble.predict(test_df)
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the EnsembleModel.
        
        Args:
            config_path: Path to configuration YAML file.
        """
        self.config = load_config(config_path)
        self.logger = get_logger('forex_signal_model.ensemble')
        
        # Initialize component models
        self.technical_system = TechnicalRulesSystem(config_path)
        self.ml_model = XGBoostTradingModel(config_path)
        
        # Ensemble configuration
        self.ensemble_config = self.config.get('models', {}).get('ensemble', {})
        self.weights = {
            'technical': self.ensemble_config.get('technical_weight', 0.4),
            'ml': self.ensemble_config.get('ml_weight', 0.6)
        }
        self.disagreement_penalty = self.ensemble_config.get('disagreement_penalty', 20)
        self.agreement_bonus = self.ensemble_config.get('agreement_bonus', 10)
        self.min_confidence = self.config.get('risk', {}).get('min_confidence_to_trade', 40)
        
        self.is_trained = False
        
        self.logger.info(f"EnsembleModel initialized with weights: {self.weights}")
    
    def train(
        self,
        df: pd.DataFrame,
        tune_hyperparams: bool = False
    ) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Note: Technical rules don't require training (rule-based).
        Only XGBoost needs training.
        
        Args:
            df: Features DataFrame.
            tune_hyperparams: Whether to tune XGBoost hyperparameters.
        
        Returns:
            Training metrics dictionary.
        """
        self.logger.info("Training ensemble model...")
        
        # Train XGBoost models
        ml_metrics = self.ml_model.train_all(df, tune_hyperparams)
        
        self.is_trained = True
        
        return {
            'ml_metrics': ml_metrics,
            'weights': self.weights
        }
    
    def combine_signals(
        self,
        technical_signals: pd.DataFrame,
        ml_signals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine signals from both models.
        
        Args:
            technical_signals: DataFrame from TechnicalRulesSystem.
            ml_signals: DataFrame from XGBoostTradingModel.
        
        Returns:
            Combined signals DataFrame.
        """
        # Merge on Date and Ticker
        merged = technical_signals.merge(
            ml_signals,
            on=['Date', 'Ticker'],
            suffixes=('_tech', '_ml')
        )
        
        result = pd.DataFrame({
            'Date': merged['Date'],
            'Ticker': merged['Ticker']
        })
        
        # Get signals and confidences
        tech_signal = merged['Signal_tech']
        ml_signal = merged['Signal_ml']
        tech_conf = merged['Confidence_tech']
        ml_conf = merged['Confidence_ml']
        
        # Calculate weighted signal
        weighted_signal = (
            self.weights['technical'] * tech_signal +
            self.weights['ml'] * ml_signal
        )
        
        # Discretize: >0.5 = BUY, <-0.5 = SELL, else HOLD
        final_signal = pd.Series(0, index=merged.index)
        final_signal[weighted_signal > 0.3] = 1
        final_signal[weighted_signal < -0.3] = -1
        
        # Calculate confidence with agreement/disagreement adjustment
        weighted_conf = (
            self.weights['technical'] * tech_conf +
            self.weights['ml'] * ml_conf
        )
        
        # Both agree → bonus
        agreement = (tech_signal == ml_signal) & (tech_signal != 0)
        # Disagree → penalty
        disagreement = (tech_signal != ml_signal) & (tech_signal != 0) & (ml_signal != 0)
        
        final_conf = weighted_conf.copy()
        final_conf[agreement] += self.agreement_bonus
        final_conf[disagreement] -= self.disagreement_penalty
        
        # Clamp confidence
        final_conf = final_conf.clip(0, 100)
        
        # If both low confidence → HOLD
        low_conf_both = (tech_conf < self.min_confidence) & (ml_conf < self.min_confidence)
        final_signal[low_conf_both] = 0
        
        result['Signal'] = final_signal
        result['Confidence'] = final_conf
        result['Signal_Tech'] = tech_signal
        result['Signal_ML'] = ml_signal
        result['Confidence_Tech'] = tech_conf
        result['Confidence_ML'] = ml_conf
        result['Agreement'] = agreement.astype(int)
        
        # Get stop loss from technical system (uses ATR)
        if 'StopLoss' in merged.columns:
            result['StopLoss'] = merged['StopLoss']
        elif 'StopLoss_tech' in merged.columns:
            result['StopLoss'] = merged['StopLoss_tech']
        else:
            result['StopLoss'] = 0
        
        return result
    
    def predict(
        self,
        df: pd.DataFrame,
        ticker: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions.
        
        Args:
            df: Features DataFrame.
            ticker: Specific ticker or None for all.
        
        Returns:
            Combined signals DataFrame.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        main_tickers = self.config['data']['tickers']['main']
        if ticker:
            main_tickers = [ticker]
        
        all_signals = []
        
        for t in main_tickers:
            try:
                # Get technical signals
                tech_signals = self.technical_system.generate_signals(df, t)
                
                # Get ML signals
                ml_signals = self.ml_model.predict(df, t)
                
                # Combine
                combined = self.combine_signals(tech_signals, ml_signals)
                all_signals.append(combined)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate ensemble signals for {t}: {e}")
        
        if all_signals:
            result = pd.concat(all_signals, ignore_index=True)
            self._log_signal_summary(result)
            return result
        
        return pd.DataFrame()
    
    def _log_signal_summary(self, signals_df: pd.DataFrame) -> None:
        """Log summary of generated signals."""
        total = len(signals_df)
        buy = (signals_df['Signal'] == 1).sum()
        sell = (signals_df['Signal'] == -1).sum()
        hold = (signals_df['Signal'] == 0).sum()
        agree = signals_df['Agreement'].sum()
        
        self.logger.info(
            f"Ensemble signals: BUY={buy}, SELL={sell}, HOLD={hold}, "
            f"Agreement={agree}/{total} ({agree/total*100:.1f}%)"
        )
    
    def get_signal_analysis(
        self,
        signals_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze signal quality and model agreement.
        
        Args:
            signals_df: DataFrame from predict().
        
        Returns:
            Analysis dictionary.
        """
        analysis = {
            'total_signals': len(signals_df),
            'buy_count': (signals_df['Signal'] == 1).sum(),
            'sell_count': (signals_df['Signal'] == -1).sum(),
            'hold_count': (signals_df['Signal'] == 0).sum(),
            'agreement_rate': signals_df['Agreement'].mean() * 100,
            'avg_confidence': signals_df['Confidence'].mean(),
            'high_confidence_signals': (signals_df['Confidence'] > 70).sum(),
        }
        
        # Per-ticker analysis
        for ticker in signals_df['Ticker'].unique():
            ticker_df = signals_df[signals_df['Ticker'] == ticker]
            analysis[f'{ticker}_buy'] = (ticker_df['Signal'] == 1).sum()
            analysis[f'{ticker}_sell'] = (ticker_df['Signal'] == -1).sum()
            analysis[f'{ticker}_agreement'] = ticker_df['Agreement'].mean() * 100
        
        return analysis
    
    def save(self, path: str = 'models/ensemble') -> None:
        """
        Save ensemble model components.
        
        Args:
            path: Base path for model files.
        """
        import pickle
        from pathlib import Path
        
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost models for each ticker
        for ticker in self.ml_model.models.keys():
            self.ml_model.save_model(ticker, f"{path}/xgboost_{ticker.replace('=', '_')}.pkl")
        
        # Save ensemble config
        config_data = {
            'weights': self.weights,
            'disagreement_penalty': self.disagreement_penalty,
            'agreement_bonus': self.agreement_bonus,
            'min_confidence': self.min_confidence
        }
        
        with open(f"{path}/ensemble_config.pkl", 'wb') as f:
            pickle.dump(config_data, f)
        
        self.logger.info(f"Saved ensemble model to {path}")
    
    def load(self, path: str = 'models/ensemble') -> None:
        """
        Load ensemble model components.
        
        Args:
            path: Base path for model files.
        """
        import pickle
        from pathlib import Path
        
        # Load XGBoost models
        for model_file in Path(path).glob("xgboost_*.pkl"):
            ticker = model_file.stem.replace('xgboost_', '').replace('_', '=')
            self.ml_model.load_model(ticker, str(model_file))
        
        # Load ensemble config
        config_path = Path(path) / "ensemble_config.pkl"
        if config_path.exists():
            with open(config_path, 'rb') as f:
                config_data = pickle.load(f)
            
            self.weights = config_data['weights']
            self.disagreement_penalty = config_data['disagreement_penalty']
            self.agreement_bonus = config_data['agreement_bonus']
            self.min_confidence = config_data.get('min_confidence', 40)
        
        self.is_trained = True
        self.logger.info(f"Loaded ensemble model from {path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    from pathlib import Path
    
    # Load features
    features_path = Path('data/processed/features.parquet')
    if features_path.exists():
        df = pd.read_parquet(features_path)
        
        # Initialize and train ensemble
        ensemble = EnsembleModel()
        metrics = ensemble.train(df, tune_hyperparams=False)
        
        # Generate predictions
        signals = ensemble.predict(df)
        
        # Analyze
        analysis = ensemble.get_signal_analysis(signals)
        
        print("\nEnsemble Analysis:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Save
        ensemble.save()
    else:
        print(f"Features not found at {features_path}")
