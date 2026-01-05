"""
Market Regime Detection Module - Classify market conditions for adaptive trading.

This module provides:
- Trend/range classification
- Volatility regime detection
- VIX-based risk sentiment
- Regime-filtered trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Enum for market regime classifications."""
    STRONG_TREND_UP = "strong_trend_up"
    STRONG_TREND_DOWN = "strong_trend_down"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class MarketRegimeDetector:
    """
    Classify market conditions and adapt trading strategy.
    
    Regimes:
    1. Strong Trend Up (ADX>25, price>SMA50>SMA200)
    2. Strong Trend Down (ADX>25, price<SMA50<SMA200)
    3. Weak Trend (15<ADX<25)
    4. Ranging/Choppy (ADX<15)
    5. High Volatility (VIX>25 or ATR percentile>80)
    6. Low Volatility (VIX<15 or ATR percentile<20)
    
    Features:
    - Multi-factor regime classification
    - Regime history tracking
    - Trade filtering by regime
    - Regime-specific performance analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize regime detector.
        
        Args:
            config: Configuration dictionary with regime thresholds
        """
        self.config = config or {}
        self.regime_config = self.config.get('regime', {
            'adx_trend_threshold': 25,
            'adx_weak_threshold': 15,
            'vix_high': 25,
            'vix_low': 15,
            'atr_high_percentile': 80,
            'atr_low_percentile': 20
        })
        
        self.regime_history = {}
        
    def detect_regime(
        self,
        df: pd.DataFrame,
        ticker: str,
        row_idx: int = -1
    ) -> MarketRegime:
        """
        Detect current market regime for a single point.
        
        Args:
            df: DataFrame with price and indicator data
            ticker: Ticker symbol
            row_idx: Row index to evaluate (-1 for last)
            
        Returns:
            MarketRegime enum value
        """
        row = df.iloc[row_idx]
        
        # Get column names
        close_col = f'{ticker}_Close'
        sma50_col = f'{ticker}_SMA_50'
        sma200_col = f'{ticker}_SMA_200'
        adx_col = f'{ticker}_ADX_14'
        atr_col = f'{ticker}_ATR_14'
        
        # Check for VIX
        vix_col = None
        for col in df.columns:
            if 'VIX' in col and 'Close' in col:
                vix_col = col
                break
        
        # Extract values (with fallbacks)
        adx = row.get(adx_col, 15)
        close = row.get(close_col, 0)
        sma50 = row.get(sma50_col, close)
        sma200 = row.get(sma200_col, close)
        atr = row.get(atr_col, 0)
        vix = row.get(vix_col, 20) if vix_col else 20
        
        # Handle NaN
        if pd.isna(adx):
            adx = 15
        if pd.isna(vix):
            vix = 20
        
        # Regime classification logic
        adx_trend = self.regime_config.get('adx_trend_threshold', 25)
        adx_weak = self.regime_config.get('adx_weak_threshold', 15)
        vix_high = self.regime_config.get('vix_high', 25)
        vix_low = self.regime_config.get('vix_low', 15)
        
        # Priority: Volatility regimes first
        if vix > vix_high:
            return MarketRegime.HIGH_VOLATILITY
        
        if vix < vix_low:
            return MarketRegime.LOW_VOLATILITY
        
        # Then trend regimes
        if adx > adx_trend:
            if close > sma50 > sma200:
                return MarketRegime.STRONG_TREND_UP
            elif close < sma50 < sma200:
                return MarketRegime.STRONG_TREND_DOWN
            else:
                return MarketRegime.WEAK_TREND
        
        if adx < adx_weak:
            return MarketRegime.RANGING
        
        return MarketRegime.WEAK_TREND
    
    def get_regime_history(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.Series:
        """
        Get regime classification for entire DataFrame.
        
        Args:
            df: DataFrame with price and indicator data
            ticker: Ticker symbol
            
        Returns:
            Series with regime classifications
        """
        logger.info(f"Computing regime history for {ticker}")
        
        regimes = []
        
        for i in range(len(df)):
            regime = self.detect_regime(df, ticker, i)
            regimes.append(regime.value)
        
        regime_series = pd.Series(regimes, index=df.index, name=f'{ticker}_regime')
        
        # Store in history
        self.regime_history[ticker] = regime_series
        
        # Log regime distribution
        regime_counts = regime_series.value_counts()
        logger.info(f"Regime distribution for {ticker}:")
        for regime, count in regime_counts.items():
            pct = count / len(regime_series) * 100
            logger.info(f"  {regime}: {count} ({pct:.1f}%)")
        
        return regime_series
    
    def add_regime_columns(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Add regime classification columns to DataFrame.
        
        Args:
            df: DataFrame to add regime columns to
            ticker: Ticker symbol
            
        Returns:
            DataFrame with regime columns added
        """
        df = df.copy()
        
        # Get regime history
        regime_series = self.get_regime_history(df, ticker)
        df[f'{ticker}_regime'] = regime_series
        
        # Add one-hot encoded regime columns
        for regime in MarketRegime:
            if regime != MarketRegime.UNKNOWN:
                df[f'{ticker}_is_{regime.value}'] = (regime_series == regime.value).astype(int)
        
        # Add regime-based trading rules
        df[f'{ticker}_regime_tradeable'] = df[f'{ticker}_regime'].apply(
            lambda x: self._is_tradeable_regime(x)
        ).astype(int)
        
        # Add position size multiplier based on regime
        df[f'{ticker}_regime_size_mult'] = df[f'{ticker}_regime'].apply(
            lambda x: self._get_position_size_multiplier(x)
        )
        
        return df
    
    def _is_tradeable_regime(self, regime: str) -> bool:
        """
        Determine if a regime is suitable for trading.
        
        Args:
            regime: Regime string value
            
        Returns:
            True if regime is tradeable
        """
        tradeable = [
            MarketRegime.STRONG_TREND_UP.value,
            MarketRegime.STRONG_TREND_DOWN.value,
            MarketRegime.WEAK_TREND.value,
            MarketRegime.LOW_VOLATILITY.value  # Good for breakouts
        ]
        return regime in tradeable
    
    def _get_position_size_multiplier(self, regime: str) -> float:
        """
        Get position size multiplier based on regime.
        
        Args:
            regime: Regime string value
            
        Returns:
            Position size multiplier (0.0 to 1.0)
        """
        multipliers = {
            MarketRegime.STRONG_TREND_UP.value: 1.0,
            MarketRegime.STRONG_TREND_DOWN.value: 1.0,
            MarketRegime.WEAK_TREND.value: 0.75,
            MarketRegime.RANGING.value: 0.5,
            MarketRegime.HIGH_VOLATILITY.value: 0.5,  # Reduce size in high vol
            MarketRegime.LOW_VOLATILITY.value: 0.75,
            MarketRegime.UNKNOWN.value: 0.5
        }
        return multipliers.get(regime, 0.5)
    
    def filter_by_regime(
        self,
        signals_df: pd.DataFrame,
        regime_col: str,
        allowed_regimes: List[str] = None
    ) -> pd.DataFrame:
        """
        Filter trading signals to only include favorable regimes.
        
        Args:
            signals_df: DataFrame with trading signals
            regime_col: Column name for regime classification
            allowed_regimes: List of allowed regime values
            
        Returns:
            Filtered DataFrame
        """
        if allowed_regimes is None:
            allowed_regimes = [
                MarketRegime.STRONG_TREND_UP.value,
                MarketRegime.STRONG_TREND_DOWN.value,
                MarketRegime.WEAK_TREND.value
            ]
        
        if regime_col not in signals_df.columns:
            logger.warning(f"Regime column {regime_col} not found")
            return signals_df
        
        filtered = signals_df[signals_df[regime_col].isin(allowed_regimes)].copy()
        
        logger.info(f"Filtered signals: {len(signals_df)} -> {len(filtered)} "
                   f"({len(filtered)/len(signals_df)*100:.1f}% retained)")
        
        return filtered
    
    def analyze_regime_performance(
        self,
        df: pd.DataFrame,
        predictions_col: str,
        actuals_col: str,
        regime_col: str
    ) -> pd.DataFrame:
        """
        Analyze model performance by regime.
        
        Args:
            df: DataFrame with predictions and actuals
            predictions_col: Column name for predictions
            actuals_col: Column name for actual values
            regime_col: Column name for regime
            
        Returns:
            DataFrame with performance by regime
        """
        if not all(col in df.columns for col in [predictions_col, actuals_col, regime_col]):
            logger.error("Required columns not found")
            return pd.DataFrame()
        
        # Drop NaN
        valid = df[[predictions_col, actuals_col, regime_col]].dropna()
        
        results = []
        
        for regime in valid[regime_col].unique():
            regime_data = valid[valid[regime_col] == regime]
            
            if len(regime_data) < 10:
                continue
            
            preds = regime_data[predictions_col]
            actuals = regime_data[actuals_col]
            
            # Calculate accuracy
            accuracy = (preds == actuals).mean()
            
            # Calculate win rate and other metrics
            n_trades = len(regime_data)
            n_correct = (preds == actuals).sum()
            
            results.append({
                'regime': regime,
                'n_samples': n_trades,
                'accuracy': accuracy,
                'n_correct': n_correct,
                'pct_of_total': n_trades / len(valid) * 100,
                'is_profitable': accuracy > 0.5
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        logger.info("Regime Performance Analysis:")
        for _, row in results_df.iterrows():
            status = "✅" if row['is_profitable'] else "❌"
            logger.info(f"  {status} {row['regime']}: {row['accuracy']:.1%} "
                       f"(n={row['n_samples']}, {row['pct_of_total']:.1f}%)")
        
        return results_df
    
    def get_regime_trading_rules(self, regime: MarketRegime) -> Dict:
        """
        Get trading rules for a specific regime.
        
        Args:
            regime: MarketRegime enum
            
        Returns:
            Dictionary with trading rules
        """
        rules = {
            MarketRegime.STRONG_TREND_UP: {
                'direction': 'long_only',
                'min_confidence': 50,
                'stop_multiplier': 2.0,
                'take_profit_multiplier': 3.0,
                'max_holding_days': 10,
                'position_size_mult': 1.0,
                'description': 'Trade with the trend, wider stops, larger targets'
            },
            MarketRegime.STRONG_TREND_DOWN: {
                'direction': 'short_only',
                'min_confidence': 50,
                'stop_multiplier': 2.0,
                'take_profit_multiplier': 3.0,
                'max_holding_days': 10,
                'position_size_mult': 1.0,
                'description': 'Trade with the trend, wider stops, larger targets'
            },
            MarketRegime.WEAK_TREND: {
                'direction': 'both',
                'min_confidence': 60,
                'stop_multiplier': 1.5,
                'take_profit_multiplier': 2.0,
                'max_holding_days': 5,
                'position_size_mult': 0.75,
                'description': 'Cautious trading, higher confidence required'
            },
            MarketRegime.RANGING: {
                'direction': 'mean_reversion',
                'min_confidence': 70,
                'stop_multiplier': 1.0,
                'take_profit_multiplier': 1.5,
                'max_holding_days': 3,
                'position_size_mult': 0.5,
                'description': 'Mean reversion only, tight stops, quick profits'
            },
            MarketRegime.HIGH_VOLATILITY: {
                'direction': 'both',
                'min_confidence': 75,
                'stop_multiplier': 2.5,
                'take_profit_multiplier': 4.0,
                'max_holding_days': 3,
                'position_size_mult': 0.5,
                'description': 'Reduce size, widen stops, very high confidence only'
            },
            MarketRegime.LOW_VOLATILITY: {
                'direction': 'breakout',
                'min_confidence': 55,
                'stop_multiplier': 1.5,
                'take_profit_multiplier': 2.5,
                'max_holding_days': 7,
                'position_size_mult': 0.75,
                'description': 'Wait for breakout confirmation'
            },
            MarketRegime.UNKNOWN: {
                'direction': 'none',
                'min_confidence': 100,
                'stop_multiplier': 0,
                'take_profit_multiplier': 0,
                'max_holding_days': 0,
                'position_size_mult': 0,
                'description': 'Do not trade - unknown conditions'
            }
        }
        
        return rules.get(regime, rules[MarketRegime.UNKNOWN])
    
    def get_current_regime_summary(self, df: pd.DataFrame, tickers: List[str]) -> Dict:
        """
        Get summary of current regime for all tickers.
        
        Args:
            df: DataFrame with latest data
            tickers: List of ticker symbols
            
        Returns:
            Dictionary with current regime for each ticker
        """
        summary = {}
        
        for ticker in tickers:
            regime = self.detect_regime(df, ticker, -1)
            rules = self.get_regime_trading_rules(regime)
            
            summary[ticker] = {
                'regime': regime.value,
                'rules': rules,
                'tradeable': self._is_tradeable_regime(regime.value),
                'position_mult': self._get_position_size_multiplier(regime.value)
            }
        
        return summary


def main():
    """Main function to run regime detection."""
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load features
    features_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features.parquet'
    df = pd.read_parquet(features_path)
    
    # Initialize detector
    detector = MarketRegimeDetector(config)
    
    # Add regime columns for each ticker
    for ticker in ['GBPUSD_X', 'EURUSD_X', 'GC_F']:
        df = detector.add_regime_columns(df, ticker)
    
    # Get current regime summary
    summary = detector.get_current_regime_summary(df, ['GBPUSD_X', 'EURUSD_X', 'GC_F'])
    
    print("\n📊 Current Market Regime Summary")
    print("=" * 50)
    for ticker, info in summary.items():
        print(f"\n{ticker}:")
        print(f"  Regime: {info['regime']}")
        print(f"  Tradeable: {'✅' if info['tradeable'] else '❌'}")
        print(f"  Position Multiplier: {info['position_mult']}")
        print(f"  Rules: {info['rules']['description']}")
    
    # Save enhanced features
    output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features_with_regime.parquet'
    df.to_parquet(output_path)
    
    print(f"\n✅ Saved features with regime to {output_path}")


if __name__ == "__main__":
    main()
