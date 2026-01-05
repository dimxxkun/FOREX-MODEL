"""
Technical Rules Trading System.

Rule-based trading system using technical indicators for signal generation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config


class TechnicalRulesSystem:
    """
    Rule-based trading system using technical indicators.
    
    Signal generation based on:
    - Trend: SMA crossovers, EMA alignment
    - Momentum: RSI, MACD, Stochastic
    - Volatility: Bollinger Bands, ATR
    - Multi-timeframe: Weekly trend alignment
    - Intermarket: DXY, VIX correlation
    
    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        thresholds: Signal thresholds from config.
    
    Example:
        >>> system = TechnicalRulesSystem('config/config.yaml')
        >>> signals = system.generate_signals(features_df, 'GBPUSD')
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the TechnicalRulesSystem.
        
        Args:
            config_path: Path to configuration YAML file.
        """
        self.config = load_config(config_path)
        self.logger = get_logger('forex_signal_model.technical_rules')
        
        # Load thresholds from config
        self.model_config = self.config.get('models', {}).get('technical_rules', {})
        self.thresholds = {
            'sma_fast': self.model_config.get('sma_crossover_fast', 50),
            'sma_slow': self.model_config.get('sma_crossover_slow', 200),
            'rsi_oversold': self.model_config.get('rsi_oversold', 30),
            'rsi_overbought': self.model_config.get('rsi_overbought', 70),
            'bb_entry_threshold': self.model_config.get('bb_entry_threshold', 0.2),
            'min_conditions': self.model_config.get('min_conditions', 3),
            'vix_threshold': self.model_config.get('vix_threshold', 25),
            'atr_stop_multiplier': self.model_config.get('atr_stop_multiplier', 2.0),
        }
        
        self.logger.info(f"TechnicalRulesSystem initialized with thresholds: {self.thresholds}")
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Generate trading signals for a ticker.
        
        Args:
            df: DataFrame with features (from feature_engineering).
            ticker: Ticker symbol (e.g., 'GBPUSD', 'EURUSD', 'GC_F').
        
        Returns:
            DataFrame with columns: [Date, Ticker, Signal, Confidence, StopLoss]
            Signal: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        self.logger.info(f"Generating signals for {ticker}...")
        
        # Normalize ticker name for column matching
        ticker_clean = ticker.replace('=', '_').replace('^', '').replace('-', '_').replace('.', '_')
        
        # Calculate conditions
        buy_conditions = self._calculate_buy_conditions(df, ticker_clean)
        sell_conditions = self._calculate_sell_conditions(df, ticker_clean)
        
        # Count satisfied conditions
        buy_count = buy_conditions.sum(axis=1)
        sell_count = sell_conditions.sum(axis=1)
        
        # Generate signals based on condition counts
        min_conditions = self.thresholds['min_conditions']
        
        signals = pd.Series(0, index=df.index, name='Signal')
        signals[buy_count >= min_conditions] = 1
        signals[sell_count >= min_conditions] = -1
        
        # Calculate confidence scores
        confidence = self._calculate_confidence(buy_count, sell_count, buy_conditions.shape[1])
        
        # Calculate stop losses
        stop_loss = self._calculate_stop_loss(df, ticker_clean, signals)
        
        # Create output DataFrame
        result = pd.DataFrame({
            'Date': df.index,
            'Ticker': ticker,
            'Signal': signals.values,
            'Confidence': confidence.values,
            'StopLoss': stop_loss.values,
            'BuyConditions': buy_count.values,
            'SellConditions': sell_count.values
        })
        
        # Log signal distribution
        signal_counts = result['Signal'].value_counts()
        self.logger.info(f"{ticker} signals: BUY={signal_counts.get(1, 0)}, "
                        f"SELL={signal_counts.get(-1, 0)}, HOLD={signal_counts.get(0, 0)}")
        
        return result
    
    def _calculate_buy_conditions(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Calculate individual buy conditions.
        
        Args:
            df: Features DataFrame.
            ticker: Cleaned ticker name.
        
        Returns:
            DataFrame with boolean columns for each condition.
        """
        conditions = pd.DataFrame(index=df.index)
        
        # 1. Trend: Price > SMA50 > SMA200 (uptrend)
        sma_50 = df.get(f'{ticker}_SMA_50')
        sma_200 = df.get(f'{ticker}_SMA_200')
        close = df.get(f'{ticker}_Close')
        
        if sma_50 is not None and sma_200 is not None and close is not None:
            conditions['uptrend'] = (close > sma_50) & (sma_50 > sma_200)
        else:
            conditions['uptrend'] = False
        
        # 2. Momentum: RSI between 30-70 (not overbought)
        rsi = df.get(f'{ticker}_RSI')
        if rsi is not None:
            conditions['rsi_ok'] = (rsi > self.thresholds['rsi_oversold']) & \
                                   (rsi < self.thresholds['rsi_overbought'])
        else:
            conditions['rsi_ok'] = False
        
        # 3. MACD > Signal line (bullish momentum)
        macd = df.get(f'{ticker}_MACD')
        macd_signal = df.get(f'{ticker}_MACD_Signal')
        if macd is not None and macd_signal is not None:
            conditions['macd_bullish'] = macd > macd_signal
        else:
            conditions['macd_bullish'] = False
        
        # 4. Price near lower Bollinger Band (entry point)
        bb_pctb = df.get(f'{ticker}_BB_PctB')
        if bb_pctb is not None:
            conditions['bb_low'] = bb_pctb < self.thresholds['bb_entry_threshold']
        else:
            conditions['bb_low'] = False
        
        # 5. Weekly trend aligned (weekly SMA50 > SMA200)
        weekly_trend = df.get(f'{ticker}_Weekly_Trend_Align')
        if weekly_trend is not None:
            conditions['weekly_bullish'] = weekly_trend > 0
        else:
            conditions['weekly_bullish'] = False
        
        # 6. DXY weakening (for USD pairs - inverse relationship)
        dxy_return = df.get('DXY_Return_1d')
        if dxy_return is not None and ticker in ['GBPUSD', 'EURUSD']:
            conditions['dxy_weak'] = dxy_return < 0  # DXY falling = USD pairs rising
        elif dxy_return is not None:
            conditions['dxy_weak'] = dxy_return > 0  # For gold, DXY falling is bullish
        else:
            conditions['dxy_weak'] = False
        
        # 7. VIX < 25 (risk-on environment)
        vix_level = df.get('VIX_Level')
        if vix_level is not None:
            conditions['vix_low'] = vix_level < self.thresholds['vix_threshold']
        else:
            conditions['vix_low'] = False
        
        return conditions.fillna(False).astype(bool)
    
    def _calculate_sell_conditions(
        self,
        df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Calculate individual sell conditions.
        
        Args:
            df: Features DataFrame.
            ticker: Cleaned ticker name.
        
        Returns:
            DataFrame with boolean columns for each condition.
        """
        conditions = pd.DataFrame(index=df.index)
        
        # 1. Trend: Price < SMA50 < SMA200 (downtrend)
        sma_50 = df.get(f'{ticker}_SMA_50')
        sma_200 = df.get(f'{ticker}_SMA_200')
        close = df.get(f'{ticker}_Close')
        
        if sma_50 is not None and sma_200 is not None and close is not None:
            conditions['downtrend'] = (close < sma_50) & (sma_50 < sma_200)
        else:
            conditions['downtrend'] = False
        
        # 2. RSI between 30-70 (not oversold)
        rsi = df.get(f'{ticker}_RSI')
        if rsi is not None:
            conditions['rsi_ok'] = (rsi > self.thresholds['rsi_oversold']) & \
                                   (rsi < self.thresholds['rsi_overbought'])
        else:
            conditions['rsi_ok'] = False
        
        # 3. MACD < Signal line (bearish momentum)
        macd = df.get(f'{ticker}_MACD')
        macd_signal = df.get(f'{ticker}_MACD_Signal')
        if macd is not None and macd_signal is not None:
            conditions['macd_bearish'] = macd < macd_signal
        else:
            conditions['macd_bearish'] = False
        
        # 4. Price near upper Bollinger Band
        bb_pctb = df.get(f'{ticker}_BB_PctB')
        if bb_pctb is not None:
            conditions['bb_high'] = bb_pctb > (1 - self.thresholds['bb_entry_threshold'])
        else:
            conditions['bb_high'] = False
        
        # 5. Weekly trend aligned bearish
        weekly_trend = df.get(f'{ticker}_Weekly_Trend_Align')
        if weekly_trend is not None:
            conditions['weekly_bearish'] = weekly_trend < 0
        else:
            conditions['weekly_bearish'] = False
        
        # 6. DXY strengthening
        dxy_return = df.get('DXY_Return_1d')
        if dxy_return is not None and ticker in ['GBPUSD', 'EURUSD']:
            conditions['dxy_strong'] = dxy_return > 0  # DXY rising = USD pairs falling
        elif dxy_return is not None:
            conditions['dxy_strong'] = dxy_return < 0  # For gold, DXY rising is bearish
        else:
            conditions['dxy_strong'] = False
        
        # 7. VIX > 25 (risk-off environment)
        vix_level = df.get('VIX_Level')
        if vix_level is not None:
            conditions['vix_high'] = vix_level > self.thresholds['vix_threshold']
        else:
            conditions['vix_high'] = False
        
        return conditions.fillna(False).astype(bool)
    
    def _calculate_confidence(
        self,
        buy_count: pd.Series,
        sell_count: pd.Series,
        total_conditions: int
    ) -> pd.Series:
        """
        Calculate confidence scores based on conditions satisfied.
        
        Args:
            buy_count: Number of buy conditions satisfied.
            sell_count: Number of sell conditions satisfied.
            total_conditions: Total number of conditions.
        
        Returns:
            Series with confidence scores (0-100).
        """
        # Confidence = (conditions satisfied / total) * 100
        max_count = np.maximum(buy_count, sell_count)
        confidence = (max_count / total_conditions) * 100
        
        # Minimum confidence floor
        confidence = confidence.clip(lower=0, upper=100)
        
        return confidence
    
    def _calculate_stop_loss(
        self,
        df: pd.DataFrame,
        ticker: str,
        signals: pd.Series
    ) -> pd.Series:
        """
        Calculate dynamic stop loss prices based on ATR.
        
        Args:
            df: Features DataFrame.
            ticker: Cleaned ticker name.
            signals: Signal series (1=BUY, -1=SELL, 0=HOLD).
        
        Returns:
            Series with stop loss prices.
        """
        close = df.get(f'{ticker}_Close')
        atr = df.get(f'{ticker}_ATR')
        
        if close is None or atr is None:
            return pd.Series(0, index=df.index)
        
        stop_distance = atr * self.thresholds['atr_stop_multiplier']
        
        # For BUY: stop loss below entry
        # For SELL: stop loss above entry
        stop_loss = pd.Series(0.0, index=df.index)
        stop_loss[signals == 1] = (close - stop_distance)[signals == 1]
        stop_loss[signals == -1] = (close + stop_distance)[signals == -1]
        
        return stop_loss
    
    def generate_all_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for all main tickers.
        
        Args:
            df: Features DataFrame with all tickers.
        
        Returns:
            Combined DataFrame with signals for all tickers.
        """
        main_tickers = self.config['data']['tickers']['main']
        all_signals = []
        
        for ticker in main_tickers:
            try:
                signals = self.generate_signals(df, ticker)
                all_signals.append(signals)
            except Exception as e:
                self.logger.warning(f"Failed to generate signals for {ticker}: {e}")
        
        if all_signals:
            return pd.concat(all_signals, ignore_index=True)
        return pd.DataFrame()
    
    def get_signal_summary(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of generated signals.
        
        Args:
            signals_df: DataFrame from generate_signals().
        
        Returns:
            Dictionary with signal statistics.
        """
        summary = {
            'total_signals': len(signals_df),
            'buy_signals': (signals_df['Signal'] == 1).sum(),
            'sell_signals': (signals_df['Signal'] == -1).sum(),
            'hold_signals': (signals_df['Signal'] == 0).sum(),
            'avg_confidence': signals_df['Confidence'].mean(),
            'high_confidence_signals': (signals_df['Confidence'] > 70).sum(),
        }
        
        # Per-ticker breakdown
        if 'Ticker' in signals_df.columns:
            for ticker in signals_df['Ticker'].unique():
                ticker_signals = signals_df[signals_df['Ticker'] == ticker]
                summary[f'{ticker}_buy'] = (ticker_signals['Signal'] == 1).sum()
                summary[f'{ticker}_sell'] = (ticker_signals['Signal'] == -1).sum()
        
        return summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    import pandas as pd
    from pathlib import Path
    
    # Load features
    features_path = Path('data/processed/features.parquet')
    if features_path.exists():
        df = pd.read_parquet(features_path)
        
        # Initialize system
        system = TechnicalRulesSystem()
        
        # Generate signals for all tickers
        signals = system.generate_all_signals(df)
        
        # Print summary
        summary = system.get_signal_summary(signals)
        print("\nSignal Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        print(f"Features not found at {features_path}")
