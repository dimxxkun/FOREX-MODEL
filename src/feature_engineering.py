"""
Feature Engineering Module for Forex Signal Model.

This module creates all technical, fundamental, and derived features
for the forex trading signal model.

Key features:
- Tier 1 Technical Indicators (SMA, EMA, RSI, MACD, Bollinger, ATR, etc.)
- Multitimeframe Features (weekly aggregations)
- Tier 2 Intermarket Features (DXY, VIX, correlations)
- Target Variable Creation with proper lagging
- Look-ahead bias prevention
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

from src.utils import (
    get_logger,
    load_config,
    timer_decorator,
    safe_divide,
    save_dataframe
)


class FeatureEngine:
    """
    Creates all technical, fundamental, and derived features.
    
    This class handles comprehensive feature engineering for the forex
    signal model, including trend, momentum, volatility, and intermarket
    features.
    
    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        data: Input price data DataFrame.
        features: Output feature DataFrame.
    
    Example:
        >>> engine = FeatureEngine('config/config.yaml')
        >>> features = engine.generate_features(price_data)
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the FeatureEngine.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = load_config(config_path)
        self.logger = get_logger('forex_signal_model.feature_engineering')
        
        # Feature configuration
        self.tech_config = self.config['features']['technical']
        self.intermarket_config = self.config['features']['intermarket']
        
        # Main tickers for feature generation
        self.main_tickers = [
            t.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
            for t in self.config['data']['tickers']['main']
        ]
        
        # Intermarket tickers
        self.intermarket_tickers = [
            t.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
            for t in self.config['data']['tickers']['intermarket']
        ]
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.feature_metadata: Dict[str, Dict] = {}
        
        # Paths
        self.features_path = Path(self.config['data']['paths']['features'])
        
        self.logger.info(
            f"FeatureEngine initialized for {len(self.main_tickers)} main tickers"
        )
    
    # ========================================================================
    # TIER 1: TECHNICAL INDICATORS
    # ========================================================================
    
    @timer_decorator
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all Tier 1 technical indicators for each main pair.
        
        Args:
            df: DataFrame with OHLCV data (wide format).
        
        Returns:
            DataFrame with added technical indicators.
        """
        self.logger.info("Adding technical indicators...")
        
        for ticker in self.main_tickers:
            self.logger.debug(f"Processing {ticker}...")
            
            # Get OHLCV columns for this ticker
            open_col = f'{ticker}_Open'
            high_col = f'{ticker}_High'
            low_col = f'{ticker}_Low'
            close_col = f'{ticker}_Close'
            volume_col = f'{ticker}_Volume'
            
            if close_col not in df.columns:
                self.logger.warning(f"Ticker {ticker} not found in data")
                continue
            
            # Extract series
            close = df[close_col]
            high = df[high_col]
            low = df[low_col]
            open_price = df[open_col]
            volume = df.get(volume_col)
            
            # ----------------------------------------------------------------
            # TREND INDICATORS
            # ----------------------------------------------------------------
            
            # Simple Moving Averages
            for period in self.tech_config['sma_periods']:
                df[f'{ticker}_SMA_{period}'] = close.rolling(window=period).mean()
                self._add_metadata(f'{ticker}_SMA_{period}', 'trend', 
                                  f'{period}-day Simple Moving Average')
            
            # Exponential Moving Averages
            for period in self.tech_config['ema_periods']:
                df[f'{ticker}_EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
                self._add_metadata(f'{ticker}_EMA_{period}', 'trend',
                                  f'{period}-day Exponential Moving Average')
            
            # Price vs SMA (normalized distance)
            for period in self.tech_config['sma_periods']:
                sma = df[f'{ticker}_SMA_{period}']
                df[f'{ticker}_Price_vs_SMA_{period}'] = safe_divide(close - sma, sma) * 100
                self._add_metadata(f'{ticker}_Price_vs_SMA_{period}', 'trend',
                                  f'Price distance from SMA{period} in %')
            
            # SMA Crossovers (signals)
            df[f'{ticker}_SMA_Cross_20_50'] = (
                (df[f'{ticker}_SMA_20'] > df[f'{ticker}_SMA_50']).astype(int)
            )
            df[f'{ticker}_SMA_Cross_50_200'] = (
                (df[f'{ticker}_SMA_50'] > df[f'{ticker}_SMA_200']).astype(int)
            )
            self._add_metadata(f'{ticker}_SMA_Cross_20_50', 'trend',
                              'SMA20 > SMA50 (1=bullish)')
            self._add_metadata(f'{ticker}_SMA_Cross_50_200', 'trend',
                              'SMA50 > SMA200 (1=bullish, golden cross)')
            
            # Slope of SMA (trend direction)
            slope_short_period = self.tech_config.get('slope_short_period', 5)
            slope_long_period = self.tech_config.get('slope_long_period', 10)
            df[f'{ticker}_SMA_20_Slope'] = df[f'{ticker}_SMA_20'].diff(slope_short_period) / slope_short_period
            df[f'{ticker}_SMA_50_Slope'] = df[f'{ticker}_SMA_50'].diff(slope_long_period) / slope_long_period
            self._add_metadata(f'{ticker}_SMA_20_Slope', 'trend', f'{slope_short_period}-day slope of SMA20')
            self._add_metadata(f'{ticker}_SMA_50_Slope', 'trend', f'{slope_long_period}-day slope of SMA50')
            
            # ----------------------------------------------------------------
            # MOMENTUM INDICATORS
            # ----------------------------------------------------------------
            
            # RSI
            rsi_period = self.tech_config['rsi_period']
            df[f'{ticker}_RSI'] = ta.rsi(close, length=rsi_period)
            self._add_metadata(f'{ticker}_RSI', 'momentum', f'{rsi_period}-day RSI')
            
            # MACD
            macd_fast, macd_slow, macd_signal = self.tech_config['macd']
            macd_result = ta.macd(close, fast=macd_fast, slow=macd_slow, signal=macd_signal)
            if macd_result is not None:
                df[f'{ticker}_MACD'] = macd_result.iloc[:, 0]
                df[f'{ticker}_MACD_Signal'] = macd_result.iloc[:, 1]
                df[f'{ticker}_MACD_Histogram'] = macd_result.iloc[:, 2]
                self._add_metadata(f'{ticker}_MACD', 'momentum', 'MACD line')
                self._add_metadata(f'{ticker}_MACD_Signal', 'momentum', 'MACD signal line')
                self._add_metadata(f'{ticker}_MACD_Histogram', 'momentum', 'MACD histogram')
            
            # Stochastic Oscillator
            stoch_k, stoch_d, stoch_smooth = self.tech_config['stochastic']
            stoch_result = ta.stoch(high, low, close, k=stoch_k, d=stoch_d, smooth_k=stoch_smooth)
            if stoch_result is not None:
                df[f'{ticker}_Stoch_K'] = stoch_result.iloc[:, 0]
                df[f'{ticker}_Stoch_D'] = stoch_result.iloc[:, 1]
                self._add_metadata(f'{ticker}_Stoch_K', 'momentum', 'Stochastic %K')
                self._add_metadata(f'{ticker}_Stoch_D', 'momentum', 'Stochastic %D')
            
            # Rate of Change (ROC)
            df[f'{ticker}_ROC_10'] = ta.roc(close, length=10)
            df[f'{ticker}_ROC_20'] = ta.roc(close, length=20)
            self._add_metadata(f'{ticker}_ROC_10', 'momentum', '10-day Rate of Change')
            self._add_metadata(f'{ticker}_ROC_20', 'momentum', '20-day Rate of Change')
            
            # ----------------------------------------------------------------
            # VOLATILITY INDICATORS
            # ----------------------------------------------------------------
            
            # ATR
            atr_period = self.tech_config['atr_period']
            df[f'{ticker}_ATR'] = ta.atr(high, low, close, length=atr_period)
            self._add_metadata(f'{ticker}_ATR', 'volatility', f'{atr_period}-day ATR')
            
            # ATR as percentage of price
            df[f'{ticker}_ATR_Pct'] = safe_divide(df[f'{ticker}_ATR'], close) * 100
            self._add_metadata(f'{ticker}_ATR_Pct', 'volatility', 'ATR as % of price')
            
            # Bollinger Bands
            bb_period, bb_std = self.tech_config['bollinger']
            bb_result = ta.bbands(close, length=bb_period, std=bb_std)
            if bb_result is not None:
                df[f'{ticker}_BB_Upper'] = bb_result.iloc[:, 0]
                df[f'{ticker}_BB_Middle'] = bb_result.iloc[:, 1]
                df[f'{ticker}_BB_Lower'] = bb_result.iloc[:, 2]
                df[f'{ticker}_BB_Width'] = bb_result.iloc[:, 3] if bb_result.shape[1] > 3 else (
                    bb_result.iloc[:, 0] - bb_result.iloc[:, 2]
                )
                # %B (position within bands)
                df[f'{ticker}_BB_PctB'] = safe_divide(
                    close - df[f'{ticker}_BB_Lower'],
                    df[f'{ticker}_BB_Upper'] - df[f'{ticker}_BB_Lower']
                )
                self._add_metadata(f'{ticker}_BB_Upper', 'volatility', 'Bollinger Upper Band')
                self._add_metadata(f'{ticker}_BB_Lower', 'volatility', 'Bollinger Lower Band')
                self._add_metadata(f'{ticker}_BB_Width', 'volatility', 'Bollinger Band Width')
                self._add_metadata(f'{ticker}_BB_PctB', 'volatility', 'Bollinger %B (0-1)')
            
            # ADX (trend strength)
            adx_period = self.tech_config['adx_period']
            adx_result = ta.adx(high, low, close, length=adx_period)
            if adx_result is not None:
                df[f'{ticker}_ADX'] = adx_result.iloc[:, 0]
                df[f'{ticker}_DI_Plus'] = adx_result.iloc[:, 1]
                df[f'{ticker}_DI_Minus'] = adx_result.iloc[:, 2]
                self._add_metadata(f'{ticker}_ADX', 'volatility', f'{adx_period}-day ADX')
                self._add_metadata(f'{ticker}_DI_Plus', 'volatility', 'Directional +DI')
                self._add_metadata(f'{ticker}_DI_Minus', 'volatility', 'Directional -DI')
            
            # ----------------------------------------------------------------
            # VOLUME INDICATORS (if available)
            # ----------------------------------------------------------------
            
            if volume is not None and not volume.isna().all():
                # OBV
                df[f'{ticker}_OBV'] = ta.obv(close, volume)
                self._add_metadata(f'{ticker}_OBV', 'volume', 'On-Balance Volume')
                
                # Volume SMA ratio
                volume_sma_period = self.tech_config.get('volume_sma_period', 20)
                vol_sma = volume.rolling(window=volume_sma_period).mean()
                df[f'{ticker}_Volume_Ratio'] = safe_divide(volume, vol_sma)
                self._add_metadata(f'{ticker}_Volume_Ratio', 'volume', 
                                  f'Volume / {volume_sma_period}-day avg volume')
            
            # ----------------------------------------------------------------
            # PRICE PATTERNS
            # ----------------------------------------------------------------
            
            # High/low window from config (default 20)
            price_range_period = self.tech_config.get('price_range_period', 20)
            high_n = high.rolling(window=price_range_period).max()
            low_n = low.rolling(window=price_range_period).min()
            df[f'{ticker}_High_{price_range_period}'] = high_n
            df[f'{ticker}_Low_{price_range_period}'] = low_n
            
            # Price position in range
            df[f'{ticker}_Price_Position'] = safe_divide(
                close - low_n, high_n - low_n
            )
            self._add_metadata(f'{ticker}_Price_Position', 'pattern',
                              f'Price position in {price_range_period}-day range (0=low, 1=high)')
            
            # Distance to support/resistance
            df[f'{ticker}_Dist_to_High'] = safe_divide(high_n - close, close) * 100
            df[f'{ticker}_Dist_to_Low'] = safe_divide(close - low_n, close) * 100
            self._add_metadata(f'{ticker}_Dist_to_High', 'pattern', 
                              f'Distance to {price_range_period}-day high (%)')
            self._add_metadata(f'{ticker}_Dist_to_Low', 'pattern',
                              f'Distance to {price_range_period}-day low (%)')
            
            # ----------------------------------------------------------------
            # RETURNS
            # ----------------------------------------------------------------
            
            # Daily returns
            df[f'{ticker}_Return_1d'] = close.pct_change(1) * 100
            df[f'{ticker}_Return_5d'] = close.pct_change(5) * 100
            df[f'{ticker}_Return_20d'] = close.pct_change(20) * 100
            self._add_metadata(f'{ticker}_Return_1d', 'return', '1-day return (%)')
            self._add_metadata(f'{ticker}_Return_5d', 'return', '5-day return (%)')
            self._add_metadata(f'{ticker}_Return_20d', 'return', '20-day return (%)')
            
            # Rolling volatility
            df[f'{ticker}_Volatility_20'] = df[f'{ticker}_Return_1d'].rolling(20).std()
            self._add_metadata(f'{ticker}_Volatility_20', 'volatility',
                              '20-day rolling volatility of returns')
        
        self.logger.info(f"Added technical indicators. New shape: {df.shape}")
        return df
    
    # ========================================================================
    # MULTITIMEFRAME FEATURES
    # ========================================================================
    
    @timer_decorator
    def add_multitimeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weekly and monthly aggregation features.
        
        Args:
            df: DataFrame with daily data.
        
        Returns:
            DataFrame with multitimeframe features.
        """
        self.logger.info("Adding multitimeframe features...")
        
        for ticker in self.main_tickers:
            close_col = f'{ticker}_Close'
            
            if close_col not in df.columns:
                continue
            
            close = df[close_col]
            
            # Weekly features (5-day approximation)
            weekly_sma_periods = self.config['features']['multitimeframe']['weekly_sma']
            
            for period in weekly_sma_periods:
                # Weekly SMA (multiply period by 5 for daily data)
                daily_period = period * 5
                if daily_period <= len(df):
                    df[f'{ticker}_Weekly_SMA_{period}'] = close.rolling(
                        window=daily_period
                    ).mean()
                    self._add_metadata(f'{ticker}_Weekly_SMA_{period}', 'multitimeframe',
                                      f'Weekly {period}-period SMA (daily equivalent)')
            
            # Trend alignment: 1 if short SMA > long SMA
            if (f'{ticker}_Weekly_SMA_50' in df.columns and 
                f'{ticker}_Weekly_SMA_200' in df.columns):
                df[f'{ticker}_Weekly_Trend_Align'] = (
                    df[f'{ticker}_Weekly_SMA_50'] > df[f'{ticker}_Weekly_SMA_200']
                ).astype(int) * 2 - 1  # Convert to -1 or 1
                self._add_metadata(f'{ticker}_Weekly_Trend_Align', 'multitimeframe',
                                  'Weekly trend alignment (1=bullish, -1=bearish)')
            
            # Monthly returns and volatility (21-day approximation)
            df[f'{ticker}_Monthly_Return'] = close.pct_change(21) * 100
            df[f'{ticker}_Monthly_Volatility'] = (
                df[f'{ticker}_Return_1d'].rolling(21).std() * np.sqrt(21)
            )
            self._add_metadata(f'{ticker}_Monthly_Return', 'multitimeframe',
                              '21-day (monthly) return (%)')
            self._add_metadata(f'{ticker}_Monthly_Volatility', 'multitimeframe',
                              '21-day annualized volatility')
        
        self.logger.info("Multitimeframe features added")
        return df
    
    # ========================================================================
    # TIER 2: INTERMARKET FEATURES
    # ========================================================================
    
    @timer_decorator
    def add_intermarket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add intermarket features (DXY, VIX, yields, oil).
        
        Args:
            df: DataFrame with all ticker data.
        
        Returns:
            DataFrame with intermarket features.
        """
        self.logger.info("Adding intermarket features...")
        
        correlation_window = self.intermarket_config['correlation_window']
        lag_periods = self.intermarket_config['lag_periods']
        
        # ====================================================================
        # DXY (US Dollar Index)
        # ====================================================================
        
        dxy_col = 'DX_Y_NYB_Close'  # Cleaned ticker name
        if dxy_col in df.columns:
            dxy = df[dxy_col]
            
            # DXY returns
            for lag in lag_periods:
                df[f'DXY_Return_{lag}d'] = dxy.pct_change(lag) * 100
                self._add_metadata(f'DXY_Return_{lag}d', 'intermarket',
                                  f'DXY {lag}-day return (%)')
            
            # Rolling correlations with main pairs
            for ticker in self.main_tickers:
                return_col = f'{ticker}_Return_1d'
                if return_col in df.columns:
                    df[f'{ticker}_DXY_Corr'] = (
                        df[return_col].rolling(correlation_window).corr(dxy.pct_change())
                    )
                    self._add_metadata(f'{ticker}_DXY_Corr', 'intermarket',
                                      f'{ticker} vs DXY {correlation_window}-day correlation')
            
            # Relative strength vs DXY
            dxy_norm = dxy / dxy.iloc[0] * 100  # Normalize to 100
            for ticker in self.main_tickers:
                close_col = f'{ticker}_Close'
                if close_col in df.columns:
                    pair_norm = df[close_col] / df[close_col].iloc[0] * 100
                    df[f'{ticker}_vs_DXY'] = pair_norm - dxy_norm
                    self._add_metadata(f'{ticker}_vs_DXY', 'intermarket',
                                      f'{ticker} relative strength vs DXY')
        
        # ====================================================================
        # VIX (Volatility Index)
        # ====================================================================
        
        vix_col = 'VIX_Close'
        if vix_col in df.columns:
            vix = df[vix_col]
            
            # VIX level
            df['VIX_Level'] = vix
            self._add_metadata('VIX_Level', 'intermarket', 'VIX current value')
            
            # VIX changes
            for lag in lag_periods:
                df[f'VIX_Change_{lag}d'] = vix.diff(lag)
                self._add_metadata(f'VIX_Change_{lag}d', 'intermarket',
                                  f'VIX {lag}-day change')
            
            # VIX regime (low < 15, medium 15-25, high > 25)
            df['VIX_Regime'] = pd.cut(
                vix,
                bins=[0, 15, 25, np.inf],
                labels=[0, 1, 2]  # 0=low, 1=medium, 2=high
            ).astype(float)
            self._add_metadata('VIX_Regime', 'intermarket',
                              'VIX regime (0=low, 1=medium, 2=high)')
            
            # VIX percentile (rolling)
            vix_percentile_window = self.intermarket_config.get('vix_percentile_window', 252)
            df['VIX_Percentile'] = vix.rolling(vix_percentile_window).apply(
                lambda x: (x.iloc[-1] > x).mean() * 100 if len(x) > 0 else 50
            )
            self._add_metadata('VIX_Percentile', 'intermarket',
                              f'VIX percentile over {vix_percentile_window} days')
        
        # ====================================================================
        # TNX (US 10Y Treasury Yield)
        # ====================================================================
        
        tnx_col = 'TNX_Close'
        if tnx_col in df.columns:
            tnx = df[tnx_col]
            
            # TNX level
            df['TNX_Level'] = tnx
            self._add_metadata('TNX_Level', 'intermarket', '10Y Treasury Yield')
            
            # TNX changes
            for lag in lag_periods:
                df[f'TNX_Change_{lag}d'] = tnx.diff(lag)
                self._add_metadata(f'TNX_Change_{lag}d', 'intermarket',
                                  f'10Y yield {lag}-day change')
        
        # ====================================================================
        # Oil (CL=F)
        # ====================================================================
        
        oil_col = 'CL_F_Close'
        if oil_col in df.columns:
            oil = df[oil_col]
            
            # Oil returns
            for lag in lag_periods:
                df[f'Oil_Return_{lag}d'] = oil.pct_change(lag) * 100
                self._add_metadata(f'Oil_Return_{lag}d', 'intermarket',
                                  f'Oil {lag}-day return (%)')
            
            # Rolling correlations with main pairs
            for ticker in self.main_tickers:
                return_col = f'{ticker}_Return_1d'
                if return_col in df.columns:
                    df[f'{ticker}_Oil_Corr'] = (
                        df[return_col].rolling(correlation_window).corr(oil.pct_change())
                    )
                    self._add_metadata(f'{ticker}_Oil_Corr', 'intermarket',
                                      f'{ticker} vs Oil {correlation_window}-day correlation')
        
        # ====================================================================
        # Cross-correlations between main pairs
        # ====================================================================
        
        for i, ticker1 in enumerate(self.main_tickers):
            for ticker2 in self.main_tickers[i+1:]:
                ret1 = f'{ticker1}_Return_1d'
                ret2 = f'{ticker2}_Return_1d'
                if ret1 in df.columns and ret2 in df.columns:
                    df[f'{ticker1}_{ticker2}_Corr'] = (
                        df[ret1].rolling(correlation_window).corr(df[ret2])
                    )
                    self._add_metadata(f'{ticker1}_{ticker2}_Corr', 'intermarket',
                                      f'{ticker1} vs {ticker2} {correlation_window}-day correlation')
        
        self.logger.info("Intermarket features added")
        return df
    
    # ========================================================================
    # DERIVED FEATURES
    # ========================================================================
    
    @timer_decorator
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived and composite features.
        
        Args:
            df: DataFrame with technical and intermarket features.
        
        Returns:
            DataFrame with derived features.
        """
        self.logger.info("Adding derived features...")
        
        for ticker in self.main_tickers:
            close_col = f'{ticker}_Close'
            
            if close_col not in df.columns:
                continue
            
            close = df[close_col]
            
            # ----------------------------------------------------------------
            # Mean Reversion Indicators
            # ----------------------------------------------------------------
            
            sma_20 = df.get(f'{ticker}_SMA_20')
            if sma_20 is not None:
                # Z-score from 20-day mean
                std_20 = close.rolling(20).std()
                df[f'{ticker}_ZScore'] = safe_divide(close - sma_20, std_20)
                self._add_metadata(f'{ticker}_ZScore', 'derived',
                                  'Z-score from 20-day mean')
            
            # ----------------------------------------------------------------
            # Volatility Percentile
            # ----------------------------------------------------------------
            
            atr_col = f'{ticker}_ATR'
            if atr_col in df.columns:
                atr = df[atr_col]
                # Current ATR percentile vs rolling history
                percentile_window = self.tech_config.get('percentile_window', 50)
                df[f'{ticker}_ATR_Percentile'] = atr.rolling(percentile_window).apply(
                    lambda x: (x.iloc[-1] > x).mean() * 100 if len(x) > 0 else 50
                )
                self._add_metadata(f'{ticker}_ATR_Percentile', 'derived',
                                  f'ATR percentile over {percentile_window} days (0-100)')
            
            # ----------------------------------------------------------------
            # Trend Strength Score
            # ----------------------------------------------------------------
            
            adx_col = f'{ticker}_ADX'
            if adx_col in df.columns:
                # Trend strength: ADX weighted by direction
                di_plus = df.get(f'{ticker}_DI_Plus', 0)
                di_minus = df.get(f'{ticker}_DI_Minus', 0)
                direction = np.sign(di_plus - di_minus)
                df[f'{ticker}_Trend_Strength'] = df[adx_col] * direction
                self._add_metadata(f'{ticker}_Trend_Strength', 'derived',
                                  'ADX × direction (positive=uptrend)')
            
            # ----------------------------------------------------------------
            # Momentum Composite
            # ----------------------------------------------------------------
            
            rsi_col = f'{ticker}_RSI'
            stoch_col = f'{ticker}_Stoch_K'
            
            if rsi_col in df.columns and stoch_col in df.columns:
                # Normalize RSI and Stochastic to same scale and average
                rsi_norm = (df[rsi_col] - 50) / 50  # -1 to 1
                stoch_norm = (df[stoch_col] - 50) / 50  # -1 to 1
                df[f'{ticker}_Momentum_Composite'] = (rsi_norm + stoch_norm) / 2
                self._add_metadata(f'{ticker}_Momentum_Composite', 'derived',
                                  'Avg of normalized RSI and Stochastic (-1 to 1)')
            
            # ----------------------------------------------------------------
            # Volatility State
            # ----------------------------------------------------------------
            
            bb_width = df.get(f'{ticker}_BB_Width')
            if bb_width is not None:
                # BB width percentile
                percentile_window = self.tech_config.get('percentile_window', 50)
                df[f'{ticker}_BB_Width_Percentile'] = bb_width.rolling(percentile_window).apply(
                    lambda x: (x.iloc[-1] > x).mean() * 100 if len(x) > 0 else 50
                )
                self._add_metadata(f'{ticker}_BB_Width_Percentile', 'derived',
                                  f'BB Width percentile over {percentile_window} days (0-100)')
        
        self.logger.info("Derived features added")
        return df
    
    # ========================================================================
    # TARGET VARIABLE
    # ========================================================================
    
    @timer_decorator
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction.
        
        Creates:
        - target_next_day: Binary (1 if next day's close > today's close)
        - target_return: Continuous (next day's return)
        
        CRITICAL: All features must be shifted to prevent look-ahead bias.
        
        Args:
            df: DataFrame with features.
        
        Returns:
            DataFrame with target variables.
        """
        self.logger.info("Creating target variables...")
        
        for ticker in self.main_tickers:
            close_col = f'{ticker}_Close'
            
            if close_col not in df.columns:
                continue
            
            close = df[close_col]
            
            # Next day's return (for labeling)
            next_return = close.pct_change(1).shift(-1) * 100
            
            # Binary target: 1 if next day is up
            df[f'{ticker}_Target_Direction'] = (next_return > 0).astype(int)
            self._add_metadata(f'{ticker}_Target_Direction', 'target',
                              'Next day direction (1=up, 0=down)')
            
            # Continuous target: next day's return
            df[f'{ticker}_Target_Return'] = next_return
            self._add_metadata(f'{ticker}_Target_Return', 'target',
                              'Next day return (%)')
        
        self.logger.info("Target variables created")
        return df
    
    # ========================================================================
    # FEATURE VALIDATION
    # ========================================================================
    
    def validate_no_lookahead(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that features don't contain look-ahead bias.
        
        This check ensures that:
        1. Target variables are properly shifted
        2. No perfect correlations exist between features and targets
        
        Args:
            df: DataFrame with features and targets.
        
        Returns:
            Validation report dictionary.
        
        Raises:
            ValueError: If look-ahead bias is detected.
        """
        self.logger.info("Validating for look-ahead bias...")
        
        report = {
            'is_valid': True,
            'issues': [],
            'perfect_correlations': []
        }
        
        # Check each main ticker's target
        for ticker in self.main_tickers:
            target_col = f'{ticker}_Target_Direction'
            
            if target_col not in df.columns:
                continue
            
            target = df[target_col]
            
            # Get feature columns for this ticker
            feature_cols = [
                c for c in df.columns 
                if c.startswith(ticker) and 'Target' not in c
            ]
            
            for col in feature_cols:
                try:
                    # Check correlation at time t (should not be perfect)
                    corr = df[col].corr(target)
                    if abs(corr) > 0.95:
                        report['is_valid'] = False
                        report['perfect_correlations'].append({
                            'feature': col,
                            'target': target_col,
                            'correlation': round(corr, 4)
                        })
                except Exception:
                    pass
        
        if report['perfect_correlations']:
            report['issues'].append(
                f"Found {len(report['perfect_correlations'])} suspiciously high correlations"
            )
            self.logger.warning(f"Potential look-ahead bias detected: {report['perfect_correlations']}")
        else:
            self.logger.info("No look-ahead bias detected")
        
        return report
    
    # ========================================================================
    # FEATURE MATRIX GENERATION
    # ========================================================================
    
    @timer_decorator
    def generate_feature_matrix(
        self,
        df: pd.DataFrame,
        lag_features: bool = True,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Generate final ML-ready feature matrix.
        
        Args:
            df: DataFrame with all features.
            lag_features: Whether to lag features by 1 day (prevent look-ahead).
            drop_na: Whether to drop rows with NaN values.
        
        Returns:
            Clean feature matrix ready for modeling.
        """
        self.logger.info("Generating final feature matrix...")
        
        # Separate target columns
        target_cols = [c for c in df.columns if 'Target' in c]
        feature_cols = [c for c in df.columns if 'Target' not in c]
        
        # Copy to avoid modifying original
        result = df.copy()
        
        # Lag features by 1 day to prevent look-ahead bias
        if lag_features:
            self.logger.info("Lagging features by 1 day...")
            for col in feature_cols:
                # Keep original OHLCV columns for reference
                if any(x in col for x in ['_Open', '_High', '_Low', '_Close', '_Volume']):
                    # Only if it's a raw price column, not an indicator
                    if not any(ind in col for ind in ['SMA', 'EMA', 'RSI', 'MACD', 'ATR', 
                                                       'BB', 'Stoch', 'ADX', 'Return', 
                                                       'ROC', 'OBV', 'Volatility']):
                        continue
                result[col] = result[col].shift(1)
        
        # Handle NaN values
        if drop_na:
            initial_rows = len(result)
            result = result.dropna()
            dropped = initial_rows - len(result)
            self.logger.info(f"Dropped {dropped} rows with NaN ({dropped/initial_rows*100:.1f}%)")
        
        self.features = result
        self.logger.info(f"Final feature matrix shape: {result.shape}")
        
        return result
    
    # ========================================================================
    # SAVE AND LOAD
    # ========================================================================
    
    @timer_decorator
    def save_features(
        self,
        df: Optional[pd.DataFrame] = None,
        path: Optional[str] = None
    ) -> None:
        """
        Save feature DataFrame and metadata.
        
        Args:
            df: DataFrame to save. Uses self.features if None.
            path: Output path. Uses config path if None.
        """
        df = df if df is not None else self.features
        path = Path(path) if path else self.features_path
        
        if df is None:
            raise ValueError("No features to save")
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save features
        df.to_parquet(path, compression='snappy')
        self.logger.info(f"Saved features to {path}")
        
        # Save metadata
        metadata_path = path.parent / 'feature_metadata.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(self.feature_metadata, f, indent=2)
        self.logger.info(f"Saved feature metadata to {metadata_path}")
    
    def load_features(self, path: Optional[str] = None) -> pd.DataFrame:
        """
        Load feature DataFrame.
        
        Args:
            path: Path to load from. Uses config path if None.
        
        Returns:
            Loaded feature DataFrame.
        """
        path = Path(path) if path else self.features_path
        
        if not path.exists():
            raise FileNotFoundError(f"Features file not found: {path}")
        
        self.features = pd.read_parquet(path)
        self.logger.info(f"Loaded features: {self.features.shape}")
        return self.features
    
    # ========================================================================
    # CORRELATION ANALYSIS
    # ========================================================================
    
    def get_feature_correlations(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        threshold: float = 0.7
    ) -> pd.DataFrame:
        """
        Compute feature correlation matrix.
        
        Args:
            df: Feature DataFrame. Uses self.features if None.
            target_col: If specified, show correlations with this target.
            threshold: Highlight correlations above this threshold.
        
        Returns:
            Correlation DataFrame.
        """
        df = df if df is not None else self.features
        
        if df is None:
            raise ValueError("No features available")
        
        # Compute correlation matrix
        corr_matrix = df.corr()
        
        if target_col and target_col in df.columns:
            # Sort by correlation with target
            target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
            return target_corr.to_frame()
        
        return corr_matrix
    
    # ========================================================================
    # FULL PIPELINE
    # ========================================================================
    
    @timer_decorator
    def run_full_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Steps:
        1. Add technical indicators
        2. Add multitimeframe features
        3. Add intermarket features
        4. Add derived features
        5. Create target variables
        6. Generate feature matrix
        7. Validate and save
        
        Args:
            df: Input price DataFrame (wide format).
        
        Returns:
            Complete feature matrix.
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Input shape: {df.shape}")
        
        # Store original data
        self.data = df.copy()
        
        # Step 1: Technical indicators
        df = self.add_technical_indicators(df)
        
        # Step 2: Multitimeframe features
        df = self.add_multitimeframe_features(df)
        
        # Step 3: Intermarket features
        df = self.add_intermarket_features(df)
        
        # Step 4: Derived features
        df = self.add_derived_features(df)
        
        # Step 5: Target variables
        df = self.create_target(df)
        
        # Step 6: Generate feature matrix
        df = self.generate_feature_matrix(df, lag_features=True, drop_na=True)
        
        # Step 7: Validate
        validation = self.validate_no_lookahead(df)
        if not validation['is_valid']:
            self.logger.warning(f"Validation issues: {validation['issues']}")
        
        # Save
        self.save_features(df)
        
        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("FEATURE ENGINEERING COMPLETE")
        self.logger.info(f"Final shape: {df.shape}")
        self.logger.info(f"Features: {len([c for c in df.columns if 'Target' not in c])}")
        self.logger.info(f"Targets: {len([c for c in df.columns if 'Target' in c])}")
        self.logger.info("=" * 60)
        
        return df
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _add_metadata(self, name: str, category: str, description: str) -> None:
        """
        Add metadata for a feature.
        
        Args:
            name: Feature name.
            category: Feature category (trend, momentum, etc.).
            description: Feature description.
        """
        self.feature_metadata[name] = {
            'category': category,
            'description': description
        }
    
    def get_feature_list(self, category: Optional[str] = None) -> List[str]:
        """
        Get list of feature names, optionally filtered by category.
        
        Args:
            category: Filter by category (trend, momentum, etc.).
        
        Returns:
            List of feature names.
        """
        if category:
            return [
                name for name, meta in self.feature_metadata.items()
                if meta.get('category') == category
            ]
        return list(self.feature_metadata.keys())


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def align_features_for_model(
    df: pd.DataFrame,
    expected_features: List[str],
    fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Align DataFrame features to match model's expected feature set.
    
    This function ensures the DataFrame has exactly the same features
    in the same order as the model expects, adding missing features
    with fill values and removing extra features.
    
    Args:
        df: Input DataFrame with features.
        expected_features: List of feature names the model expects.
        fill_value: Value to use for missing features (default 0.0).
    
    Returns:
        DataFrame with aligned features in correct order.
    
    Example:
        >>> model_features = model.get_booster().feature_names
        >>> X_test_aligned = align_features_for_model(X_test, model_features)
        >>> predictions = model.predict(X_test_aligned)
    """
    df_aligned = df.copy()
    
    # Find missing features
    current_features = set(df_aligned.columns)
    expected_set = set(expected_features)
    
    missing_features = expected_set - current_features
    extra_features = current_features - expected_set
    
    # Add missing features with fill value
    if missing_features:
        print(f"⚠️ Adding {len(missing_features)} missing features with value {fill_value}")
        for feat in missing_features:
            df_aligned[feat] = fill_value
    
    # Report extra features (they will be dropped)
    if extra_features:
        print(f"ℹ️ Removing {len(extra_features)} extra features not in model")
    
    # Select only expected features in correct order
    df_aligned = df_aligned[expected_features]
    
    return df_aligned


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar/time-based features to a DataFrame with DatetimeIndex.
    
    Args:
        df: DataFrame with DatetimeIndex.
    
    Returns:
        DataFrame with added calendar features.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    result = df.copy()
    
    # Day of week (0=Monday, 4=Friday)
    result['day_of_week'] = df.index.dayofweek
    
    # One-hot encode trading days
    for i in range(5):  # Only weekdays 0-4
        result[f'is_day_{i}'] = (df.index.dayofweek == i).astype(int)
    
    # Month features
    result['month'] = df.index.month
    result['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    result['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Quarter
    result['quarter'] = df.index.quarter
    
    # Week of year
    result['week_of_year'] = df.index.isocalendar().week.astype(int)
    
    # End/start of month/year
    result['is_year_end'] = ((df.index.month == 12) & (df.index.day >= 20)).astype(int)
    result['day_of_month'] = df.index.day
    result['is_month_end'] = (df.index.day >= 25).astype(int)
    result['is_month_start'] = (df.index.day <= 5).astype(int)
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    from src.utils import setup_logging
    from src.data_pipeline import DataPipeline
    
    # Setup logging
    logger = setup_logging('logs/feature_engineering.log', level='INFO')
    
    # Load combined data
    pipeline = DataPipeline()
    combined_data = pipeline.load_combined_data()
    
    # Run feature engineering
    engine = FeatureEngine()
    features = engine.run_full_pipeline(combined_data)
    
    print(f"\nFeature engineering complete!")
    print(f"Feature matrix shape: {features.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")
