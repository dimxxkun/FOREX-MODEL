"""
Feature Engineering V2 - Advanced features for improved signal quality.

This module adds sophisticated features including:
- Market regime detection
- Momentum quality indicators
- Advanced volatility measures
- Cross-asset signals
- Time-based features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AdvancedFeatureEngine:
    """
    Enhanced feature engineering for better signal quality.
    
    New features include:
    - Market regime detection (trending/ranging/volatile)
    - Momentum quality (acceleration, divergences)
    - Cross-pair relative strength
    - Volatility regimes
    - Time-based seasonality
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the advanced feature engine.
        
        Args:
            config: Configuration dictionary with feature parameters
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
        
    def add_all_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all advanced features to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data and basic features
            
        Returns:
            DataFrame with all advanced features added
        """
        logger.info("Adding advanced features (V2)...")
        
        df = df.copy()
        
        # Add regime detection features
        df = self.add_regime_features(df)
        
        # Add momentum quality features
        df = self.add_momentum_quality_features(df)
        
        # Add advanced volatility features
        df = self.add_advanced_volatility_features(df)
        
        # Add cross-asset signals
        df = self.add_cross_asset_features(df)
        
        # Add time-based features
        df = self.add_time_features(df)
        
        # Add pattern recognition features
        df = self.add_pattern_features(df)
        
        logger.info(f"Added {len([c for c in df.columns if '_v2' in c.lower() or 'regime' in c.lower()])} advanced features")
        
        return df
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime detection features.
        
        Regimes:
        - Trending (ADX > 25)
        - Ranging (ADX < 15)
        - High/Low volatility based on ATR percentile
        """
        logger.info("Adding regime detection features...")
        
        # Get unique tickers
        tickers = df['Ticker'].unique() if 'Ticker' in df.columns else ['default']
        
        for ticker in tickers:
            mask = df['Ticker'] == ticker if 'Ticker' in df.columns else slice(None)
            ticker_data = df.loc[mask].copy()
            
            # ADX-based trend strength (if ADX column exists)
            adx_col = f'{ticker}_ADX_14' if f'{ticker}_ADX_14' in df.columns else 'ADX_14'
            if adx_col in df.columns:
                df.loc[mask, f'{ticker}_regime_trending'] = (df.loc[mask, adx_col] > self.regime_config['adx_trend_threshold']).astype(int)
                df.loc[mask, f'{ticker}_regime_ranging'] = (df.loc[mask, adx_col] < self.regime_config['adx_weak_threshold']).astype(int)
                df.loc[mask, f'{ticker}_regime_weak_trend'] = ((df.loc[mask, adx_col] >= self.regime_config['adx_weak_threshold']) & 
                                                               (df.loc[mask, adx_col] <= self.regime_config['adx_trend_threshold'])).astype(int)
            
            # ATR percentile for volatility regime
            atr_col = f'{ticker}_ATR_14' if f'{ticker}_ATR_14' in df.columns else 'ATR_14'
            if atr_col in df.columns:
                atr_percentile = df.loc[mask, atr_col].rolling(50).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
                )
                df.loc[mask, f'{ticker}_atr_percentile'] = atr_percentile
                df.loc[mask, f'{ticker}_regime_high_vol'] = (atr_percentile > self.regime_config['atr_high_percentile']).astype(int)
                df.loc[mask, f'{ticker}_regime_low_vol'] = (atr_percentile < self.regime_config['atr_low_percentile']).astype(int)
            
            # Hurst exponent approximation for trend vs mean-reversion
            close_col = f'{ticker}_Close' if f'{ticker}_Close' in df.columns else 'Close'
            if close_col in df.columns:
                df.loc[mask, f'{ticker}_hurst_approx'] = self._calculate_hurst_approximation(df.loc[mask, close_col])
                df.loc[mask, f'{ticker}_regime_mean_reverting'] = (df.loc[mask, f'{ticker}_hurst_approx'] < 0.4).astype(int)
                df.loc[mask, f'{ticker}_regime_persistent'] = (df.loc[mask, f'{ticker}_hurst_approx'] > 0.6).astype(int)
        
        return df
    
    def _calculate_hurst_approximation(self, series: pd.Series, window: int = 100) -> pd.Series:
        """
        Calculate simplified Hurst exponent approximation.
        
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending/persistent
        """
        def hurst_rs(x):
            if len(x) < 20:
                return 0.5
            try:
                # Calculate returns
                returns = np.diff(np.log(x))
                if len(returns) < 10:
                    return 0.5
                    
                # Mean and cumulative deviation
                mean_ret = np.mean(returns)
                cumdev = np.cumsum(returns - mean_ret)
                
                # Range and standard deviation
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(returns, ddof=1)
                
                if S == 0 or R == 0:
                    return 0.5
                    
                # R/S ratio
                RS = R / S
                
                # Hurst approximation
                H = np.log(RS) / np.log(len(returns))
                return np.clip(H, 0, 1)
            except:
                return 0.5
        
        return series.rolling(window).apply(hurst_rs, raw=True)
    
    def add_momentum_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum quality indicators including divergences.
        """
        logger.info("Adding momentum quality features...")
        
        tickers = df['Ticker'].unique() if 'Ticker' in df.columns else ['default']
        
        for ticker in tickers:
            mask = df['Ticker'] == ticker if 'Ticker' in df.columns else slice(None)
            
            close_col = f'{ticker}_Close' if f'{ticker}_Close' in df.columns else 'Close'
            rsi_col = f'{ticker}_RSI_14' if f'{ticker}_RSI_14' in df.columns else 'RSI_14'
            macd_col = f'{ticker}_MACD_12_26_9' if f'{ticker}_MACD_12_26_9' in df.columns else 'MACD_12_26_9'
            macd_hist_col = f'{ticker}_MACDh_12_26_9' if f'{ticker}_MACDh_12_26_9' in df.columns else 'MACDh_12_26_9'
            
            # RSI Divergence Detection
            if close_col in df.columns and rsi_col in df.columns:
                # Price higher high but RSI lower high = bearish divergence
                price_hh = df.loc[mask, close_col].rolling(20).apply(
                    lambda x: 1 if x.iloc[-1] > x.iloc[:-1].max() else 0, raw=False
                )
                rsi_lh = df.loc[mask, rsi_col].rolling(20).apply(
                    lambda x: 1 if x.iloc[-1] < x.iloc[:-1].max() else 0, raw=False
                )
                df.loc[mask, f'{ticker}_bearish_divergence'] = (price_hh * rsi_lh).fillna(0).astype(int)
                
                # Price lower low but RSI higher low = bullish divergence
                price_ll = df.loc[mask, close_col].rolling(20).apply(
                    lambda x: 1 if x.iloc[-1] < x.iloc[:-1].min() else 0, raw=False
                )
                rsi_hl = df.loc[mask, rsi_col].rolling(20).apply(
                    lambda x: 1 if x.iloc[-1] > x.iloc[:-1].min() else 0, raw=False
                )
                df.loc[mask, f'{ticker}_bullish_divergence'] = (price_ll * rsi_hl).fillna(0).astype(int)
            
            # MACD Histogram slope (acceleration)
            if macd_hist_col in df.columns:
                df.loc[mask, f'{ticker}_macd_hist_slope'] = df.loc[mask, macd_hist_col].diff()
                df.loc[mask, f'{ticker}_macd_hist_acceleration'] = df.loc[mask, f'{ticker}_macd_hist_slope'].diff()
                df.loc[mask, f'{ticker}_macd_accelerating_up'] = (
                    (df.loc[mask, f'{ticker}_macd_hist_slope'] > 0) & 
                    (df.loc[mask, f'{ticker}_macd_hist_acceleration'] > 0)
                ).astype(int)
            
            # Momentum consistency (5-day RSI trend)
            if rsi_col in df.columns:
                rsi_5d_change = df.loc[mask, rsi_col].diff(5)
                df.loc[mask, f'{ticker}_rsi_momentum_5d'] = rsi_5d_change
                df.loc[mask, f'{ticker}_rsi_consistent_up'] = (
                    (df.loc[mask, rsi_col].diff(1) > 0) &
                    (df.loc[mask, rsi_col].diff(2) > 0) &
                    (df.loc[mask, rsi_col].diff(3) > 0)
                ).astype(int)
                df.loc[mask, f'{ticker}_rsi_consistent_down'] = (
                    (df.loc[mask, rsi_col].diff(1) < 0) &
                    (df.loc[mask, rsi_col].diff(2) < 0) &
                    (df.loc[mask, rsi_col].diff(3) < 0)
                ).astype(int)
            
            # ROC momentum quality
            if close_col in df.columns:
                roc_10 = df.loc[mask, close_col].pct_change(10) * 100
                roc_5 = df.loc[mask, close_col].pct_change(5) * 100
                df.loc[mask, f'{ticker}_roc_10'] = roc_10
                df.loc[mask, f'{ticker}_roc_5'] = roc_5
                # Momentum convergence (short-term and medium-term agree)
                df.loc[mask, f'{ticker}_momentum_convergence'] = (
                    ((roc_5 > 0) & (roc_10 > 0)) | ((roc_5 < 0) & (roc_10 < 0))
                ).astype(int)
        
        return df
    
    def add_advanced_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced volatility features.
        """
        logger.info("Adding advanced volatility features...")
        
        tickers = df['Ticker'].unique() if 'Ticker' in df.columns else ['default']
        
        for ticker in tickers:
            mask = df['Ticker'] == ticker if 'Ticker' in df.columns else slice(None)
            
            close_col = f'{ticker}_Close' if f'{ticker}_Close' in df.columns else 'Close'
            atr_col = f'{ticker}_ATR_14' if f'{ticker}_ATR_14' in df.columns else 'ATR_14'
            bb_upper = f'{ticker}_BBU_20_2.0' if f'{ticker}_BBU_20_2.0' in df.columns else 'BBU_20_2.0'
            bb_lower = f'{ticker}_BBL_20_2.0' if f'{ticker}_BBL_20_2.0' in df.columns else 'BBL_20_2.0'
            bb_mid = f'{ticker}_BBM_20_2.0' if f'{ticker}_BBM_20_2.0' in df.columns else 'BBM_20_2.0'
            
            # Bollinger Band squeeze detector
            if bb_upper in df.columns and bb_lower in df.columns:
                bb_width = df.loc[mask, bb_upper] - df.loc[mask, bb_lower]
                bb_width_percentile = bb_width.rolling(50).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
                )
                df.loc[mask, f'{ticker}_bb_width'] = bb_width
                df.loc[mask, f'{ticker}_bb_width_percentile'] = bb_width_percentile
                df.loc[mask, f'{ticker}_bb_squeeze'] = (bb_width_percentile < 20).astype(int)
                df.loc[mask, f'{ticker}_bb_expansion'] = (bb_width_percentile > 80).astype(int)
            
            # ATR expansion/contraction
            if atr_col in df.columns:
                atr_5 = df.loc[mask, atr_col].rolling(5).mean()
                atr_20 = df.loc[mask, atr_col].rolling(20).mean()
                df.loc[mask, f'{ticker}_atr_ratio_5_20'] = atr_5 / atr_20
                df.loc[mask, f'{ticker}_atr_expanding'] = (atr_5 > atr_20).astype(int)
                df.loc[mask, f'{ticker}_atr_contracting'] = (atr_5 < atr_20 * 0.8).astype(int)
            
            # Volatility-adjusted returns
            if close_col in df.columns and atr_col in df.columns:
                daily_return = df.loc[mask, close_col].pct_change()
                df.loc[mask, f'{ticker}_vol_adjusted_return'] = daily_return / (df.loc[mask, atr_col] / df.loc[mask, close_col])
            
            # Realized volatility vs implied (ATR proxy)
            if close_col in df.columns:
                realized_vol = df.loc[mask, close_col].pct_change().rolling(20).std() * np.sqrt(252) * 100
                df.loc[mask, f'{ticker}_realized_vol_20d'] = realized_vol
                if atr_col in df.columns:
                    implied_vol_proxy = (df.loc[mask, atr_col] / df.loc[mask, close_col]) * np.sqrt(252) * 100
                    df.loc[mask, f'{ticker}_vol_ratio'] = realized_vol / implied_vol_proxy
        
        return df
    
    def add_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-asset correlation and divergence features.
        """
        logger.info("Adding cross-asset features...")
        
        # Check for DXY (Dollar Index) columns
        dxy_close = None
        for col in df.columns:
            if 'DX-Y' in col and 'Close' in col:
                dxy_close = col
                break
        
        # Check for VIX columns
        vix_close = None
        for col in df.columns:
            if 'VIX' in col and 'Close' in col:
                vix_close = col
                break
        
        tickers = ['GBPUSD_X', 'EURUSD_X', 'GC_F']
        
        for ticker in tickers:
            close_col = f'{ticker}_Close'
            if close_col not in df.columns:
                continue
            
            # DXY momentum divergence
            if dxy_close is not None:
                pair_roc_5 = df[close_col].pct_change(5)
                dxy_roc_5 = df[dxy_close].pct_change(5)
                
                # For USD pairs, expect inverse relationship
                if 'USD' in ticker and ticker != 'GC_F':
                    df[f'{ticker}_dxy_divergence'] = (
                        ((pair_roc_5 > 0) & (dxy_roc_5 > 0)) |
                        ((pair_roc_5 < 0) & (dxy_roc_5 < 0))
                    ).astype(int)  # Unusual - same direction
                else:
                    df[f'{ticker}_dxy_divergence'] = (
                        ((pair_roc_5 > 0) & (dxy_roc_5 < 0)) |
                        ((pair_roc_5 < 0) & (dxy_roc_5 > 0))
                    ).astype(int)  # Unusual for gold
                
                # Rolling correlation with DXY
                df[f'{ticker}_dxy_corr_20d'] = df[close_col].rolling(20).corr(df[dxy_close])
            
            # VIX regime features
            if vix_close is not None:
                df[f'{ticker}_vix_high'] = (df[vix_close] > self.regime_config['vix_high']).astype(int)
                df[f'{ticker}_vix_low'] = (df[vix_close] < self.regime_config['vix_low']).astype(int)
                df[f'{ticker}_vix_spike'] = (df[vix_close].pct_change(5) > 0.2).astype(int)
                
                # Rolling correlation with VIX
                df[f'{ticker}_vix_corr_20d'] = df[close_col].rolling(20).corr(df[vix_close])
        
        # Cross-pair correlations
        if 'GBPUSD_X_Close' in df.columns and 'EURUSD_X_Close' in df.columns:
            df['GBPUSD_EURUSD_corr_20d'] = df['GBPUSD_X_Close'].rolling(20).corr(df['EURUSD_X_Close'])
            df['GBPUSD_EURUSD_spread'] = df['GBPUSD_X_Close'] / df['EURUSD_X_Close']
            df['GBPUSD_EURUSD_spread_zscore'] = (
                df['GBPUSD_EURUSD_spread'] - df['GBPUSD_EURUSD_spread'].rolling(50).mean()
            ) / df['GBPUSD_EURUSD_spread'].rolling(50).std()
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based seasonality features.
        """
        logger.info("Adding time-based features...")
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df = df.set_index('Date')
            else:
                logger.warning("No datetime index found, skipping time features")
                return df
        
        # Day of week (one-hot encoded)
        df['day_of_week'] = df.index.dayofweek
        for i in range(5):  # Monday=0 to Friday=4
            df[f'is_day_{i}'] = (df['day_of_week'] == i).astype(int)
        
        # Month (one-hot or cyclical)
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Quarter
        df['quarter'] = df.index.quarter
        
        # Week of year
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        
        # Year-end effect (December and January)
        df['is_year_end'] = df['month'].isin([12, 1]).astype(int)
        
        # Month-end effect (last 5 trading days)
        df['day_of_month'] = df.index.day
        df['is_month_end'] = (df.index.day > 25).astype(int)
        df['is_month_start'] = (df.index.day <= 5).astype(int)
        
        # Days since last major move (>2 ATR)
        for ticker in ['GBPUSD_X', 'EURUSD_X', 'GC_F']:
            close_col = f'{ticker}_Close'
            atr_col = f'{ticker}_ATR_14'
            
            if close_col in df.columns and atr_col in df.columns:
                daily_move = df[close_col].diff().abs()
                major_move = daily_move > (2 * df[atr_col])
                
                # Calculate days since last major move
                df[f'{ticker}_days_since_major_move'] = (~major_move).cumsum() - (~major_move).cumsum().where(major_move).ffill()
                df[f'{ticker}_days_since_major_move'] = df[f'{ticker}_days_since_major_move'].fillna(0)
        
        return df
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern recognition features.
        """
        logger.info("Adding pattern recognition features...")
        
        for ticker in ['GBPUSD_X', 'EURUSD_X', 'GC_F']:
            open_col = f'{ticker}_Open'
            high_col = f'{ticker}_High'
            low_col = f'{ticker}_Low'
            close_col = f'{ticker}_Close'
            
            if not all(col in df.columns for col in [open_col, high_col, low_col, close_col]):
                continue
            
            o = df[open_col]
            h = df[high_col]
            l = df[low_col]
            c = df[close_col]
            
            # Candle body and wick sizes
            body = abs(c - o)
            upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
            lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
            candle_range = h - l
            
            # Doji (small body relative to range)
            df[f'{ticker}_is_doji'] = (body < candle_range * 0.1).astype(int)
            
            # Hammer/Hanging Man (small body at top, long lower wick)
            df[f'{ticker}_is_hammer'] = (
                (lower_wick > body * 2) &
                (upper_wick < body * 0.5) &
                (c > o)  # Bullish
            ).astype(int)
            
            # Shooting Star (small body at bottom, long upper wick)
            df[f'{ticker}_is_shooting_star'] = (
                (upper_wick > body * 2) &
                (lower_wick < body * 0.5) &
                (c < o)  # Bearish
            ).astype(int)
            
            # Engulfing patterns (2-day)
            prev_body = body.shift(1)
            prev_bullish = (c.shift(1) > o.shift(1))
            prev_bearish = (c.shift(1) < o.shift(1))
            
            df[f'{ticker}_bullish_engulfing'] = (
                prev_bearish &
                (c > o) &
                (body > prev_body) &
                (c > o.shift(1)) &
                (o < c.shift(1))
            ).astype(int)
            
            df[f'{ticker}_bearish_engulfing'] = (
                prev_bullish &
                (c < o) &
                (body > prev_body) &
                (c < o.shift(1)) &
                (o > c.shift(1))
            ).astype(int)
            
            # Inside bar (today's range within yesterday's range)
            df[f'{ticker}_inside_bar'] = (
                (h < h.shift(1)) &
                (l > l.shift(1))
            ).astype(int)
            
            # Outside bar (today's range exceeds yesterday's range)
            df[f'{ticker}_outside_bar'] = (
                (h > h.shift(1)) &
                (l < l.shift(1))
            ).astype(int)
            
            # Consecutive up/down days
            up_day = (c > c.shift(1)).astype(int)
            down_day = (c < c.shift(1)).astype(int)
            
            df[f'{ticker}_consecutive_up'] = up_day.groupby((up_day != up_day.shift()).cumsum()).cumsum()
            df[f'{ticker}_consecutive_down'] = down_day.groupby((down_day != down_day.shift()).cumsum()).cumsum()
        
        return df
    
    def generate_v2_features(self, features_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load existing features and add V2 features.
        
        Args:
            features_path: Path to existing features.parquet
            output_path: Optional path to save enhanced features
            
        Returns:
            DataFrame with all V2 features added
        """
        logger.info(f"Loading features from {features_path}")
        df = pd.read_parquet(features_path)
        
        logger.info(f"Original shape: {df.shape}")
        
        # Add all advanced features
        df = self.add_all_advanced_features(df)
        
        logger.info(f"Enhanced shape: {df.shape}")
        
        # Drop any rows with NaN in critical columns
        initial_len = len(df)
        df = df.dropna(subset=[c for c in df.columns if 'target' in c.lower()], how='all')
        logger.info(f"Dropped {initial_len - len(df)} rows with missing targets")
        
        if output_path:
            df.to_parquet(output_path)
            logger.info(f"Saved enhanced features to {output_path}")
        
        return df


def main():
    """Main function to run feature engineering V2."""
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize engine
    engine = AdvancedFeatureEngine(config)
    
    # Generate V2 features
    features_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features.parquet'
    output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features_v2.parquet'
    
    df = engine.generate_v2_features(str(features_path), str(output_path))
    
    print(f"\n✅ Feature Engineering V2 Complete!")
    print(f"   Shape: {df.shape}")
    print(f"   New columns: {len([c for c in df.columns if 'v2' in c.lower() or 'regime' in c.lower() or 'divergence' in c.lower()])}")


if __name__ == "__main__":
    main()
