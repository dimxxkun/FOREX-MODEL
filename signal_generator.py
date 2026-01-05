"""
Daily Signal Generator for Forex Trading Model.

Generates end-of-day trading signals with:
- Direction (BUY/SELL/HOLD)
- Confidence score
- Entry, Stop Loss, Take Profit levels
- Position size (volatility-scaled)

Usage:
    python signal_generator.py                    # Generate today's signals
    python signal_generator.py --date 2026-01-05  # Generate for specific date
    python signal_generator.py --schedule         # Run on schedule (5 PM EST)
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import DataPipeline
from src.feature_engineering import FeatureEngine
from src.regime_detector import MarketRegimeDetector
from src.signal_filter import SignalFilter
from src.risk_management import RiskManager
from src.utils import get_logger, load_config

# Configure logging
logger = get_logger('signal_generator')


class SignalGenerator:
    """
    Generate daily trading signals for forex and gold.
    
    Integrates:
    - Data loading (latest market data)
    - Feature generation
    - Model prediction
    - Regime detection
    - Signal filtering
    - Position sizing
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the signal generator.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = load_config(config_path)
        self.logger = get_logger('signal_generator')
        
        # Initialize components
        self.data_pipeline = DataPipeline(config_path)
        self.feature_engine = FeatureEngine(config_path)
        self.regime_detector = MarketRegimeDetector(self.config)
        self.signal_filter = SignalFilter(self.config)
        self.risk_manager = RiskManager(config_path)
        
        # Load trained model
        self.model = self._load_model()
        
        # Tickers
        self.tickers = [
            t.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
            for t in self.config['data']['tickers']['main']
        ]
        
        # Output directory
        self.output_dir = PROJECT_ROOT / 'signals'
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info("SignalGenerator initialized")
    
    def _load_model(self) -> object:
        """Load trained model or train a simple one on the fly."""
        model_paths = [
            PROJECT_ROOT / 'results' / 'walk_forward' / 'best_wf_model.pkl',
            PROJECT_ROOT / 'models' / 'xgboost_GBPUSD_X.pkl',
            PROJECT_ROOT / 'models' / 'optimized' / 'xgboost_optimized_GBPUSD.pkl',
            PROJECT_ROOT / 'notebooks' / 'models' / 'xgboost_GBPUSD_X.pkl',
        ]
        
        # Try loading existing models
        for path in model_paths:
            if path.exists():
                try:
                    self.logger.info(f"Trying to load model from {path}")
                    import pickle
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                    if hasattr(model, 'predict'):
                        return model
                except Exception as e:
                    self.logger.warning(f"Failed to load {path}: {e}")
                    continue
        
        # Fallback: Train a simple model on the fly
        self.logger.warning("No loadable model found. Training quick XGBoost model...")
        return self._train_quick_model()
    
    def _train_quick_model(self):
        """Train a quick XGBoost model for signal generation."""
        from xgboost import XGBClassifier
        
        # Load features
        features_path = PROJECT_ROOT / 'data' / 'processed' / 'features.parquet'
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        
        df = pd.read_parquet(features_path)
        
        # Prepare data
        target_col = 'GBPUSD_Target_Direction'
        feature_cols = [c for c in df.columns if 'Target' not in c and c not in ['Date']]
        
        df_clean = df.dropna(subset=[target_col])
        X = df_clean[feature_cols].fillna(0)
        y = df_clean[target_col]
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        
        # Train model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        
        self.logger.info(f"Trained quick model with {len(X_train)} samples")
        return model
    
    def get_latest_data(self, lookback_days: int = 300) -> pd.DataFrame:
        """
        Get latest market data for signal generation.
        
        Args:
            lookback_days: Number of historical days to load.
            
        Returns:
            DataFrame with latest market data.
        """
        self.logger.info(f"Loading latest data (lookback: {lookback_days} days)")
        
        # Try to load from processed data first
        features_path = PROJECT_ROOT / 'data' / 'processed' / 'features.parquet'
        
        if features_path.exists():
            df = pd.read_parquet(features_path)
            # Get most recent data
            df = df.tail(lookback_days)
            self.logger.info(f"Loaded {len(df)} rows from {features_path}")
            return df
        
        # Otherwise, run pipeline to get fresh data
        self.logger.info("No cached data, running data pipeline...")
        combined = self.data_pipeline.run_full_pipeline()
        df = self.feature_engine.run_full_pipeline(combined)
        return df.tail(lookback_days)
    
    def generate_signal(
        self,
        ticker: str,
        df: pd.DataFrame,
        account_value: float = 10000.0
    ) -> Dict:
        """
        Generate trading signal for a single ticker.
        
        Args:
            ticker: Ticker symbol.
            df: DataFrame with features.
            account_value: Current account value for position sizing.
            
        Returns:
            Signal dictionary with all trade parameters.
        """
        self.logger.info(f"Generating signal for {ticker}")
        
        # Get latest row
        latest = df.iloc[-1]
        latest_date = df.index[-1]
        
        # Get feature columns (exclude targets and metadata)
        feature_cols = [c for c in df.columns if 'Target' not in c and c not in ['Date', 'ticker']]
        
        # Prepare features for prediction
        X = df[feature_cols].iloc[[-1]].fillna(0)
        
        # Align features with model
        try:
            model_features = self.model.get_booster().feature_names
            # Add missing features
            for feat in model_features:
                if feat not in X.columns:
                    X[feat] = 0
            X = X[model_features]
        except:
            pass  # Not all models have feature_names
        
        # Get prediction
        try:
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0, 1]
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return self._hold_signal(ticker, latest_date, "prediction_error")
        
        # Calculate confidence (0-100 scale)
        confidence = abs(probability - 0.5) * 2 * 100
        
        # Get current price and ATR
        close_col = f'{ticker}_Close'
        atr_col = f'{ticker}_ATR'
        
        current_price = latest.get(close_col, 0)
        current_atr = latest.get(atr_col, current_price * 0.01)
        
        if current_price == 0:
            return self._hold_signal(ticker, latest_date, "no_price_data")
        
        # Detect regime
        regime = self.regime_detector.detect_regime(df, ticker, -1)
        regime_str = regime.value
        
        # Filter signal
        filtered_signal, adj_confidence, reason = self.signal_filter.filter_signal(
            signal=prediction,
            confidence=confidence,
            regime=regime_str,
            current_date=latest_date
        )
        
        # If filtered out, return hold
        if filtered_signal == -1:
            return self._hold_signal(ticker, latest_date, reason)
        
        # Calculate position levels
        direction = 1 if prediction == 1 else -1
        stop_loss = self.risk_manager.get_stop_loss_price(
            current_price, current_atr, direction
        )
        take_profit = self.risk_manager.get_take_profit_price(
            current_price, stop_loss, direction, risk_reward_ratio=1.5
        )
        
        # Calculate position size with volatility scaling
        avg_atr = df[atr_col].tail(20).mean() if atr_col in df.columns else current_atr
        position_size = self.risk_manager.calculate_volatility_scaled_position(
            entry_price=current_price,
            stop_loss_price=stop_loss,
            account_value=account_value,
            current_atr=current_atr,
            avg_atr=avg_atr
        )
        
        # Build signal
        signal = {
            'ticker': ticker,
            'date': str(latest_date.date()),
            'timestamp': datetime.now().isoformat(),
            'signal': 'BUY' if prediction == 1 else 'SELL',
            'direction': direction,
            'confidence': round(confidence, 1),
            'regime': regime_str,
            'entry_price': round(current_price, 5),
            'stop_loss': round(stop_loss, 5),
            'take_profit': round(take_profit, 5),
            'position_size': round(position_size, 2),
            'risk_pips': round(abs(current_price - stop_loss) * 10000, 1),
            'reward_pips': round(abs(take_profit - current_price) * 10000, 1),
            'atr': round(current_atr, 5),
            'probability': round(probability, 4),
            'filter_status': 'passed'
        }
        
        return signal
    
    def _hold_signal(self, ticker: str, date, reason: str) -> Dict:
        """Generate a HOLD signal."""
        return {
            'ticker': ticker,
            'date': str(date.date()) if hasattr(date, 'date') else str(date),
            'timestamp': datetime.now().isoformat(),
            'signal': 'HOLD',
            'direction': 0,
            'confidence': 0,
            'regime': 'unknown',
            'entry_price': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'position_size': 0,
            'filter_status': reason
        }
    
    def generate_all_signals(
        self,
        account_value: float = 10000.0
    ) -> List[Dict]:
        """
        Generate signals for all tickers.
        
        Args:
            account_value: Current account value.
            
        Returns:
            List of signal dictionaries.
        """
        self.logger.info("Generating signals for all tickers...")
        
        # Load data
        df = self.get_latest_data()
        
        signals = []
        for ticker in self.tickers:
            try:
                signal = self.generate_signal(ticker, df, account_value)
                signals.append(signal)
                
                # Log signal
                self.logger.info(
                    f"{ticker}: {signal['signal']} @ {signal['entry_price']} "
                    f"(conf: {signal['confidence']}%, regime: {signal['regime']})"
                )
            except Exception as e:
                self.logger.error(f"Error generating signal for {ticker}: {e}")
                signals.append(self._hold_signal(ticker, datetime.now(), str(e)))
        
        return signals
    
    def save_signals(self, signals: List[Dict], filename: Optional[str] = None) -> Path:
        """
        Save signals to JSON file.
        
        Args:
            signals: List of signal dictionaries.
            filename: Custom filename (optional).
            
        Returns:
            Path to saved file.
        """
        if filename is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
            filename = f'signals_{date_str}.json'
        
        filepath = self.output_dir / filename
        
        output = {
            'generated_at': datetime.now().isoformat(),
            'account_value': 10000,
            'signals': signals
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"Signals saved to {filepath}")
        return filepath
    
    def run(self, account_value: float = 10000.0) -> Tuple[List[Dict], Path]:
        """
        Run the full signal generation pipeline.
        
        Args:
            account_value: Current account value.
            
        Returns:
            Tuple of (signals, output_path).
        """
        signals = self.generate_all_signals(account_value)
        output_path = self.save_signals(signals)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 DAILY TRADING SIGNALS")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        
        for signal in signals:
            icon = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "⚪"
            print(f"\n{icon} {signal['ticker']}: {signal['signal']}")
            
            if signal['signal'] != 'HOLD':
                print(f"   Entry:      {signal['entry_price']}")
                print(f"   Stop Loss:  {signal['stop_loss']}")
                print(f"   Take Profit: {signal['take_profit']}")
                print(f"   Size:       {signal['position_size']} units")
                print(f"   Confidence: {signal['confidence']}%")
                print(f"   Regime:     {signal['regime']}")
            else:
                print(f"   Reason: {signal['filter_status']}")
        
        print("\n" + "=" * 60)
        print(f"📁 Saved to: {output_path}")
        print("=" * 60)
        
        return signals, output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate daily trading signals')
    parser.add_argument('--account', type=float, default=10000,
                       help='Account value for position sizing')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Generate signals
    generator = SignalGenerator(args.config)
    signals, output_path = generator.run(args.account)
    
    return signals


if __name__ == '__main__':
    main()
