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
import yfinance as yf
import xgboost as xgb

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
        
        # Load trained models (dictionary of ticker -> model)
        self.models = self._load_all_models()
        
        # Tickers
        self.tickers = [
            t.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_').replace('=', '_')
            for t in self.config['data']['tickers']['main']
        ]
        
        # Output directory
        self.output_dir = PROJECT_ROOT / 'signals'
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger.info("SignalGenerator initialized")
    
    def _load_all_models(self) -> Dict[str, object]:
        """Load trained models for all tickers."""
        models = {}
        for ticker in self.config['data']['tickers']['main']:
            ticker_clean = ticker.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_').replace('=', '_')
            model = self._load_ticker_model(ticker, ticker_clean)
            models[ticker_clean] = model
        return models

    def _load_ticker_model(self, ticker_raw: str, ticker_clean: str) -> object:
        """Load trained model for a specific ticker."""
        model_paths = [
            PROJECT_ROOT / 'models' / f'xgboost_{ticker_clean}.pkl',
            PROJECT_ROOT / 'models' / 'ensemble' / f'xgboost_{ticker_clean}.pkl',
            PROJECT_ROOT / 'models' / 'optimized' / f'xgboost_optimized_{ticker_clean.replace("_X", "")}.pkl',
            PROJECT_ROOT / 'results' / 'walk_forward' / 'best_wf_model.pkl',
            PROJECT_ROOT / 'notebooks' / 'models' / f'xgboost_{ticker_clean}.pkl',
        ]
        
        # Try loading existing models
        for path in model_paths:
            if path.exists():
                try:
                    self.logger.info(f"Trying to load model for {ticker_clean} from {path}")
                    import pickle
                    with open(path, 'rb') as f:
                        model = pickle.load(f)
                    if isinstance(model, dict) and 'model' in model:
                        actual_model = model['model']
                        # Store feature names if available in dict
                        if 'feature_names' in model:
                            actual_model.feature_names_from_dict = model['feature_names']
                        return actual_model
                    if hasattr(model, 'predict'):
                        return model
                except Exception as e:
                    self.logger.warning(f"Failed to load {path}: {e}")
                    continue
        
        # Fallback: Train a simple model on the fly
        self.logger.warning(f"No loadable model found for {ticker_clean}. Training quick XGBoost model...")
        return self._train_quick_model(ticker_raw, ticker_clean)
    
    def _train_quick_model(self, ticker_raw: str, ticker_clean: str):
        """Train a quick XGBoost model for signal generation."""
        from xgboost import XGBClassifier
        
        # Load features
        today_override = self.config['data'].get('today_override')
        features_path = PROJECT_ROOT / 'data' / 'processed' / 'features.parquet'
        if not features_path.exists():
            # If features don't exist, we might need to run the pipeline
            self.logger.info("Features file not found, running pipeline for training data...")
            combined = self.data_pipeline.run_full_pipeline()
            df = self.feature_engine.run_full_pipeline(combined)
        else:
            df = pd.read_parquet(features_path)
            if today_override:
                self.logger.info(f"Applying Date Override: {today_override}")
                df = df[df.index <= pd.to_datetime(today_override)]
        
        # Prepare data
        target_col = f'{ticker_clean}_Target_Direction'
        if target_col not in df.columns:
            # Fallback to GBPUSD if specific ticker target not found (for safety)
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
        
        self.logger.info(f"Trained quick model for {ticker_clean} with {len(X_train)} samples")
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
            today_override = self.config['data'].get('today_override')
            if today_override:
                df = df[df.index <= pd.to_datetime(today_override)]
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
        
        # Prepare features for prediction
        X = df[feature_cols].iloc[[-1]].fillna(0)
        
        # Get ticker-specific model
        model = self.models.get(ticker)
        if model is None:
            self.logger.error(f"No model found for {ticker}")
            return self._hold_signal(ticker, latest_date, "no_model")

        # Align features with model
        try:
            # Check if we stored feature names in our custom attribute or booster
            if hasattr(model, 'feature_names_from_dict'):
                model_features = model.feature_names_from_dict
            else:
                model_features = model.get_booster().feature_names
            
            # Create a new DataFrame with ONLY the columns the model expects, in the correct order
            X_aligned = pd.DataFrame(index=X.index)
            for feat in model_features:
                if feat in X.columns:
                    X_aligned[feat] = X[feat]
                else:
                    # Feature expected by model but missing in current data
                    X_aligned[feat] = 0.0
            
            X = X_aligned
            self.logger.debug(f"Features aligned for {ticker} using {len(model_features)} features")
        except Exception as e:
            self.logger.warning(f"Could not align features for {ticker}: {e}")
            pass

        # Get prediction
        try:
            # Use native booster to bypass scikit-learn wrapper strictness on feature names
            dmatrix = xgb.DMatrix(X)
            # prediction results from booster.predict is probabilities for binary classification
            output = model.get_booster().predict(dmatrix)[0]
            
            # Handle multi-class or multi-output if necessary, though XGBClassifier.predict usually gives 0/1
            # but booster.predict gives probabilities [prob_0, prob_1] or just prob_1
            if isinstance(output, (list, np.ndarray)) and len(output) > 1:
                probability = float(output[1])
            else:
                probability = float(output)
                
            prediction = 1 if probability > 0.5 else 0
            self.logger.info(f"Generated prediction for {ticker}: {prediction} (prob: {probability:.4f})")
        except Exception as e:
            self.logger.error(f"Prediction error for {ticker}: {e}")
            return self._hold_signal(ticker, latest_date, f"prediction_error: {str(e)}")
        
        # Calculate confidence (0-100 scale)
        confidence = max(probability, 1 - probability) * 100
        
        # Get REAL-TIME price from yfinance (Confirmed accurate for 2026)
        try:
            # Map ticker to yfinance format if needed
            yf_ticker = ticker
            if ticker == 'GOLD' or ticker == 'GC_F': yf_ticker = 'GC=F'
            elif 'GBPUSD' in ticker: yf_ticker = 'GBPUSD=X'
            elif 'EURUSD' in ticker: yf_ticker = 'EURUSD=X'
            
            self.logger.info(f"Fetching live price for {yf_ticker} from yfinance...")
            live_data = yf.download(yf_ticker, period='1d', interval='1m', progress=False)
            
            if not live_data.empty:
                # Handle multi-index columns if they exist
                if isinstance(live_data.columns, pd.MultiIndex):
                    current_price = float(live_data['Close'][yf_ticker].iloc[-1])
                else:
                    current_price = float(live_data['Close'].iloc[-1])
                self.logger.info(f"✅ yfinance Live Price for {ticker}: {current_price}")
            else:
                current_price = float(latest.get(f'{ticker}_Close', 0))
                self.logger.warning(f"yfinance returned empty, using stale price: {current_price}")
        except Exception as e:
            current_price = float(latest.get(f'{ticker}_Close', 0))
            self.logger.error(f"❌ Error fetching yfinance price for {ticker}: {e}. Using STALE price ({current_price})")
        
        # Get ATR from features
        atr_col = f'{ticker}_ATR'
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
            ticker=ticker,
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
        
        # Calculate pips correctly for Gold vs Forex
        pip_multiplier = 10000
        if current_price > 100:  # Gold/Commodity
            pip_multiplier = 10   # $1.00 move = 10 "pips" (0.10 increments)
            
        # Build signal
        signal = {
            'ticker': ticker,
            'date': str(latest_date.date()),
            'timestamp': datetime.now().isoformat(),
            'signal': 'BUY' if prediction == 1 else 'SELL',
            'direction': int(direction),
            'confidence': float(round(confidence, 1)),
            'regime': regime_str,
            'entry_price': float(round(current_price, 5)),
            'stop_loss': float(round(stop_loss, 5)),
            'take_profit': float(round(take_profit, 5)),
            'position_size': float(round(position_size, 2)),
            'risk_pips': float(round(abs(current_price - stop_loss) * pip_multiplier, 1)),
            'reward_pips': float(round(abs(take_profit - current_price) * pip_multiplier, 1)),
            'atr': float(round(current_atr, 5)),
            'probability': float(round(probability, 4)),
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
    parser.add_argument('--notify', action='store_true',
                       help='Send Telegram notification for signals')
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
    
    # Send Telegram notification if enabled
    if args.notify:
        try:
            from telegram_notifier import TelegramNotifier
            notifier = TelegramNotifier()
            
            # Send individual signals for BUY/SELL
            actionable_signals = [s for s in signals if s['signal'] != 'HOLD']
            
            if actionable_signals:
                for signal in actionable_signals:
                    notifier.send_signal(signal)
                print("📱 Telegram notifications sent!")
            else:
                # Send summary that no trades today
                notifier.send_daily_summary(signals)
                print("📱 Daily summary sent to Telegram")
        except Exception as e:
            print(f"⚠️ Telegram notification failed: {e}")
            print("   Run: python telegram_notifier.py --setup")
    
    return signals


if __name__ == '__main__':
    main()
