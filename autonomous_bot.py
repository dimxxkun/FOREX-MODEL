import time
import logging
import json
import yfinance as yf
from datetime import datetime
from pathlib import Path
from signal_generator import SignalGenerator
from paper_trading_tracker import PaperTradingTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/autonomous_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('autonomous_bot')

class AutonomousTradingBot:
    def __init__(self, interval_minutes=5, balance=115.0):
        self.interval = interval_minutes * 60
        self.generator = SignalGenerator()
        self.tracker = PaperTradingTracker(data_dir='trades')
        self.balance = balance
        self.is_running = False
        
        # Ticker mapping for yfinance
        self.yf_map = {
            'GBPUSD': 'GBPUSD=X',
            'EURUSD': 'EURUSD=X',
            'GC_F': 'GC=F'
        }
        
    def fetch_latest_prices(self):
        """Fetch real-time prices for all monitored assets."""
        prices = {}
        tickers = list(self.yf_map.values())
        try:
            # Batch fetch for efficiency
            data = yf.download(tickers, period='1d', interval='1m', progress=False)
            if not data.empty:
                import pandas as pd
                for yf_ticker in tickers:
                    if isinstance(data.columns, pd.MultiIndex):
                        prices[yf_ticker] = float(data['Close'][yf_ticker].iloc[-1])
                    else:
                        prices[yf_ticker] = float(data['Close'].iloc[-1])
            
            # 2. Map back and handle missing/nan with single-ticker fallback
            mapped_prices = {}
            for key, val in self.yf_map.items():
                p = prices.get(val)
                if p is None or (isinstance(p, float) and pd.isna(p)):
                    # Try individual fetch as fallback
                    try:
                        single_data = yf.download(val, period='1d', interval='1m', progress=False)
                        if not single_data.empty:
                            p = float(single_data['Close'].iloc[-1])
                    except:
                        pass
                
                if p is not None and not (isinstance(p, float) and pd.isna(p)):
                    mapped_prices[key] = p
            return mapped_prices
        except Exception as e:
            logger.error(f"Error fetching prices: {e}")
            return {}

    def run_cycle(self):
        """Perform one iteration of the trading loop."""
        logger.info("-" * 40)
        logger.info(f"SCANNING MARKET @ {datetime.now().strftime('%H:%M:%S')}")
        
        # 1. Update existing positions
        latest_prices = self.fetch_latest_prices()
        if latest_prices:
            closed_trades = self.tracker.update_positions(latest_prices)
            for trade in closed_trades:
                logger.info(f"POSITION CLOSED: {trade['ticker']} | Reason: {trade['close_reason']} | PnL: ${trade['pnl']}")
        
        # 2. Check for new signals
        # We use a 1-day lookback for the feature pipeline
        df = self.generator.get_latest_data(lookback_days=50) 
        
        for ticker in ['GBPUSD', 'EURUSD', 'GC_F']:
            # Skip if we already have an open position
            if ticker in self.tracker.open_positions:
                continue
                
            signal = self.generator.generate_signal(ticker, df, account_value=self.balance)
            
            if signal['signal'] in ['BUY', 'SELL']:
                logger.info(f"NEW SIGNAL: {ticker} {signal['signal']} @ {signal['entry_price']} (Confidence: {signal['confidence']:.1f}%)")
                self.tracker.record_signal(signal)
        
        # 3. Print Dashboard
        self.display_dashboard()

    def display_dashboard(self):
        """Print a summary of the current bot state."""
        summary = self.tracker.get_performance_summary()
        open_pos = self.tracker.get_open_positions_summary()
        
        print("\n" + "=" * 50)
        print(f"AUTONOMOUS BOT DASHBOARD | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        print(f"💰 Realized PnL: ${summary['total_pnl']:>8.2f}")
        print(f"📊 Win Rate:    {summary['win_rate']:>7.1f}%")
        print(f"📍 Open Trades: {len(open_pos):>8}")
        
        if open_pos:
            print("-" * 50)
            print(f"{'Ticker':<8} | {'Side':<4} | {'Entry':<8} | {'PnL (%)'}")
            print("-" * 50)
            for pos in open_pos:
                print(f"{pos['ticker']:<8} | {pos['signal']:<4} | {pos['entry_price']:<8.4f} | ---")
        
        print("=" * 50 + "\n")

    def start(self):
        """Start the continuous loop."""
        self.is_running = True
        logger.info(f"Bot started. Interval: {self.interval/60} minutes.")
        
        try:
            while self.is_running:
                self.run_cycle()
                logger.info(f"Waiting {self.interval/60} minutes for next cycle...")
                time.sleep(self.interval)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Critical error in loop: {e}")
            self.stop()

    def stop(self):
        """Stop the bot."""
        self.is_running = False
        logger.info("Bot stopping...")

if __name__ == "__main__":
    # Start with $100 demo balance
    bot = AutonomousTradingBot(interval_minutes=5, balance=115.0)
    bot.start()
