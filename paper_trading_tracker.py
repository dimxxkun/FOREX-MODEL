"""
Paper Trading Tracker for Forex Signal Model.

Tracks signals, open positions, and performance metrics.

Usage:
    from paper_trading_tracker import PaperTradingTracker
    tracker = PaperTradingTracker()
    tracker.record_signal(signal)
    tracker.update_position(ticker, current_price)
    tracker.get_performance_summary()
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PaperTradingTracker:
    """
    Track paper trading signals and performance.
    
    Features:
    - Record all signals generated
    - Track open positions
    - Log trade outcomes
    - Calculate running performance metrics
    """
    
    def __init__(self, data_dir: str = 'paper_trading'):
        """
        Initialize the tracker.
        
        Args:
            data_dir: Directory for storing trade data.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.signals_file = self.data_dir / 'signals_history.json'
        self.positions_file = self.data_dir / 'open_positions.json'
        self.trades_file = self.data_dir / 'closed_trades.json'
        
        # Load existing data
        self.signals_history = self._load_json(self.signals_file, [])
        self.open_positions = self._load_json(self.positions_file, {})
        self.closed_trades = self._load_json(self.trades_file, [])
        
        logger.info("PaperTradingTracker initialized")
    
    def _load_json(self, path: Path, default):
        """Load JSON file or return default."""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return default
    
    def _save_json(self, path: Path, data):
        """Save data to JSON file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def record_signal(self, signal: Dict) -> None:
        """
        Record a generated signal.
        
        Args:
            signal: Signal dictionary from SignalGenerator.
        """
        signal['recorded_at'] = datetime.now().isoformat()
        self.signals_history.append(signal)
        self._save_json(self.signals_file, self.signals_history)
        
        # Open position if actionable signal
        if signal['signal'] in ['BUY', 'SELL']:
            self.open_position(signal)
        
        logger.info(f"Recorded signal: {signal['ticker']} {signal['signal']}")
    
    def open_position(self, signal: Dict) -> None:
        """
        Open a new position.
        
        Args:
            signal: Signal dictionary.
        """
        ticker = signal['ticker']
        
        # Check if position already open
        if ticker in self.open_positions:
            logger.warning(f"Position already open for {ticker}")
            return
        
        position = {
            'ticker': ticker,
            'direction': signal['direction'],
            'signal': signal['signal'],
            'entry_price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['take_profit'],
            'position_size': signal['position_size'],
            'entry_date': signal['date'],
            'entry_time': datetime.now().isoformat(),
            'confidence': signal['confidence'],
            'regime': signal['regime']
        }
        
        self.open_positions[ticker] = position
        self._save_json(self.positions_file, self.open_positions)
        
        logger.info(f"Opened position: {ticker} {signal['signal']} @ {signal['entry_price']}")
    
    def close_position(
        self,
        ticker: str,
        exit_price: float,
        reason: str = 'manual'
    ) -> Optional[Dict]:
        """
        Close an open position.
        
        Args:
            ticker: Ticker symbol.
            exit_price: Price at exit.
            reason: Close reason (stop_loss, take_profit, manual).
            
        Returns:
            Closed trade dictionary or None.
        """
        if ticker not in self.open_positions:
            logger.warning(f"No open position for {ticker}")
            return None
        
        position = self.open_positions[ticker]
        
        # Calculate P&L
        if position['direction'] == 1:  # Long
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:  # Short
            pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        pnl_pct = (pnl / (position['entry_price'] * position['position_size'])) * 100
        
        # Create trade record
        trade = {
            **position,
            'exit_price': exit_price,
            'exit_date': datetime.now().strftime('%Y-%m-%d'),
            'exit_time': datetime.now().isoformat(),
            'close_reason': reason,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'won': pnl > 0
        }
        
        # Update records
        self.closed_trades.append(trade)
        del self.open_positions[ticker]
        
        self._save_json(self.trades_file, self.closed_trades)
        self._save_json(self.positions_file, self.open_positions)
        
        status = "WIN ✅" if pnl > 0 else "LOSS ❌"
        logger.info(f"Closed {ticker}: {status} ${pnl:.2f} ({pnl_pct:.1f}%)")
        
        return trade
    
    def update_positions(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Update positions with current prices, check stops/targets.
        
        Args:
            prices: Dictionary of ticker -> current price.
            
        Returns:
            List of closed trades.
        """
        closed = []
        
        for ticker, position in list(self.open_positions.items()):
            if ticker not in prices:
                continue
            
            current_price = prices[ticker]
            
            # Check stop loss
            if position['direction'] == 1:  # Long
                if current_price <= position['stop_loss']:
                    trade = self.close_position(ticker, current_price, 'stop_loss')
                    closed.append(trade)
                elif current_price >= position['take_profit']:
                    trade = self.close_position(ticker, current_price, 'take_profit')
                    closed.append(trade)
            else:  # Short
                if current_price >= position['stop_loss']:
                    trade = self.close_position(ticker, current_price, 'stop_loss')
                    closed.append(trade)
                elif current_price <= position['take_profit']:
                    trade = self.close_position(ticker, current_price, 'take_profit')
                    closed.append(trade)
        
        return closed
    
    def get_performance_summary(self) -> Dict:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics.
        """
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'sharpe': 0,
                'profit_factor': 0
            }
        
        trades_df = pd.DataFrame(self.closed_trades)
        
        # Calculate metrics
        total_trades = len(trades_df)
        wins = trades_df['won'].sum()
        win_rate = wins / total_trades * 100
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Sharpe (simplified)
        returns = trades_df['pnl_pct']
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'wins': int(wins),
            'losses': total_trades - int(wins),
            'win_rate': round(win_rate, 1),
            'total_pnl': round(total_pnl, 2),
            'avg_pnl': round(avg_pnl, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe': round(sharpe, 2),
            'best_trade': round(trades_df['pnl'].max(), 2),
            'worst_trade': round(trades_df['pnl'].min(), 2)
        }
    
    def get_open_positions_summary(self) -> List[Dict]:
        """Get summary of open positions."""
        return list(self.open_positions.values())
    
    def print_summary(self) -> None:
        """Print performance summary."""
        summary = self.get_performance_summary()
        open_pos = self.get_open_positions_summary()
        
        print("\n" + "=" * 60)
        print("📊 PAPER TRADING SUMMARY")
        print("=" * 60)
        
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Trades: {summary['total_trades']}")
        print(f"   Win Rate: {summary['win_rate']}%")
        print(f"   Total P&L: ${summary['total_pnl']}")
        print(f"   Profit Factor: {summary['profit_factor']}")
        print(f"   Sharpe Ratio: {summary['sharpe']}")
        
        if open_pos:
            print(f"\n📍 Open Positions ({len(open_pos)}):")
            for pos in open_pos:
                print(f"   {pos['ticker']}: {pos['signal']} @ {pos['entry_price']}")
        else:
            print(f"\n📍 No open positions")
        
        print("=" * 60)


def main():
    """Test the tracker."""
    tracker = PaperTradingTracker()
    
    # Example: Record a signal
    test_signal = {
        'ticker': 'GBPUSD',
        'date': '2026-01-05',
        'signal': 'BUY',
        'direction': 1,
        'confidence': 65.0,
        'regime': 'weak_trend',
        'entry_price': 1.2500,
        'stop_loss': 1.2400,
        'take_profit': 1.2650,
        'position_size': 1000
    }
    
    tracker.record_signal(test_signal)
    tracker.print_summary()


if __name__ == '__main__':
    main()
