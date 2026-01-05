"""
Backtesting Engine for Forex Signal Model.

Realistic backtesting with transaction costs and risk management.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, timer_decorator


class Trade:
    """Represents a single trade."""
    
    def __init__(
        self,
        ticker: str,
        entry_date: datetime,
        entry_price: float,
        direction: int,
        position_size: float,
        stop_loss: float,
        confidence: float
    ):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.direction = direction  # 1=LONG, -1=SHORT
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.confidence = confidence
        
        self.exit_date: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.exit_reason: Optional[str] = None
        self.pnl: float = 0.0
        self.pnl_pct: float = 0.0
    
    def close(
        self,
        exit_date: datetime,
        exit_price: float,
        reason: str,
        transaction_cost: float = 0.0
    ) -> float:
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        
        # Calculate raw P&L
        if self.direction == 1:  # LONG
            raw_pnl = (exit_price - self.entry_price) * self.position_size
        else:  # SHORT
            raw_pnl = (self.entry_price - exit_price) * self.position_size
        
        # Subtract transaction costs
        self.pnl = raw_pnl - transaction_cost
        self.pnl_pct = (self.pnl / (self.entry_price * self.position_size)) * 100
        
        return self.pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'ticker': self.ticker,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'direction': 'LONG' if self.direction == 1 else 'SHORT',
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'confidence': self.confidence,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'holding_days': (self.exit_date - self.entry_date).days if self.exit_date else 0
        }


class BacktestEngine:
    """
    Realistic backtesting with transaction costs and risk management.
    
    Features:
    - Walk-forward testing (no look-ahead bias)
    - Transaction costs: spreads + slippage
    - Position sizing: 1% risk per trade
    - Stop losses: ATR-based
    - Max 2 simultaneous positions
    - Correlation management
    - Max holding period
    - Drawdown circuit breaker
    
    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        initial_capital: Starting capital.
        current_capital: Current portfolio value.
        open_positions: List of open Trade objects.
        closed_trades: List of closed Trade objects.
        equity_curve: Daily equity values.
    
    Example:
        >>> engine = BacktestEngine('config/config.yaml')
        >>> results = engine.run_backtest(signals_df, features_df)
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the BacktestEngine.
        
        Args:
            config_path: Path to configuration YAML file.
        """
        self.config = load_config(config_path)
        self.logger = get_logger('forex_signal_model.backtest')
        
        # Backtest configuration
        bt_config = self.config.get('backtest', {})
        self.initial_capital = bt_config.get('initial_capital', 10000)
        self.max_position_pct = bt_config.get('max_position_pct', 0.20)
        self.slippage_pips = bt_config.get('slippage_pips', 1)
        
        # Transaction costs per ticker
        self.transaction_costs = bt_config.get('transaction_costs', {
            'GBPUSD': 0.0002,
            'EURUSD': 0.00015,
            'GC': 0.00025
        })
        
        # Risk configuration
        risk_config = self.config.get('risk', {})
        self.max_risk_per_trade = risk_config.get('max_risk_per_trade', 0.01)
        self.max_drawdown_pct = risk_config.get('drawdown_circuit_breaker', 0.15)
        self.max_holding_days = risk_config.get('max_holding_days', 5)
        self.max_open_positions = risk_config.get('max_open_positions', 2)
        self.min_confidence = risk_config.get('min_confidence_to_trade', 40)
        
        # State
        self.current_capital = self.initial_capital
        self.open_positions: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []
        
        # Correlation tracking
        self.position_correlations = {
            ('GBPUSD', 'EURUSD'): 0.85,  # High correlation
            ('GBPUSD', 'GC_F'): -0.3,
            ('EURUSD', 'GC_F'): -0.2,
        }
        
        self.circuit_breaker_triggered = False
        
        self.logger.info(f"BacktestEngine initialized: Capital=${self.initial_capital}")
    
    def reset(self) -> None:
        """Reset backtest state."""
        self.current_capital = self.initial_capital
        self.open_positions = []
        self.closed_trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.circuit_breaker_triggered = False
    
    def _get_transaction_cost(self, ticker: str, price: float) -> float:
        """Get transaction cost for a ticker."""
        ticker_clean = ticker.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
        
        # Find matching cost
        for key, cost in self.transaction_costs.items():
            if key.replace('=', '_').replace('-', '_') in ticker_clean or \
               ticker_clean in key.replace('=', '_').replace('-', '_'):
                return cost * price
        
        # Default cost
        return 0.0002 * price
    
    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_value: float
    ) -> float:
        """
        Calculate position size based on 1% risk rule.
        
        Args:
            entry_price: Entry price.
            stop_loss: Stop loss price.
            account_value: Current account value.
        
        Returns:
            Position size (units).
        """
        if stop_loss == 0 or entry_price == 0:
            return 0
        
        risk_amount = account_value * self.max_risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0
        
        position_size = risk_amount / stop_distance
        
        # Cap at max position percentage
        max_position_value = account_value * self.max_position_pct
        max_units = max_position_value / entry_price
        
        return min(position_size, max_units)
    
    def _check_correlation(self, ticker: str) -> bool:
        """
        Check if opening a position would create high correlation exposure.
        
        Args:
            ticker: Ticker to check.
        
        Returns:
            True if OK to trade, False if too correlated.
        """
        ticker_clean = ticker.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
        
        for position in self.open_positions:
            pos_ticker = position.ticker.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
            
            # Check correlation (both directions)
            for (t1, t2), corr in self.position_correlations.items():
                t1_clean = t1.replace('=', '_').replace('-', '_')
                t2_clean = t2.replace('=', '_').replace('-', '_')
                
                if (ticker_clean in t1_clean or t1_clean in ticker_clean) and \
                   (pos_ticker in t2_clean or t2_clean in pos_ticker):
                    if abs(corr) > 0.7:
                        self.logger.debug(f"Skipping {ticker} due to correlation with {position.ticker}")
                        return False
                if (ticker_clean in t2_clean or t2_clean in ticker_clean) and \
                   (pos_ticker in t1_clean or t1_clean in pos_ticker):
                    if abs(corr) > 0.7:
                        return False
        
        return True
    
    def _check_drawdown(self) -> bool:
        """Check if drawdown circuit breaker should trigger."""
        if self.current_capital <= 0:
            return True
        
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        
        if drawdown >= self.max_drawdown_pct:
            if not self.circuit_breaker_triggered:
                self.logger.warning(f"Circuit breaker triggered! Drawdown: {drawdown*100:.1f}%")
                self.circuit_breaker_triggered = True
            return True
        
        return False
    
    def _get_price(
        self,
        df: pd.DataFrame,
        date: datetime,
        ticker: str,
        price_type: str = 'Close'
    ) -> Optional[float]:
        """Get price for a ticker on a date."""
        ticker_clean = ticker.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
        col = f'{ticker_clean}_{price_type}'
        
        if col not in df.columns:
            return None
        
        if date in df.index:
            return df.loc[date, col]
        
        return None
    
    def _apply_slippage(self, price: float, direction: int) -> float:
        """Apply slippage to price."""
        slippage = self.slippage_pips * 0.0001
        if direction == 1:  # BUY - pay higher
            return price * (1 + slippage)
        else:  # SELL - receive lower
            return price * (1 - slippage)
    
    @timer_decorator
    def run_backtest(
        self,
        signals_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run full backtest.
        
        Args:
            signals_df: DataFrame with signals [Date, Ticker, Signal, Confidence, StopLoss].
            features_df: DataFrame with price data and features.
        
        Returns:
            Backtest results dictionary.
        """
        self.reset()
        self.logger.info("Starting backtest...")
        
        # Get unique dates
        signals_df = signals_df.copy()
        signals_df['Date'] = pd.to_datetime(signals_df['Date'])
        
        dates = sorted(signals_df['Date'].unique())
        
        for date in dates:
            if self._check_drawdown():
                break
            
            # Get day's signals
            day_signals = signals_df[signals_df['Date'] == date]
            
            # Update open positions (check stops, max holding, etc.)
            self._update_positions(date, features_df)
            
            # Process new signals
            for _, signal in day_signals.iterrows():
                self._process_signal(signal, date, features_df)
            
            # Record equity
            self._record_equity(date, features_df)
        
        # Close remaining positions
        if dates:
            self._close_all_positions(dates[-1], features_df, "End of backtest")
        
        # Calculate results
        results = self._calculate_results()
        
        self.logger.info(f"Backtest complete: {len(self.closed_trades)} trades, "
                        f"Final capital: ${self.current_capital:.2f}")
        
        return results
    
    def _process_signal(
        self,
        signal: pd.Series,
        date: datetime,
        features_df: pd.DataFrame
    ) -> None:
        """Process a single signal."""
        ticker = signal['Ticker']
        signal_value = signal['Signal']
        confidence = signal.get('Confidence', 50)
        stop_loss = signal.get('StopLoss', 0)
        
        # Skip HOLD signals
        if signal_value == 0:
            return
        
        # Check minimum confidence
        if confidence < self.min_confidence:
            return
        
        # Check max positions
        if len(self.open_positions) >= self.max_open_positions:
            return
        
        # Check correlation
        if not self._check_correlation(ticker):
            return
        
        # Check if already in this ticker
        for pos in self.open_positions:
            if pos.ticker == ticker:
                return
        
        # Get entry price
        entry_price = self._get_price(features_df, date, ticker, 'Close')
        if entry_price is None:
            return
        
        # Apply slippage
        entry_price = self._apply_slippage(entry_price, signal_value)
        
        # Calculate stop loss if not provided
        if stop_loss == 0:
            atr = self._get_price(features_df, date, ticker, 'ATR')
            if atr:
                if signal_value == 1:
                    stop_loss = entry_price - (atr * 2)
                else:
                    stop_loss = entry_price + (atr * 2)
        
        # Calculate position size
        position_size = self._calculate_position_size(
            entry_price, stop_loss, self.current_capital
        )
        
        if position_size <= 0:
            return
        
        # Create and open trade
        trade = Trade(
            ticker=ticker,
            entry_date=date,
            entry_price=entry_price,
            direction=signal_value,
            position_size=position_size,
            stop_loss=stop_loss,
            confidence=confidence
        )
        
        self.open_positions.append(trade)
        self.logger.debug(f"Opened {trade.direction} {ticker} @ {entry_price:.5f}")
    
    def _update_positions(
        self,
        date: datetime,
        features_df: pd.DataFrame
    ) -> None:
        """Update open positions, check for exits."""
        positions_to_close = []
        
        for trade in self.open_positions:
            current_price = self._get_price(features_df, date, trade.ticker, 'Close')
            if current_price is None:
                continue
            
            # Check stop loss
            if trade.direction == 1 and current_price <= trade.stop_loss:
                positions_to_close.append((trade, current_price, "Stop loss"))
                continue
            elif trade.direction == -1 and current_price >= trade.stop_loss:
                positions_to_close.append((trade, current_price, "Stop loss"))
                continue
            
            # Check max holding period
            holding_days = (date - trade.entry_date).days
            if holding_days >= self.max_holding_days:
                positions_to_close.append((trade, current_price, "Max holding"))
                continue
        
        # Close positions
        for trade, price, reason in positions_to_close:
            self._close_position(trade, date, price, reason)
    
    def _close_position(
        self,
        trade: Trade,
        date: datetime,
        exit_price: float,
        reason: str
    ) -> None:
        """Close a position."""
        # Apply slippage (opposite to entry)
        exit_price = self._apply_slippage(exit_price, -trade.direction)
        
        # Calculate transaction cost
        cost = self._get_transaction_cost(trade.ticker, trade.entry_price) * trade.position_size
        cost += self._get_transaction_cost(trade.ticker, exit_price) * trade.position_size
        
        # Close trade
        pnl = trade.close(date, exit_price, reason, cost)
        
        # Update capital
        self.current_capital += pnl
        
        # Move to closed trades
        self.open_positions.remove(trade)
        self.closed_trades.append(trade)
        
        self.logger.debug(f"Closed {trade.ticker}: PnL=${pnl:.2f} ({reason})")
    
    def _close_all_positions(
        self,
        date: datetime,
        features_df: pd.DataFrame,
        reason: str
    ) -> None:
        """Close all open positions."""
        for trade in list(self.open_positions):
            price = self._get_price(features_df, date, trade.ticker, 'Close')
            if price:
                self._close_position(trade, date, price, reason)
    
    def _record_equity(
        self,
        date: datetime,
        features_df: pd.DataFrame
    ) -> None:
        """Record daily equity value."""
        # Calculate unrealized P&L
        unrealized = 0
        for trade in self.open_positions:
            current_price = self._get_price(features_df, date, trade.ticker, 'Close')
            if current_price:
                if trade.direction == 1:
                    unrealized += (current_price - trade.entry_price) * trade.position_size
                else:
                    unrealized += (trade.entry_price - current_price) * trade.position_size
        
        total_equity = self.current_capital + unrealized
        
        self.equity_curve.append({
            'date': date,
            'equity': total_equity,
            'capital': self.current_capital,
            'unrealized': unrealized,
            'open_positions': len(self.open_positions)
        })
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]['equity']
            daily_return = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_return)
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results."""
        trades_df = pd.DataFrame([t.to_dict() for t in self.closed_trades])
        equity_df = pd.DataFrame(self.equity_curve)
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return_pct': (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            'total_trades': len(self.closed_trades),
            'trades_df': trades_df,
            'equity_df': equity_df,
            'circuit_breaker_triggered': self.circuit_breaker_triggered
        }
        
        if not trades_df.empty:
            results['winning_trades'] = (trades_df['pnl'] > 0).sum()
            results['losing_trades'] = (trades_df['pnl'] <= 0).sum()
            results['win_rate'] = results['winning_trades'] / len(trades_df) * 100
            results['avg_trade_pnl'] = trades_df['pnl'].mean()
            results['avg_trade_pnl_pct'] = trades_df['pnl_pct'].mean()
            results['best_trade'] = trades_df['pnl'].max()
            results['worst_trade'] = trades_df['pnl'].min()
            results['avg_holding_days'] = trades_df['holding_days'].mean()
        
        if not equity_df.empty:
            equity_df['drawdown'] = equity_df['equity'] / equity_df['equity'].cummax() - 1
            results['max_drawdown_pct'] = abs(equity_df['drawdown'].min()) * 100
        
        return results
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get closed trades as DataFrame."""
        return pd.DataFrame([t.to_dict() for t in self.closed_trades])
    
    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame(self.equity_curve)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("BacktestEngine module loaded.")
    print("Use with signals and features DataFrames.")
