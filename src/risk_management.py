"""
Risk Management Module for Forex Signal Model.

Position sizing, stop losses, correlation management, and risk controls.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import get_logger, load_config


class RiskManager:
    """
    Position sizing, stop losses, correlation management.
    
    Implements conservative risk management rules:
    - 1% risk per trade max
    - 2× ATR stop losses
    - Correlation-based position reduction
    - Drawdown circuit breaker
    - Maximum open positions limit
    
    Attributes:
        config: Configuration dictionary.
        logger: Logger instance.
        max_risk_per_trade: Maximum risk as fraction.
        max_drawdown: Circuit breaker threshold.
    
    Example:
        >>> rm = RiskManager('config/config.yaml')
        >>> size = rm.calculate_position_size(1.2500, 1.2400, 10000)
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize the RiskManager.
        
        Args:
            config_path: Path to configuration YAML file.
        """
        self.config = load_config(config_path)
        self.logger = get_logger('forex_signal_model.risk_management')
        
        # Risk configuration
        risk_config = self.config.get('risk', {})
        self.max_risk_per_trade = risk_config.get('max_risk_per_trade', 0.01)
        self.max_drawdown = risk_config.get('max_drawdown_pct', 20.0) / 100
        self.max_open_positions = risk_config.get('max_open_positions', 2)
        self.min_confidence = risk_config.get('min_confidence_to_trade', 40)
        self.atr_stop_multiplier = risk_config.get('atr_stop_multiplier', 2.0)
        
        # Backtest config
        bt_config = self.config.get('backtest', {})
        self.max_position_pct = bt_config.get('max_position_pct', 0.20)
        
        # Correlation matrix (can be computed from data)
        self.correlation_matrix = {
            ('GBPUSD', 'EURUSD'): 0.85,
            ('GBPUSD', 'XAUUSD'): -0.30,
            ('GBPUSD', 'GC_F'): -0.30,
            ('EURUSD', 'XAUUSD'): -0.20,
            ('EURUSD', 'GC_F'): -0.20,
        }
        
        self.logger.info(f"RiskManager initialized: max_risk={self.max_risk_per_trade}, "
                        f"max_dd={self.max_drawdown}")
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        account_value: float,
        leverage: float = 1.0
    ) -> float:
        """
        Calculate position size based on 1% risk rule.
        
        Position size = Risk Amount / Stop Distance
        
        Args:
            entry_price: Entry price.
            stop_loss_price: Stop loss price.
            account_value: Current account value.
            leverage: Optional leverage multiplier.
        
        Returns:
            Position size (units).
        """
        if entry_price <= 0 or stop_loss_price <= 0 or account_value <= 0:
            return 0
        
        # Calculate risk amount
        risk_amount = account_value * self.max_risk_per_trade
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss_price)
        
        if stop_distance == 0:
            self.logger.warning("Stop distance is zero, cannot calculate position size")
            return 0
        
        # Calculate base position size
        position_size = risk_amount / stop_distance
        
        # Apply leverage
        position_size *= leverage
        
        # Cap at maximum position percentage
        max_position_value = account_value * self.max_position_pct
        max_units = max_position_value / entry_price
        position_size = min(position_size, max_units)
        
        return max(0, position_size)
    
    def get_stop_loss_price(
        self,
        entry_price: float,
        atr: float,
        direction: int,
        multiplier: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price based on ATR.
        
        Args:
            entry_price: Entry price.
            atr: Average True Range value.
            direction: 1 for LONG, -1 for SHORT.
            multiplier: ATR multiplier (default from config).
        
        Returns:
            Stop loss price.
        """
        if multiplier is None:
            multiplier = self.atr_stop_multiplier
        
        stop_distance = atr * multiplier
        
        if direction == 1:  # LONG
            stop_loss = entry_price - stop_distance
        else:  # SHORT
            stop_loss = entry_price + stop_distance
        
        return stop_loss
    
    def check_correlation(
        self,
        ticker: str,
        open_positions: List[str],
        threshold: float = 0.7
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if opening a position would create high correlation exposure.
        
        Args:
            ticker: Ticker to check.
            open_positions: List of currently open ticker symbols.
            threshold: Correlation threshold (0-1).
        
        Returns:
            Tuple of (is_allowed, conflicting_ticker).
        """
        ticker_clean = self._clean_ticker(ticker)
        
        for open_ticker in open_positions:
            open_clean = self._clean_ticker(open_ticker)
            
            # Check both orderings
            corr = self._get_correlation(ticker_clean, open_clean)
            
            if abs(corr) > threshold:
                self.logger.debug(f"High correlation ({corr:.2f}) between {ticker} and {open_ticker}")
                return False, open_ticker
        
        return True, None
    
    def _get_correlation(self, ticker1: str, ticker2: str) -> float:
        """Get correlation between two tickers."""
        # Check both orderings
        if (ticker1, ticker2) in self.correlation_matrix:
            return self.correlation_matrix[(ticker1, ticker2)]
        if (ticker2, ticker1) in self.correlation_matrix:
            return self.correlation_matrix[(ticker2, ticker1)]
        
        # Default: assume low correlation
        return 0.0
    
    def _clean_ticker(self, ticker: str) -> str:
        """Normalize ticker name for comparison."""
        return ticker.replace('=X', '').replace('^', '').replace('-', '_').replace('.', '_')
    
    def adjust_size_for_correlation(
        self,
        position_size: float,
        ticker: str,
        open_positions: Dict[str, float],
        reduction_factor: float = 0.5
    ) -> float:
        """
        Reduce position size if correlated positions exist.
        
        Args:
            position_size: Calculated position size.
            ticker: Ticker for new position.
            open_positions: Dict of ticker -> position size.
            reduction_factor: Reduction multiplier (0-1).
        
        Returns:
            Adjusted position size.
        """
        ticker_clean = self._clean_ticker(ticker)
        
        for open_ticker in open_positions.keys():
            open_clean = self._clean_ticker(open_ticker)
            corr = abs(self._get_correlation(ticker_clean, open_clean))
            
            if corr > 0.5:
                # Reduce size proportionally to correlation
                reduction = 1 - (corr * reduction_factor)
                position_size *= reduction
                self.logger.debug(f"Reduced position size by {(1-reduction)*100:.0f}% "
                                 f"due to correlation with {open_ticker}")
        
        return position_size
    
    def check_max_positions(
        self,
        current_positions: int
    ) -> bool:
        """
        Check if maximum positions limit is reached.
        
        Args:
            current_positions: Number of currently open positions.
        
        Returns:
            True if can open more, False if at limit.
        """
        return current_positions < self.max_open_positions
    
    def check_drawdown(
        self,
        current_equity: float,
        peak_equity: float
    ) -> Tuple[bool, float]:
        """
        Check if drawdown exceeds circuit breaker threshold.
        
        Args:
            current_equity: Current account equity.
            peak_equity: Peak account equity.
        
        Returns:
            Tuple of (is_safe, current_drawdown_pct).
        """
        if peak_equity <= 0:
            return True, 0.0
        
        drawdown = (peak_equity - current_equity) / peak_equity
        
        if drawdown >= self.max_drawdown:
            self.logger.warning(f"Drawdown circuit breaker triggered: {drawdown*100:.1f}%")
            return False, drawdown * 100
        
        return True, drawdown * 100
    
    def check_confidence(self, confidence: float) -> bool:
        """
        Check if signal confidence meets minimum threshold.
        
        Args:
            confidence: Signal confidence (0-100).
        
        Returns:
            True if confidence is sufficient.
        """
        return confidence >= self.min_confidence
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        fraction: float = 0.25
    ) -> float:
        """
        Calculate Kelly Criterion position sizing.
        
        Args:
            win_rate: Historical win rate (0-1).
            avg_win: Average winning trade.
            avg_loss: Average losing trade (positive number).
            fraction: Kelly fraction to use (0.25 = quarter Kelly).
        
        Returns:
            Optimal position size fraction.
        """
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula: (bp - q) / b
        # where b = win/loss ratio, p = win rate, q = loss rate
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Use fractional Kelly for safety
        return max(0, kelly * fraction)
    
    def validate_trade(
        self,
        ticker: str,
        signal: int,
        confidence: float,
        entry_price: float,
        stop_loss: float,
        account_value: float,
        open_positions: List[str],
        current_equity: float,
        peak_equity: float
    ) -> Dict[str, Any]:
        """
        Comprehensive trade validation.
        
        Args:
            ticker: Ticker symbol.
            signal: Signal direction (1=BUY, -1=SELL).
            confidence: Signal confidence.
            entry_price: Entry price.
            stop_loss: Stop loss price.
            account_value: Current account value.
            open_positions: List of open position tickers.
            current_equity: Current equity.
            peak_equity: Peak equity.
        
        Returns:
            Validation result dictionary.
        """
        result = {
            'valid': True,
            'position_size': 0,
            'reasons': []
        }
        
        # Check confidence
        if not self.check_confidence(confidence):
            result['valid'] = False
            result['reasons'].append(f"Confidence {confidence:.1f} below minimum {self.min_confidence}")
            return result
        
        # Check max positions
        if not self.check_max_positions(len(open_positions)):
            result['valid'] = False
            result['reasons'].append(f"Max positions ({self.max_open_positions}) reached")
            return result
        
        # Check drawdown
        is_safe, dd = self.check_drawdown(current_equity, peak_equity)
        if not is_safe:
            result['valid'] = False
            result['reasons'].append(f"Circuit breaker: drawdown {dd:.1f}%")
            return result
        
        # Check correlation
        is_allowed, conflict = self.check_correlation(ticker, open_positions)
        if not is_allowed:
            result['valid'] = False
            result['reasons'].append(f"High correlation with {conflict}")
            return result
        
        # Calculate position size
        position_size = self.calculate_position_size(
            entry_price, stop_loss, account_value
        )
        
        if position_size <= 0:
            result['valid'] = False
            result['reasons'].append("Position size calculation failed")
            return result
        
        # Adjust for correlation
        open_positions_dict = {t: 1 for t in open_positions}  # Placeholder
        position_size = self.adjust_size_for_correlation(
            position_size, ticker, open_positions_dict
        )
        
        result['position_size'] = position_size
        
        return result
    
    def update_correlation_matrix(self, df: pd.DataFrame) -> None:
        """
        Update correlation matrix from actual data.
        
        Args:
            df: Features DataFrame with return columns.
        """
        return_cols = [c for c in df.columns if '_Return_1d' in c]
        
        if len(return_cols) < 2:
            return
        
        returns_df = df[return_cols].dropna()
        corr_matrix = returns_df.corr()
        
        # Update internal matrix
        for i, col1 in enumerate(return_cols):
            for col2 in return_cols[i+1:]:
                ticker1 = col1.replace('_Return_1d', '')
                ticker2 = col2.replace('_Return_1d', '')
                corr = corr_matrix.loc[col1, col2]
                self.correlation_matrix[(ticker1, ticker2)] = corr
        
        self.logger.info(f"Updated correlation matrix with {len(return_cols)} tickers")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Example usage
    rm = RiskManager()
    
    # Test position sizing
    size = rm.calculate_position_size(
        entry_price=1.2500,
        stop_loss_price=1.2400,
        account_value=10000
    )
    print(f"Position size: {size:.2f} units")
    
    # Test stop loss
    stop = rm.get_stop_loss_price(
        entry_price=1.2500,
        atr=0.0050,
        direction=1
    )
    print(f"Stop loss: {stop:.5f}")
    
    # Test correlation check
    allowed, conflict = rm.check_correlation('EURUSD', ['GBPUSD'])
    print(f"Can trade EURUSD: {allowed}, Conflict: {conflict}")
