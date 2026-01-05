"""
Position Sizing Module - Optimal position sizing for trades.

This module provides:
- Kelly criterion position sizing
- Volatility-adjusted sizing (ATR-based)
- Maximum position limits
- Risk-per-trade control
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calculate optimal position sizes for trades.
    
    Methods:
    - Kelly criterion (fraction of bankroll to bet)
    - Volatility scaling (ATR-based)
    - Fixed fractional
    - Combined approach
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize position sizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        sizing_config = self.config.get('position_sizing', {})
        
        # Kelly fraction (0.25 = quarter Kelly for safety)
        self.kelly_fraction = sizing_config.get('kelly_fraction', 0.25)
        
        # Maximum position as fraction of account
        self.max_position_pct = sizing_config.get('max_position_pct', 0.02)
        
        # Risk per trade as fraction of account
        self.risk_per_trade = sizing_config.get('risk_per_trade', 0.01)
        
        # Use volatility scaling
        self.use_volatility_scaling = sizing_config.get('volatility_scaling', True)
        
        # Base volatility for scaling (ATR as percentage)
        self.base_volatility = sizing_config.get('base_volatility', 0.01)  # 1%
        
        # Minimum position size
        self.min_position_pct = sizing_config.get('min_position_pct', 0.001)
        
    def kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly criterion position size.
        
        Kelly formula: f* = (bp - q) / b
        where:
        - b = odds ratio (avg_win / avg_loss)
        - p = probability of winning
        - q = probability of losing (1 - p)
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            
        Returns:
            Optimal fraction to bet (0-1)
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0
            
        b = avg_win / avg_loss  # Odds ratio
        p = win_rate
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # Apply fractional Kelly for safety
        kelly = kelly * self.kelly_fraction
        
        # Clamp to valid range
        kelly = max(0, min(kelly, self.max_position_pct))
        
        return kelly
    
    def volatility_adjusted_size(
        self,
        base_size: float,
        current_atr: float,
        price: float
    ) -> float:
        """
        Adjust position size based on current volatility.
        
        Higher volatility = smaller position
        Lower volatility = larger position
        
        Args:
            base_size: Base position size fraction
            current_atr: Current ATR value
            price: Current price
            
        Returns:
            Volatility-adjusted position size
        """
        if price == 0:
            return base_size
            
        # Convert ATR to percentage
        atr_pct = current_atr / price
        
        # Scale inversely with volatility
        if atr_pct > 0:
            vol_scalar = self.base_volatility / atr_pct
            vol_scalar = max(0.5, min(vol_scalar, 2.0))  # Limit scaling range
        else:
            vol_scalar = 1.0
            
        adjusted_size = base_size * vol_scalar
        
        # Apply limits
        adjusted_size = max(self.min_position_pct, min(adjusted_size, self.max_position_pct))
        
        return adjusted_size
    
    def fixed_fractional_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float
    ) -> Tuple[float, int]:
        """
        Calculate position size using fixed fractional method.
        
        Risk a fixed percentage of account per trade.
        
        Args:
            account_balance: Total account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Tuple of (position_value, units)
        """
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit == 0:
            return 0, 0
            
        # Maximum risk in currency
        max_risk = account_balance * self.risk_per_trade
        
        # Calculate units
        units = int(max_risk / risk_per_unit)
        
        # Calculate position value
        position_value = units * entry_price
        
        # Check max position limit
        max_position_value = account_balance * self.max_position_pct
        if position_value > max_position_value:
            units = int(max_position_value / entry_price)
            position_value = units * entry_price
            
        return position_value, units
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        current_atr: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> Dict:
        """
        Calculate optimal position size using all available methods.
        
        Args:
            account_balance: Total account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            current_atr: Current ATR (for volatility scaling)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average winning trade (for Kelly)
            avg_loss: Average losing trade (for Kelly)
            confidence: Signal confidence (for scaling)
            
        Returns:
            Dictionary with position sizing details
        """
        result = {
            'method': 'fixed_fractional',
            'position_pct': 0,
            'position_value': 0,
            'units': 0,
            'risk_amount': 0,
            'risk_pct': 0
        }
        
        # Base: Fixed fractional
        position_value, units = self.fixed_fractional_size(
            account_balance, entry_price, stop_loss
        )
        
        base_pct = position_value / account_balance if account_balance > 0 else 0
        
        # Apply Kelly if we have historical data
        if all(x is not None for x in [win_rate, avg_win, avg_loss]):
            kelly_pct = self.kelly_size(win_rate, avg_win, avg_loss)
            # Use minimum of Kelly and fixed fractional
            if kelly_pct > 0:
                base_pct = min(base_pct, kelly_pct)
                result['method'] = 'kelly_adjusted'
                result['kelly_pct'] = kelly_pct
        
        # Apply volatility scaling
        if self.use_volatility_scaling and current_atr is not None:
            base_pct = self.volatility_adjusted_size(base_pct, current_atr, entry_price)
            result['method'] += '_vol_scaled'
            result['atr_adjustment'] = current_atr / entry_price * 100
        
        # Apply confidence scaling (higher confidence = larger position)
        if confidence is not None:
            # Normalize confidence to 0-1 range, then scale
            conf_normalized = min(confidence / 100, 1.0) if confidence > 1 else confidence
            conf_scalar = 0.5 + (conf_normalized * 0.5)  # Scale 0.5 to 1.0
            base_pct = base_pct * conf_scalar
            result['confidence_scalar'] = conf_scalar
        
        # Final position calculation
        position_value = account_balance * base_pct
        units = int(position_value / entry_price) if entry_price > 0 else 0
        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount = units * risk_per_unit
        
        result.update({
            'position_pct': base_pct,
            'position_value': position_value,
            'units': units,
            'risk_amount': risk_amount,
            'risk_pct': risk_amount / account_balance if account_balance > 0 else 0
        })
        
        return result
    
    def size_for_dataframe(
        self,
        df: pd.DataFrame,
        account_balance: float,
        entry_col: str = 'Close',
        stop_col: str = 'stop_loss',
        atr_col: str = 'ATR_14',
        confidence_col: str = 'confidence'
    ) -> pd.DataFrame:
        """
        Calculate position sizes for a dataframe of signals.
        
        Args:
            df: DataFrame with signals
            account_balance: Account balance
            entry_col: Column with entry prices
            stop_col: Column with stop losses
            atr_col: Column with ATR values
            confidence_col: Column with confidence scores
            
        Returns:
            DataFrame with added position sizing columns
        """
        result = df.copy()
        
        position_pcts = []
        position_values = []
        units_list = []
        risk_amounts = []
        
        for idx, row in df.iterrows():
            entry = row.get(entry_col, 0)
            stop = row.get(stop_col, entry * 0.98)  # Default 2% stop
            atr = row.get(atr_col, None)
            conf = row.get(confidence_col, 50)
            
            sizing = self.calculate_position_size(
                account_balance=account_balance,
                entry_price=entry,
                stop_loss=stop,
                current_atr=atr,
                confidence=conf
            )
            
            position_pcts.append(sizing['position_pct'])
            position_values.append(sizing['position_value'])
            units_list.append(sizing['units'])
            risk_amounts.append(sizing['risk_amount'])
        
        result['position_pct'] = position_pcts
        result['position_value'] = position_values
        result['units'] = units_list
        result['risk_amount'] = risk_amounts
        
        return result


def main():
    """Test position sizer."""
    config = {
        'position_sizing': {
            'kelly_fraction': 0.25,
            'max_position_pct': 0.02,
            'risk_per_trade': 0.01,
            'volatility_scaling': True,
            'base_volatility': 0.01
        }
    }
    
    sizer = PositionSizer(config)
    
    # Test scenario
    account = 100000
    entry = 1.2500  # GBPUSD entry
    stop = 1.2450   # 50 pips stop
    atr = 0.0080    # 80 pips ATR
    
    result = sizer.calculate_position_size(
        account_balance=account,
        entry_price=entry,
        stop_loss=stop,
        current_atr=atr,
        win_rate=0.58,
        avg_win=0.0050,
        avg_loss=0.0050,
        confidence=65
    )
    
    print("\n📊 Position Sizing Result:")
    print(f"   Account: ${account:,.0f}")
    print(f"   Entry: {entry}")
    print(f"   Stop: {stop}")
    print(f"   Method: {result['method']}")
    print(f"   Position %: {result['position_pct']*100:.2f}%")
    print(f"   Position Value: ${result['position_value']:,.2f}")
    print(f"   Units: {result['units']:,}")
    print(f"   Risk Amount: ${result['risk_amount']:,.2f}")
    print(f"   Risk %: {result['risk_pct']*100:.2f}%")


if __name__ == "__main__":
    main()
