"""
Signal Filter Module - Filters trading signals for quality.

This module provides:
- Confidence threshold filtering
- Regime-based filtering
- Trade frequency control
- Signal quality scoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SignalFilter:
    """
    Filter trading signals to improve quality and reduce overtrading.
    
    Key features:
    - Only pass high-confidence signals (>threshold)
    - Skip signals in unfavorable market regimes
    - Enforce minimum gap between trades
    - Combine multiple quality checks
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize signal filter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        filter_config = self.config.get('signal_filter', {})
        
        # Confidence threshold (0-1, where 0.5 is neutral)
        self.confidence_threshold = filter_config.get('confidence_threshold', 0.60)
        
        # Regimes to trade in (skip others)
        self.allowed_regimes = filter_config.get('allowed_regimes', ['trending', 'normal'])
        
        # Minimum days between trades
        self.min_trade_gap = filter_config.get('min_trade_gap', 1)
        
        # Risk-reward minimum ratio
        self.min_risk_reward = filter_config.get('min_risk_reward', 1.5)
        
        # Track last trade date
        self.last_trade_date = None
        
        # Statistics
        self.stats = {
            'total_signals': 0,
            'passed_confidence': 0,
            'passed_regime': 0,
            'passed_frequency': 0,
            'final_passed': 0
        }
        
    def reset_stats(self):
        """Reset filter statistics."""
        self.stats = {k: 0 for k in self.stats}
        self.last_trade_date = None
        
    def check_confidence(
        self,
        confidence: float,
        direction: int
    ) -> bool:
        """
        Check if signal meets confidence threshold.
        
        Args:
            confidence: Model confidence score (0-100 scale)
            direction: Signal direction (1=buy, 0=sell)
            
        Returns:
            True if signal passes confidence check
        """
        # Normalize to 0-1 if on 0-100 scale
        if confidence > 1:
            confidence = confidence / 100
            
        # For a signal to pass, confidence must exceed threshold
        # Confidence represents certainty of the prediction
        passes = confidence >= self.confidence_threshold
        
        if passes:
            self.stats['passed_confidence'] += 1
            
        return passes
    
    def check_regime(
        self,
        regime: str
    ) -> bool:
        """
        Check if current market regime is favorable.
        
        Args:
            regime: Current market regime classification
            
        Returns:
            True if regime is in allowed list
        """
        if regime is None:
            return True  # If no regime info, allow
            
        regime_lower = str(regime).lower()
        passes = any(allowed.lower() in regime_lower for allowed in self.allowed_regimes)
        
        if passes:
            self.stats['passed_regime'] += 1
            
        return passes
    
    def check_trade_frequency(
        self,
        current_date: datetime
    ) -> bool:
        """
        Check if enough time has passed since last trade.
        
        Args:
            current_date: Date of proposed trade
            
        Returns:
            True if minimum gap has passed
        """
        if self.last_trade_date is None:
            return True
            
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        if isinstance(self.last_trade_date, str):
            self.last_trade_date = pd.to_datetime(self.last_trade_date)
            
        days_since = (current_date - self.last_trade_date).days
        passes = days_since >= self.min_trade_gap
        
        if passes:
            self.stats['passed_frequency'] += 1
            
        return passes
    
    def check_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        direction: int
    ) -> bool:
        """
        Check if trade meets minimum risk-reward ratio.
        
        Args:
            entry_price: Proposed entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            direction: 1 for long, 0 for short
            
        Returns:
            True if risk-reward ratio meets minimum
        """
        if direction == 1:  # Long
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:  # Short
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            
        if risk <= 0:
            return False
            
        rr_ratio = reward / risk
        return rr_ratio >= self.min_risk_reward
    
    def filter_signal(
        self,
        signal: int,
        confidence: float,
        regime: Optional[str] = None,
        current_date: Optional[datetime] = None,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Tuple[int, float, str]:
        """
        Apply all filters to a trading signal.
        
        Args:
            signal: Raw signal (1=buy, 0=sell, -1=hold)
            confidence: Model confidence (0-100)
            regime: Current market regime
            current_date: Date of signal
            entry_price: Proposed entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Tuple of (filtered_signal, adjusted_confidence, reason)
        """
        self.stats['total_signals'] += 1
        
        # If signal is hold (-1), pass through
        if signal == -1:
            return signal, confidence, "hold_signal"
            
        # Check 1: Confidence
        if not self.check_confidence(confidence, signal):
            return -1, confidence, f"low_confidence_{confidence:.1f}"
            
        # Check 2: Regime
        if regime is not None and not self.check_regime(regime):
            return -1, confidence, f"unfavorable_regime_{regime}"
            
        # Check 3: Trade frequency
        if current_date is not None and not self.check_trade_frequency(current_date):
            return -1, confidence, "too_frequent"
            
        # Check 4: Risk-reward (if prices provided)
        if all(x is not None for x in [entry_price, stop_loss, take_profit]):
            if not self.check_risk_reward(entry_price, stop_loss, take_profit, signal):
                return -1, confidence, "poor_risk_reward"
        
        # All checks passed
        self.stats['final_passed'] += 1
        if current_date is not None:
            self.last_trade_date = current_date
            
        return signal, confidence, "passed"
    
    def filter_signals_batch(
        self,
        signals: pd.Series,
        confidences: pd.Series,
        regimes: Optional[pd.Series] = None,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Filter a batch of signals.
        
        Args:
            signals: Series of signals
            confidences: Series of confidence scores
            regimes: Optional series of regime classifications
            dates: Optional datetime index
            
        Returns:
            Tuple of (filtered_signals, adjusted_confidences, filter_reasons)
        """
        self.reset_stats()
        
        filtered_signals = []
        adjusted_confidences = []
        reasons = []
        
        for i in range(len(signals)):
            signal = signals.iloc[i]
            conf = confidences.iloc[i]
            regime = regimes.iloc[i] if regimes is not None else None
            date = dates[i] if dates is not None else None
            
            f_signal, f_conf, reason = self.filter_signal(
                signal=signal,
                confidence=conf,
                regime=regime,
                current_date=date
            )
            
            filtered_signals.append(f_signal)
            adjusted_confidences.append(f_conf)
            reasons.append(reason)
            
        logger.info(f"Signal Filter Stats: {self.stats}")
        
        return (
            pd.Series(filtered_signals, index=signals.index),
            pd.Series(adjusted_confidences, index=signals.index),
            pd.Series(reasons, index=signals.index)
        )
    
    def get_filter_report(self) -> Dict:
        """Get filtering statistics."""
        total = self.stats['total_signals']
        if total == 0:
            return self.stats
            
        return {
            **self.stats,
            'pass_rate': self.stats['final_passed'] / total * 100,
            'confidence_filter_rate': (total - self.stats['passed_confidence']) / total * 100,
            'regime_filter_rate': (self.stats['passed_confidence'] - self.stats['passed_regime']) / total * 100 if self.stats['passed_confidence'] > 0 else 0
        }


def main():
    """Test signal filter."""
    # Create test signals
    np.random.seed(42)
    n = 100
    
    signals = pd.Series(np.random.choice([0, 1], n))
    confidences = pd.Series(np.random.uniform(40, 80, n))
    regimes = pd.Series(np.random.choice(['trending', 'ranging', 'volatile'], n))
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    # Create filter
    filter_config = {
        'signal_filter': {
            'confidence_threshold': 0.60,
            'allowed_regimes': ['trending'],
            'min_trade_gap': 2
        }
    }
    
    signal_filter = SignalFilter(filter_config)
    
    # Filter signals
    filtered, confs, reasons = signal_filter.filter_signals_batch(
        signals, confidences, regimes, dates
    )
    
    # Report
    print("\n📊 Signal Filter Results:")
    print(f"   Original signals: {len(signals)}")
    print(f"   Passed signals: {(filtered != -1).sum()}")
    print(f"   Filter rate: {(filtered == -1).sum() / len(signals) * 100:.1f}%")
    print(f"\n   Breakdown by reason:")
    print(reasons.value_counts())
    print(f"\n   Filter stats: {signal_filter.get_filter_report()}")


if __name__ == "__main__":
    main()
