"""
Tests for Backtesting Engine.

Unit tests for the backtesting framework.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_signals():
    """Create sample signals DataFrame."""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    
    signals = []
    for ticker in ['GBPUSD', 'EURUSD']:
        for i, date in enumerate(dates):
            # Alternate signals
            signal = 1 if i % 10 < 5 else -1 if i % 10 < 8 else 0
            signals.append({
                'Date': date,
                'Ticker': ticker,
                'Signal': signal,
                'Confidence': 50 + np.random.randint(-10, 20),
                'StopLoss': 0.01 if signal != 0 else 0
            })
    
    return pd.DataFrame(signals)


@pytest.fixture
def sample_features():
    """Create sample features DataFrame."""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    
    data = {
        'GBPUSD_Close': np.random.uniform(1.24, 1.28, 50),
        'GBPUSD_ATR': np.random.uniform(0.005, 0.01, 50),
        'GBPUSD_Open': np.random.uniform(1.24, 1.28, 50),
        'GBPUSD_High': np.random.uniform(1.25, 1.29, 50),
        'GBPUSD_Low': np.random.uniform(1.23, 1.27, 50),
        'EURUSD_Close': np.random.uniform(1.08, 1.12, 50),
        'EURUSD_ATR': np.random.uniform(0.004, 0.008, 50),
        'EURUSD_Open': np.random.uniform(1.08, 1.12, 50),
        'EURUSD_High': np.random.uniform(1.09, 1.13, 50),
        'EURUSD_Low': np.random.uniform(1.07, 1.11, 50),
    }
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        'backtest': {
            'initial_capital': 10000,
            'transaction_costs': {
                'GBPUSD': 0.0002,
                'EURUSD': 0.00015
            },
            'slippage_pips': 1,
            'max_position_pct': 0.20
        },
        'risk': {
            'max_risk_per_trade': 0.01,
            'max_drawdown_pct': 20.0,
            'drawdown_circuit_breaker': 0.15,
            'max_holding_days': 5,
            'max_open_positions': 2,
            'min_confidence_to_trade': 40,
            'atr_stop_multiplier': 2.0
        }
    }


# =============================================================================
# TESTS
# =============================================================================

class TestTrade:
    """Tests for Trade class."""
    
    def test_trade_creation(self):
        """Test creating a trade object."""
        from src.backtesting.engine import Trade
        
        trade = Trade(
            ticker='GBPUSD',
            entry_date=datetime(2024, 1, 1),
            entry_price=1.25,
            direction=1,
            position_size=1000,
            stop_loss=1.24,
            confidence=60
        )
        
        assert trade.ticker == 'GBPUSD'
        assert trade.entry_price == 1.25
        assert trade.direction == 1
        assert trade.pnl == 0
    
    def test_trade_close_profit(self):
        """Test closing a profitable trade."""
        from src.backtesting.engine import Trade
        
        trade = Trade(
            ticker='GBPUSD',
            entry_date=datetime(2024, 1, 1),
            entry_price=1.25,
            direction=1,
            position_size=1000,
            stop_loss=1.24,
            confidence=60
        )
        
        pnl = trade.close(
            exit_date=datetime(2024, 1, 5),
            exit_price=1.26,
            reason='Take profit',
            transaction_cost=0.5
        )
        
        expected_pnl = (1.26 - 1.25) * 1000 - 0.5  # 10 - 0.5 = 9.5
        assert abs(pnl - expected_pnl) < 0.01
        assert trade.exit_reason == 'Take profit'
    
    def test_trade_close_loss(self):
        """Test closing a losing trade."""
        from src.backtesting.engine import Trade
        
        trade = Trade(
            ticker='GBPUSD',
            entry_date=datetime(2024, 1, 1),
            entry_price=1.25,
            direction=1,
            position_size=1000,
            stop_loss=1.24,
            confidence=60
        )
        
        pnl = trade.close(
            exit_date=datetime(2024, 1, 5),
            exit_price=1.24,
            reason='Stop loss',
            transaction_cost=0.5
        )
        
        expected_pnl = (1.24 - 1.25) * 1000 - 0.5  # -10 - 0.5 = -10.5
        assert abs(pnl - expected_pnl) < 0.01
    
    def test_trade_to_dict(self):
        """Test converting trade to dictionary."""
        from src.backtesting.engine import Trade
        
        trade = Trade(
            ticker='GBPUSD',
            entry_date=datetime(2024, 1, 1),
            entry_price=1.25,
            direction=1,
            position_size=1000,
            stop_loss=1.24,
            confidence=60
        )
        trade.close(datetime(2024, 1, 5), 1.26, 'Test')
        
        d = trade.to_dict()
        
        assert 'ticker' in d
        assert 'pnl' in d
        assert d['direction'] == 'LONG'


class TestBacktestEngine:
    """Tests for BacktestEngine class."""
    
    @patch('src.backtesting.engine.load_config')
    def test_engine_initialization(self, mock_load_config, mock_config):
        """Test engine initialization."""
        mock_load_config.return_value = mock_config
        
        from src.backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        
        assert engine.initial_capital == 10000
        assert engine.max_risk_per_trade == 0.01
    
    @patch('src.backtesting.engine.load_config')
    def test_position_sizing(self, mock_load_config, mock_config):
        """Test position size calculation."""
        mock_load_config.return_value = mock_config
        
        from src.backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        
        # 1% of 10000 = 100 risk
        # Stop distance = 0.01 (100 pips)
        # Position size = 100 / 0.01 = 10000 max
        size = engine._calculate_position_size(
            entry_price=1.25,
            stop_loss=1.24,
            account_value=10000
        )
        
        assert size > 0
        assert size <= 10000 * 0.20 / 1.25  # Max 20% position
    
    @patch('src.backtesting.engine.load_config')
    def test_reset(self, mock_load_config, mock_config):
        """Test engine reset."""
        mock_load_config.return_value = mock_config
        
        from src.backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        
        # Modify state
        engine.current_capital = 5000
        engine.closed_trades = [1, 2, 3]
        
        engine.reset()
        
        assert engine.current_capital == 10000
        assert len(engine.closed_trades) == 0
    
    @patch('src.backtesting.engine.load_config')
    def test_transaction_cost(self, mock_load_config, mock_config):
        """Test transaction cost calculation."""
        mock_load_config.return_value = mock_config
        
        from src.backtesting.engine import BacktestEngine
        engine = BacktestEngine()
        
        cost = engine._get_transaction_cost('GBPUSD', 1.25)
        
        # 0.0002 * 1.25 = 0.00025
        assert abs(cost - 0.00025) < 0.0001


class TestMetrics:
    """Tests for performance metrics."""
    
    def test_calculate_metrics_empty(self):
        """Test metrics with empty trades."""
        from src.backtesting.metrics import calculate_performance_metrics
        
        trades_df = pd.DataFrame()
        equity_df = pd.DataFrame()
        
        metrics = calculate_performance_metrics(trades_df, equity_df)
        
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        from src.backtesting.metrics import calculate_performance_metrics
        
        trades_df = pd.DataFrame({
            'pnl': [100, -50, 75, -25, 50],
            'pnl_pct': [1.0, -0.5, 0.75, -0.25, 0.5],
            'holding_days': [3, 2, 4, 1, 2]
        })
        
        equity_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'equity': [10000, 10100, 10050, 10125, 10100]
        })
        
        metrics = calculate_performance_metrics(trades_df, equity_df)
        
        assert metrics['total_trades'] == 5
        assert metrics['winning_trades'] == 3
        assert metrics['losing_trades'] == 2
        assert abs(metrics['win_rate'] - 60.0) < 0.1
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        from src.backtesting.metrics import calculate_performance_metrics
        
        trades_df = pd.DataFrame({
            'pnl': [200, -100],  # 2:1 profit factor
            'pnl_pct': [2.0, -1.0],
            'holding_days': [3, 2]
        })
        
        equity_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2),
            'equity': [10000, 10100]
        })
        
        metrics = calculate_performance_metrics(trades_df, equity_df)
        
        assert abs(metrics['profit_factor'] - 2.0) < 0.01


class TestRiskManager:
    """Tests for RiskManager class."""
    
    @patch('src.risk_management.load_config')
    def test_position_sizing(self, mock_load_config, mock_config):
        """Test position sizing."""
        mock_load_config.return_value = mock_config
        
        from src.risk_management import RiskManager
        rm = RiskManager()
        
        size = rm.calculate_position_size(
            entry_price=1.25,
            stop_loss_price=1.24,
            account_value=10000
        )
        
        # Risk = 1% of 10000 = 100
        # Stop = 0.01
        # Size = 100 / 0.01 = 10000
        # Capped at 20% = 2000 / 1.25 = 1600
        assert size > 0
        assert size <= 10000 * 0.20 / 1.25
    
    @patch('src.risk_management.load_config')
    def test_stop_loss_long(self, mock_load_config, mock_config):
        """Test stop loss for long position."""
        mock_load_config.return_value = mock_config
        
        from src.risk_management import RiskManager
        rm = RiskManager()
        
        stop = rm.get_stop_loss_price(
            entry_price=1.25,
            atr=0.01,
            direction=1
        )
        
        # 1.25 - (0.01 * 2) = 1.23
        assert abs(stop - 1.23) < 0.001
    
    @patch('src.risk_management.load_config')
    def test_stop_loss_short(self, mock_load_config, mock_config):
        """Test stop loss for short position."""
        mock_load_config.return_value = mock_config
        
        from src.risk_management import RiskManager
        rm = RiskManager()
        
        stop = rm.get_stop_loss_price(
            entry_price=1.25,
            atr=0.01,
            direction=-1
        )
        
        # 1.25 + (0.01 * 2) = 1.27
        assert abs(stop - 1.27) < 0.001
    
    @patch('src.risk_management.load_config')
    def test_correlation_check(self, mock_load_config, mock_config):
        """Test correlation checking."""
        mock_load_config.return_value = mock_config
        
        from src.risk_management import RiskManager
        rm = RiskManager()
        
        # GBPUSD and EURUSD are highly correlated
        allowed, conflict = rm.check_correlation('EURUSD', ['GBPUSD'])
        
        assert allowed == False
        assert conflict == 'GBPUSD'
    
    @patch('src.risk_management.load_config')
    def test_drawdown_check_safe(self, mock_load_config, mock_config):
        """Test drawdown check when safe."""
        mock_load_config.return_value = mock_config
        
        from src.risk_management import RiskManager
        rm = RiskManager()
        
        is_safe, dd = rm.check_drawdown(9500, 10000)
        
        assert is_safe == True
        assert dd == 5.0  # 5% drawdown
    
    @patch('src.risk_management.load_config')
    def test_drawdown_check_triggered(self, mock_load_config, mock_config):
        """Test drawdown check when circuit breaker triggered."""
        mock_load_config.return_value = mock_config
        
        from src.risk_management import RiskManager
        rm = RiskManager()
        
        is_safe, dd = rm.check_drawdown(8000, 10000)
        
        assert is_safe == False  # 20% > 15% threshold
        assert dd == 20.0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
