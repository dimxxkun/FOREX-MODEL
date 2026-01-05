"""
Backtesting Package for Forex Signal Model.

Contains:
- BacktestEngine: Realistic backtesting with transaction costs
- Performance metrics calculation
- Visualization utilities
"""

from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import calculate_performance_metrics
from src.backtesting.visualizations import (
    plot_equity_curve,
    plot_monthly_returns,
    plot_trade_analysis,
    generate_performance_report
)

__all__ = [
    'BacktestEngine',
    'calculate_performance_metrics',
    'plot_equity_curve',
    'plot_monthly_returns',
    'plot_trade_analysis',
    'generate_performance_report'
]
