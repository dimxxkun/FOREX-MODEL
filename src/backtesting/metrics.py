"""
Performance Metrics for Forex Signal Model Backtesting.

Comprehensive trading performance calculations.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def calculate_performance_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
    """
    Calculate comprehensive trading metrics.
    
    Args:
        trades_df: DataFrame of closed trades.
        equity_df: DataFrame with daily equity values.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
    
    Returns:
        Dictionary with all performance metrics.
    """
    metrics = {}
    
    # Basic statistics
    if trades_df.empty:
        return _empty_metrics()
    
    # =========================================================================
    # RETURN METRICS
    # =========================================================================
    
    total_pnl = trades_df['pnl'].sum()
    initial_capital = equity_df['equity'].iloc[0] if not equity_df.empty else 10000
    
    metrics['total_pnl'] = total_pnl
    metrics['total_return_pct'] = (total_pnl / initial_capital) * 100
    
    # Calculate CAGR
    if not equity_df.empty and len(equity_df) > 1:
        start_equity = equity_df['equity'].iloc[0]
        end_equity = equity_df['equity'].iloc[-1]
        n_years = len(equity_df) / 252  # Trading days per year
        if start_equity > 0 and n_years > 0:
            metrics['cagr'] = ((end_equity / start_equity) ** (1 / n_years) - 1) * 100
        else:
            metrics['cagr'] = 0
    else:
        metrics['cagr'] = 0
    
    # =========================================================================
    # WIN/LOSS METRICS
    # =========================================================================
    
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    metrics['total_trades'] = len(trades_df)
    metrics['winning_trades'] = len(winning_trades)
    metrics['losing_trades'] = len(losing_trades)
    metrics['win_rate'] = (len(winning_trades) / len(trades_df)) * 100 if len(trades_df) > 0 else 0
    
    # Average trade
    metrics['avg_trade_pnl'] = trades_df['pnl'].mean()
    metrics['avg_trade_pnl_pct'] = trades_df['pnl_pct'].mean()
    metrics['avg_winning_trade'] = winning_trades['pnl'].mean() if not winning_trades.empty else 0
    metrics['avg_losing_trade'] = losing_trades['pnl'].mean() if not losing_trades.empty else 0
    
    # Best/worst
    metrics['best_trade'] = trades_df['pnl'].max()
    metrics['worst_trade'] = trades_df['pnl'].min()
    metrics['best_trade_pct'] = trades_df['pnl_pct'].max()
    metrics['worst_trade_pct'] = trades_df['pnl_pct'].min()
    
    # =========================================================================
    # PROFIT FACTOR & EXPECTANCY
    # =========================================================================
    
    gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
    gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
    
    metrics['gross_profit'] = gross_profit
    metrics['gross_loss'] = gross_loss
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Expectancy
    win_rate = metrics['win_rate'] / 100
    avg_win = metrics['avg_winning_trade']
    avg_loss = abs(metrics['avg_losing_trade'])
    metrics['expectancy'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # =========================================================================
    # RISK METRICS
    # =========================================================================
    
    if not equity_df.empty:
        # Daily returns
        equity_df = equity_df.copy()
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        
        if len(daily_returns) > 0:
            # Volatility
            metrics['daily_volatility'] = daily_returns.std()
            metrics['annualized_volatility'] = daily_returns.std() * np.sqrt(252)
            
            # Sharpe Ratio (annualized)
            excess_returns = daily_returns - (risk_free_rate / 252)
            if metrics['daily_volatility'] > 0:
                metrics['sharpe_ratio'] = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0
            
            # Sortino Ratio (only downside deviation)
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                if downside_std > 0:
                    metrics['sortino_ratio'] = (excess_returns.mean() / downside_std) * np.sqrt(252)
                else:
                    metrics['sortino_ratio'] = 0
            else:
                metrics['sortino_ratio'] = float('inf')
        else:
            metrics['daily_volatility'] = 0
            metrics['annualized_volatility'] = 0
            metrics['sharpe_ratio'] = 0
            metrics['sortino_ratio'] = 0
        
        # Drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        
        metrics['max_drawdown_pct'] = abs(equity_df['drawdown'].min()) * 100
        
        # Max drawdown duration
        in_drawdown = equity_df['drawdown'] < 0
        drawdown_groups = (~in_drawdown).cumsum()
        drawdown_lengths = in_drawdown.groupby(drawdown_groups).sum()
        metrics['max_drawdown_duration'] = drawdown_lengths.max() if len(drawdown_lengths) > 0 else 0
        
        # Calmar Ratio
        if metrics['max_drawdown_pct'] > 0:
            metrics['calmar_ratio'] = metrics['cagr'] / metrics['max_drawdown_pct']
        else:
            metrics['calmar_ratio'] = 0
        
        # Recovery Factor
        if metrics['max_drawdown_pct'] > 0:
            metrics['recovery_factor'] = metrics['total_return_pct'] / metrics['max_drawdown_pct']
        else:
            metrics['recovery_factor'] = 0
    else:
        metrics['max_drawdown_pct'] = 0
        metrics['max_drawdown_duration'] = 0
        metrics['sharpe_ratio'] = 0
        metrics['sortino_ratio'] = 0
        metrics['calmar_ratio'] = 0
        metrics['recovery_factor'] = 0
        metrics['daily_volatility'] = 0
        metrics['annualized_volatility'] = 0
    
    # =========================================================================
    # TRADE DURATION
    # =========================================================================
    
    metrics['avg_holding_days'] = trades_df['holding_days'].mean()
    metrics['max_holding_days'] = trades_df['holding_days'].max()
    metrics['min_holding_days'] = trades_df['holding_days'].min()
    
    # =========================================================================
    # CONSECUTIVE WINS/LOSSES
    # =========================================================================
    
    trades_df = trades_df.copy()
    trades_df['is_win'] = trades_df['pnl'] > 0
    
    # Calculate consecutive wins/losses
    wins_streak = _max_consecutive(trades_df['is_win'], True)
    losses_streak = _max_consecutive(trades_df['is_win'], False)
    
    metrics['max_consecutive_wins'] = wins_streak
    metrics['max_consecutive_losses'] = losses_streak
    
    # =========================================================================
    # MONTHLY RETURNS
    # =========================================================================
    
    if not equity_df.empty:
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
        
        try:
            monthly_equity = equity_df['equity'].resample('ME').last()
            monthly_returns = monthly_equity.pct_change().dropna()
            
            metrics['positive_months'] = (monthly_returns > 0).sum()
            metrics['negative_months'] = (monthly_returns <= 0).sum()
            metrics['best_month_pct'] = monthly_returns.max() * 100 if len(monthly_returns) > 0 else 0
            metrics['worst_month_pct'] = monthly_returns.min() * 100 if len(monthly_returns) > 0 else 0
            metrics['avg_monthly_return'] = monthly_returns.mean() * 100 if len(monthly_returns) > 0 else 0
        except Exception:
            metrics['positive_months'] = 0
            metrics['negative_months'] = 0
            metrics['best_month_pct'] = 0
            metrics['worst_month_pct'] = 0
            metrics['avg_monthly_return'] = 0
    else:
        metrics['positive_months'] = 0
        metrics['negative_months'] = 0
        metrics['best_month_pct'] = 0
        metrics['worst_month_pct'] = 0
        metrics['avg_monthly_return'] = 0
    
    return metrics


def _max_consecutive(series: pd.Series, value: bool) -> int:
    """Calculate maximum consecutive occurrences of a value."""
    groups = (series != value).cumsum()
    value_groups = series.groupby(groups).apply(lambda x: len(x) if x.iloc[0] == value else 0)
    return value_groups.max() if len(value_groups) > 0 else 0


def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics dictionary."""
    return {
        'total_pnl': 0,
        'total_return_pct': 0,
        'cagr': 0,
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0,
        'avg_trade_pnl': 0,
        'avg_trade_pnl_pct': 0,
        'profit_factor': 0,
        'expectancy': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'max_drawdown_pct': 0,
        'calmar_ratio': 0,
        'recovery_factor': 0,
    }


def format_metrics_report(metrics: Dict[str, Any]) -> str:
    """
    Format metrics as a readable report.
    
    Args:
        metrics: Metrics dictionary.
    
    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 60,
        "PERFORMANCE REPORT",
        "=" * 60,
        "",
        "RETURNS",
        "-" * 40,
        f"  Total Return:        {metrics.get('total_return_pct', 0):.2f}%",
        f"  CAGR:                {metrics.get('cagr', 0):.2f}%",
        f"  Total P&L:           ${metrics.get('total_pnl', 0):.2f}",
        "",
        "TRADES",
        "-" * 40,
        f"  Total Trades:        {metrics.get('total_trades', 0)}",
        f"  Winning Trades:      {metrics.get('winning_trades', 0)}",
        f"  Losing Trades:       {metrics.get('losing_trades', 0)}",
        f"  Win Rate:            {metrics.get('win_rate', 0):.1f}%",
        f"  Avg Trade:           ${metrics.get('avg_trade_pnl', 0):.2f}",
        f"  Best Trade:          ${metrics.get('best_trade', 0):.2f}",
        f"  Worst Trade:         ${metrics.get('worst_trade', 0):.2f}",
        "",
        "RISK-ADJUSTED",
        "-" * 40,
        f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}",
        f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}",
        f"  Profit Factor:       {metrics.get('profit_factor', 0):.2f}",
        f"  Expectancy:          ${metrics.get('expectancy', 0):.2f}",
        "",
        "DRAWDOWN",
        "-" * 40,
        f"  Max Drawdown:        {metrics.get('max_drawdown_pct', 0):.2f}%",
        f"  Max DD Duration:     {metrics.get('max_drawdown_duration', 0)} days",
        f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.2f}",
        f"  Recovery Factor:     {metrics.get('recovery_factor', 0):.2f}",
        "",
        "CONSISTENCY",
        "-" * 40,
        f"  Positive Months:     {metrics.get('positive_months', 0)}",
        f"  Negative Months:     {metrics.get('negative_months', 0)}",
        f"  Avg Monthly Return:  {metrics.get('avg_monthly_return', 0):.2f}%",
        f"  Max Consecutive Wins:  {metrics.get('max_consecutive_wins', 0)}",
        f"  Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}",
        "",
        "=" * 60,
    ]
    
    return "\n".join(lines)


def compare_models(
    results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare metrics across multiple models.
    
    Args:
        results: Dict of model_name -> metrics dict.
    
    Returns:
        DataFrame comparing models.
    """
    comparison = []
    
    key_metrics = [
        'total_return_pct', 'win_rate', 'profit_factor', 
        'sharpe_ratio', 'max_drawdown_pct', 'total_trades'
    ]
    
    for model_name, metrics in results.items():
        row = {'model': model_name}
        for key in key_metrics:
            row[key] = metrics.get(key, 0)
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    df.set_index('model', inplace=True)
    
    return df
