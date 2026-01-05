"""
Visualization Functions for Forex Signal Model Backtesting.

Charts and reports for performance analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


def plot_equity_curve(
    equity_df: pd.DataFrame,
    trades_df: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
    title: str = "Equity Curve"
) -> plt.Figure:
    """
    Plot equity curve with trade markers and drawdown subplot.
    
    Args:
        equity_df: DataFrame with columns [date, equity].
        trades_df: Optional DataFrame with trade information.
        save_path: Path to save figure.
        title: Plot title.
    
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                              gridspec_kw={'height_ratios': [3, 1]},
                              sharex=True)
    
    # Ensure date column
    equity_df = equity_df.copy()
    if 'date' in equity_df.columns:
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        x = equity_df['date']
    else:
        x = equity_df.index
    
    # Plot equity curve
    axes[0].plot(x, equity_df['equity'], linewidth=2, color='#2E86AB', label='Equity')
    axes[0].fill_between(x, equity_df['equity'].iloc[0], equity_df['equity'], 
                         alpha=0.3, color='#2E86AB')
    
    # Add trade markers if provided
    if trades_df is not None and not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        
        # Win/loss markers
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        if not wins.empty:
            # Find equity values at trade dates
            for _, trade in wins.iterrows():
                try:
                    idx = equity_df[equity_df['date'] >= trade['entry_date']].index[0]
                    y_val = equity_df.loc[idx, 'equity']
                    axes[0].scatter(trade['entry_date'], y_val, 
                                   color='green', marker='^', s=50, alpha=0.7)
                except (IndexError, KeyError):
                    pass
        
        if not losses.empty:
            for _, trade in losses.iterrows():
                try:
                    idx = equity_df[equity_df['date'] >= trade['entry_date']].index[0]
                    y_val = equity_df.loc[idx, 'equity']
                    axes[0].scatter(trade['entry_date'], y_val, 
                                   color='red', marker='v', s=50, alpha=0.7)
                except (IndexError, KeyError):
                    pass
    
    axes[0].set_ylabel('Equity ($)', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Format y-axis as currency
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Calculate and plot drawdown
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax'] * 100
    
    axes[1].fill_between(x, 0, equity_df['drawdown'], 
                         color='#E74C3C', alpha=0.5)
    axes[1].plot(x, equity_df['drawdown'], color='#C0392B', linewidth=1)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Format x-axis dates
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_monthly_returns(
    equity_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot heatmap of monthly returns.
    
    Args:
        equity_df: DataFrame with columns [date, equity].
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    equity_df = equity_df.copy()
    
    if 'date' in equity_df.columns:
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)
    
    # Calculate monthly returns
    try:
        monthly_equity = equity_df['equity'].resample('ME').last()
        monthly_returns = monthly_equity.pct_change() * 100
    except Exception:
        monthly_equity = equity_df['equity'].resample('M').last()
        monthly_returns = monthly_equity.pct_change() * 100
    
    # Create pivot table
    monthly_returns = monthly_returns.dropna()
    if len(monthly_returns) == 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return fig
    
    returns_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    
    pivot = returns_df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6)))
    
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(pivot, cmap=cmap, center=0, annot=True, fmt='.1f',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Return (%)'})
    
    ax.set_title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_trade_analysis(
    trades_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot trade analysis: P&L distribution, holding periods, win/loss patterns.
    
    Args:
        trades_df: DataFrame of closed trades.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    if trades_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, 'No trades to analyze', ha='center', va='center')
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. P&L Distribution
    ax = axes[0, 0]
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] <= 0]['pnl']
    
    ax.hist(wins, bins=20, color='green', alpha=0.7, label=f'Wins ({len(wins)})')
    ax.hist(losses, bins=20, color='red', alpha=0.7, label=f'Losses ({len(losses)})')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('P&L ($)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Holding Period Distribution
    ax = axes[0, 1]
    ax.hist(trades_df['holding_days'], bins=range(0, int(trades_df['holding_days'].max()) + 2),
           color='#3498DB', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Holding Days', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Holding Period Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative P&L
    ax = axes[1, 0]
    cumulative_pnl = trades_df['pnl'].cumsum()
    ax.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='#2E86AB')
    ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                    where=(cumulative_pnl >= 0), alpha=0.3, color='green')
    ax.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl, 
                    where=(cumulative_pnl < 0), alpha=0.3, color='red')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Trade Number', fontsize=11)
    ax.set_ylabel('Cumulative P&L ($)', fontsize=11)
    ax.set_title('Cumulative P&L by Trade', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. P&L by Ticker
    ax = axes[1, 1]
    ticker_pnl = trades_df.groupby('ticker')['pnl'].sum()
    colors = ['green' if x > 0 else 'red' for x in ticker_pnl.values]
    bars = ax.bar(ticker_pnl.index, ticker_pnl.values, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Ticker', fontsize=11)
    ax.set_ylabel('Total P&L ($)', fontsize=11)
    ax.set_title('P&L by Ticker', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    save_path: Optional[str] = None,
    top_n: int = 20
) -> plt.Figure:
    """
    Plot top feature importances.
    
    Args:
        feature_importance: DataFrame with [feature, importance] columns.
        save_path: Path to save figure.
        top_n: Number of features to show.
    
    Returns:
        Matplotlib figure.
    """
    top_features = feature_importance.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    # Horizontal bar chart
    y_pos = np.arange(len(top_features))
    bars = ax.barh(y_pos, top_features['importance'].values, 
                  color='#3498DB', alpha=0.8, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_performance_report(
    metrics: Dict[str, Any],
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    save_path: str = 'results/backtest_report.html',
    model_name: str = "Trading Model"
) -> str:
    """
    Generate comprehensive HTML performance report.
    
    Args:
        metrics: Performance metrics dictionary.
        trades_df: DataFrame of closed trades.
        equity_df: DataFrame with equity curve.
        save_path: Path to save HTML report.
        model_name: Name of the model.
    
    Returns:
        Path to saved report.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate charts and save as files
    chart_dir = Path(save_path).parent / 'charts'
    chart_dir.mkdir(exist_ok=True)
    
    equity_chart = chart_dir / 'equity_curve.png'
    monthly_chart = chart_dir / 'monthly_returns.png'
    trade_chart = chart_dir / 'trade_analysis.png'
    
    plot_equity_curve(equity_df, trades_df, str(equity_chart))
    plot_monthly_returns(equity_df, str(monthly_chart))
    plot_trade_analysis(trades_df, str(trade_chart))
    plt.close('all')
    
    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{model_name} - Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; border-radius: 8px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #3498db; color: white; }}
            tr:hover {{ background: #f5f5f5; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 {model_name} - Performance Report</h1>
            <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value {'positive' if metrics.get('total_return_pct', 0) > 0 else 'negative'}">{metrics.get('total_return_pct', 0):.2f}%</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('win_rate', 0):.1f}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value negative">{metrics.get('max_drawdown_pct', 0):.2f}%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('total_trades', 0)}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
            </div>
            
            <h2>Equity Curve</h2>
            <img src="charts/equity_curve.png" alt="Equity Curve">
            
            <h2>Monthly Returns</h2>
            <img src="charts/monthly_returns.png" alt="Monthly Returns">
            
            <h2>Trade Analysis</h2>
            <img src="charts/trade_analysis.png" alt="Trade Analysis">
            
            <h2>Detailed Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total P&L</td><td>${metrics.get('total_pnl', 0):,.2f}</td></tr>
                <tr><td>CAGR</td><td>{metrics.get('cagr', 0):.2f}%</td></tr>
                <tr><td>Winning Trades</td><td>{metrics.get('winning_trades', 0)}</td></tr>
                <tr><td>Losing Trades</td><td>{metrics.get('losing_trades', 0)}</td></tr>
                <tr><td>Avg Trade P&L</td><td>${metrics.get('avg_trade_pnl', 0):.2f}</td></tr>
                <tr><td>Best Trade</td><td>${metrics.get('best_trade', 0):.2f}</td></tr>
                <tr><td>Worst Trade</td><td>${metrics.get('worst_trade', 0):.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{metrics.get('sortino_ratio', 0):.2f}</td></tr>
                <tr><td>Calmar Ratio</td><td>{metrics.get('calmar_ratio', 0):.2f}</td></tr>
                <tr><td>Expectancy</td><td>${metrics.get('expectancy', 0):.2f}</td></tr>
                <tr><td>Avg Holding Days</td><td>{metrics.get('avg_holding_days', 0):.1f}</td></tr>
                <tr><td>Max Consecutive Wins</td><td>{metrics.get('max_consecutive_wins', 0)}</td></tr>
                <tr><td>Max Consecutive Losses</td><td>{metrics.get('max_consecutive_losses', 0)}</td></tr>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(save_path, 'w') as f:
        f.write(html)
    
    return save_path
