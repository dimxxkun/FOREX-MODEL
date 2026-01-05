"""Script to update notebook cells for Sharpe optimization with new config."""
import json

# Read notebook
with open(r'notebooks\04_sharpe_optimization.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Find and update the signal filter initialization cell
signal_filter_cell = '''# Initialize signal filter with updated config
import yaml
from pathlib import Path

# Load updated config
config_path = Path('../config/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract filter config from YAML
filter_config = config.get('signal_filter', {
    'confidence_threshold': 0.50,
    'allowed_regimes': ['strong_trend_up', 'strong_trend_down', 'weak_trend', 'low_volatility', 'ranging'],
    'min_trade_gap': 1,
    'min_risk_reward': 1.5
})

# Create filter with config
signal_filter = SignalFilter({'signal_filter': filter_config})
print(f"✅ Signal Filter initialized:")
print(f"   Confidence threshold: {signal_filter.confidence_threshold}")
print(f"   Allowed regimes: {signal_filter.allowed_regimes}")
print(f"   Min trade gap: {signal_filter.min_trade_gap} days")'''

# Find and update the backtest cell to use volatility scaling
backtest_cell = '''# Run backtest with volatility-scaled position sizing
def backtest_with_volatility_sizing(
    df,
    signals,
    close_col,
    atr_col,
    initial_capital=100000,
    transaction_cost=0.0003
):
    """Backtest with volatility-scaled position sizing for better Sharpe."""
    capital = initial_capital
    peak_capital = initial_capital
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    trades = []
    equity_curve = [initial_capital]
    
    # Get ATR for volatility scaling
    atr_series = df[atr_col] if atr_col in df.columns else None
    avg_atr = atr_series.mean() if atr_series is not None else None
    
    close = df[close_col]
    
    for i in range(1, len(df)):
        current_price = close.iloc[i]
        signal = signals.iloc[i] if i < len(signals) else 0
        current_atr = atr_series.iloc[i] if atr_series is not None else 0
        
        # Check exits first
        if position != 0:
            # Check stop loss / take profit
            if position == 1:  # Long
                if current_price <= stop_loss or current_price >= take_profit:
                    exit_pnl = (current_price - entry_price) * abs(position_size)
                    capital += exit_pnl - (abs(position_size) * current_price * transaction_cost)
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': exit_pnl,
                        'position': position,
                        'size': position_size
                    })
                    position = 0
            else:  # Short
                if current_price >= stop_loss or current_price <= take_profit:
                    exit_pnl = (entry_price - current_price) * abs(position_size)
                    capital += exit_pnl - (abs(position_size) * current_price * transaction_cost)
                    trades.append({
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'pnl': exit_pnl,
                        'position': position,
                        'size': position_size
                    })
                    position = 0
        
        # Check entries
        if position == 0 and signal in [0, 1]:
            # Calculate volatility-scaled position size
            if avg_atr and current_atr > 0:
                vol_ratio = current_atr / avg_atr
                vol_scalar = max(0.5, min(2.0, 1.0 / vol_ratio))
            else:
                vol_scalar = 1.0
            
            base_risk = 0.01  # 1% base risk
            adjusted_risk = base_risk * vol_scalar
            
            # Set stops based on ATR
            atr_mult = 2.0
            stop_dist = current_atr * atr_mult if current_atr else current_price * 0.01
            
            if signal == 1:  # Long
                position = 1
                entry_price = current_price
                stop_loss = entry_price - stop_dist
                take_profit = entry_price + (stop_dist * 1.5)
            else:  # Short
                position = -1
                entry_price = current_price
                stop_loss = entry_price + stop_dist
                take_profit = entry_price - (stop_dist * 1.5)
            
            # Position size with volatility scaling
            risk_amount = capital * adjusted_risk
            position_size = risk_amount / stop_dist if stop_dist > 0 else 0
            
            # Apply constraints
            max_size = (capital * 0.20) / current_price
            position_size = min(position_size, max_size)
        
        # Track equity
        if position != 0:
            unrealized = (current_price - entry_price) * position_size * position
            equity_curve.append(capital + unrealized)
        else:
            equity_curve.append(capital)
        
        peak_capital = max(peak_capital, equity_curve[-1])
    
    # Calculate metrics
    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]
    
    total_return = (equity[-1] - initial_capital) / initial_capital
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_dd = np.min(equity / np.maximum.accumulate(equity) - 1)
    
    wins = [t for t in trades if t['pnl'] > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'num_trades': len(trades),
        'win_rate': win_rate,
        'equity_curve': equity,
        'trades': trades
    }

# Run the backtest
atr_cols = [c for c in test_df.columns if 'ATR' in c and 'Pct' not in c]
atr_col = atr_cols[0] if atr_cols else None

close_cols = [c for c in test_df.columns if 'Close' in c]
close_col = close_cols[0] if close_cols else None

if 'filtered_signal' in test_df.columns:
    signals_to_use = test_df['filtered_signal']
else:
    signals_to_use = test_df['raw_signal']

results = backtest_with_volatility_sizing(
    test_df,
    signals_to_use,
    close_col,
    atr_col if atr_col else close_col,
    initial_capital=100000
)

print("\\n📊 Backtest Results with Volatility-Scaled Position Sizing:")
print(f"   Total Return: {results['total_return']*100:.2f}%")
print(f"   Sharpe Ratio: {results['sharpe']:.2f}")
print(f"   Max Drawdown: {results['max_drawdown']*100:.2f}%")
print(f"   Win Rate: {results['win_rate']:.1f}%")
print(f"   Total Trades: {results['num_trades']}")

# Check targets
print("\\n🎯 Target Verification:")
print(f"   Win Rate > 55%: {'✅' if results['win_rate'] > 55 else '❌'} ({results['win_rate']:.1f}%)")
print(f"   Sharpe > 0.5: {'✅' if results['sharpe'] > 0.5 else '❌'} ({results['sharpe']:.2f})")'''

# Track fixes
fixes = []

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Find signal filter init cell
        if 'SignalFilter' in source and 'filter_config' in source.lower():
            nb['cells'][i]['source'] = signal_filter_cell
            fixes.append(f"Cell {i}: Updated signal filter initialization")
        
        # Find backtest cell - look for the main backtest function
        if 'def backtest' in source and 'equity_curve' in source:
            nb['cells'][i]['source'] = backtest_cell
            fixes.append(f"Cell {i}: Updated backtest with volatility sizing")

print(f"Made {len(fixes)} fixes:")
for fix in fixes:
    print(f"  - {fix}")

# Save notebook
with open(r'notebooks\04_sharpe_optimization.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook saved successfully!')
