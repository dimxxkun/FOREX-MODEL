"""
Backtest Simulator for Forex Model Optimization.

Runs thousands of simulations (grid search/Optuna) to find optimal 
parameters for a strict 20-pip SL strategy.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import itertools
from tqdm import tqdm
import json
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import BacktestEngine
from src.utils import load_config, get_logger

logger = get_logger('backtest_simulator')

def run_simulation(
    confidence_threshold: float,
    tp_multiple: float,
    max_stop_pips: float,
    allowed_regimes: list,
    features_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    config_path: str = 'config/config.yaml'
):
    """Run a single backtest simulation with specific parameters."""
    # Load config and update with simulation params
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['signal_filter']['confidence_threshold'] = confidence_threshold
    config['signal_filter']['allowed_regimes'] = allowed_regimes
    config['backtest']['tp_multiple'] = tp_multiple # Custom parameter for TP
    config['backtest']['max_stop_pips'] = max_stop_pips

    # Save temp config for engine
    temp_config_path = PROJECT_ROOT / 'config' / 'temp_sim_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    try:
        engine = BacktestEngine(str(temp_config_path))
        
        # Adjust signals based on confidence threshold
        filtered_signals = signals_df[signals_df['Confidence'] >= confidence_threshold * 100].copy()
        
        if filtered_signals.empty:
            return None

        # Run backtest
        results = engine.run_backtest(filtered_signals, features_df)
        
        # Cleanup
        if temp_config_path.exists():
            os.remove(temp_config_path)
            
        return {
            'confidence': confidence_threshold,
            'tp_multiple': tp_multiple,
            'max_stop_pips': max_stop_pips,
            'sharpe': results.get('total_return_pct', 0) / max(results.get('max_drawdown_pct', 1), 1), # Simplified Sharpe
            'win_rate': results.get('win_rate', 0),
            'total_trades': results.get('total_trades', 0),
            'total_return': results.get('total_return_pct', 0),
            'max_drawdown': results.get('max_drawdown_pct', 0)
        }
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return None

def main():
    # Load data
    features_path = PROJECT_ROOT / 'data' / 'processed' / 'features.parquet'
    if not features_path.exists():
        print("Features not found. Please run data pipeline first.")
        return

    df = pd.read_parquet(features_path)
    
    # Load Real Model
    model_path = PROJECT_ROOT / 'results' / 'walk_forward' / 'best_wf_model.pkl'
    if not model_path.exists():
        print(f"Model not found at {model_path}. Please train model first.")
        return
    
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loaded model from {model_path}")

    # Prepare features (align with model)
    feature_cols = [c for c in df.columns if 'Target' not in c and c not in ['Date', 'ticker']]
    X = df[feature_cols].fillna(0)
    
    # Get model feature names if available
    try:
        model_features = model.get_booster().feature_names
        for feat in model_features:
            if feat not in X.columns:
                X[feat] = 0
        X = X[model_features]
    except:
        pass

    # Generate Real Model Predictions
    print("Generating model predictions...")
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    
    signals = []
    for i, (date, row) in enumerate(df.iterrows()):
        prob = probs[i]
        signal_val = 1 if prob > 0.5 else -1
        conf = max(prob, 1 - prob) * 100
        
        signals.append({
            'Date': date,
            'Ticker': 'GBPUSD',
            'Signal': signal_val,
            'Confidence': conf
        })
    
    signals_df = pd.DataFrame(signals)

    # Parameter grid
    confidences = [0.55, 0.60, 0.65, 0.70]
    tp_multiples = [1.5, 2.0, 2.5]
    stop_pips = [10, 15, 20]
    
    permutations = list(itertools.product(confidences, tp_multiples, stop_pips))
    print(f"Starting {len(permutations)} simulations...")
    
    results_list = []
    for params in tqdm(permutations):
        res = run_simulation(
            confidence_threshold=params[0],
            tp_multiple=params[1],
            max_stop_pips=params[2],
            allowed_regimes=['trending', 'normal', 'low_volatility'],
            features_df=df,
            signals_df=signals_df
        )
        if res:
            results_list.append(res)
    
    # Analyze results
    res_df = pd.DataFrame(results_list)
    res_df = res_df.sort_values(by='sharpe', ascending=False)
    
    print("\n🏆 Top 5 Optimal Configurations:")
    print(res_df.head(5))
    
    # Save results
    output_path = PROJECT_ROOT / 'results' / 'optimization' / 'sim_results.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
