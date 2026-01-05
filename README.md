# Forex Signal Model

A production-ready ML trading signal system for GBPUSD, EURUSD, and Gold (XAU/USD).

## Quick Start

### Generate Daily Signals
**Option 1:** Double-click `run_signals.bat`

**Option 2:** Command line
```bash
python signal_generator.py --account 10000
```

Signals are saved to `signals/signals_YYYY-MM-DD.json`

---

## Project Structure

```
forex_signal_model/
├── config/              # Configuration files
│   └── config.yaml
├── data/                # Market data
│   ├── raw/             # Downloaded ticker data
│   └── processed/       # Feature-engineered data
├── models/              # Trained ML models
├── notebooks/           # Development notebooks
├── results/             # Backtest & optimization results
├── signals/             # Generated trading signals
├── src/                 # Core source code
│   ├── data_pipeline.py
│   ├── feature_engineering.py
│   ├── regime_detector.py
│   ├── risk_management.py
│   ├── signal_filter.py
│   └── models/
├── signal_generator.py  # Daily signal generation
├── paper_trading_tracker.py  # Trade tracking
└── main.py              # Data pipeline entry
```

---

## Performance (Walk-Forward Validation)

| Metric | Value | Target |
|--------|-------|--------|
| **Sharpe Ratio** | 0.60 | >0.5 ✅ |
| **Win Rate** | 58.3% | >55% ✅ |
| **Max Drawdown** | 11.5% | <20% ✅ |

---

## Daily Workflow

1. **6:00 AM (Nigeria)** - Run `run_signals.bat`
2. **8:00 AM** - Review signals, enter trades at London open
3. **1:00-5:00 PM** - Monitor during London-NYSE overlap

---

## Configuration

Edit `config/config.yaml` to adjust:
- Confidence threshold (default: 50%)
- Risk per trade (default: 1%)
- Stop loss multiplier
- Allowed market regimes

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: pandas, numpy, xgboost, yfinance, pandas-ta

---

## License

MIT License - Educational purposes only. Trading involves risk.
