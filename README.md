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

Free noncommercial use under [LICENSE](LICENSE). Commercial use requires a separate written agreement and an agreed fee. Trading involves risk.

## Commercial permission

Original material in this repository is available under the [Noncommercial Use and Commercial Permission License](LICENSE). **Noncommercial use is free. Commercial use requires prior written permission and a separate written agreement specifying the scope and agreed fee.** Contact [work.abdulfatai@gmail.com](mailto:work.abdulfatai@gmail.com) with the repository name and intended use.

Portfolio review and recruitment evaluation are permitted without a fee. Third-party components retain their own licenses. Permissions already granted for earlier versions remain valid; the new terms do not revoke those grants. These are custom source-available terms, not an OSI-approved open-source license.
