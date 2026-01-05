# Forex Signal Model

A production-ready forex and gold trading signal system for generating daily BUY/SELL/HOLD signals with confidence scores and dynamic stop losses.

## Overview

This project implements a machine learning-based trading signal model for:
- **GBP/USD** (British Pound / US Dollar)
- **EUR/USD** (Euro / US Dollar)  
- **XAU/USD** (Gold Spot / US Dollar)

### Philosophy
- **Conservative Approach**: Capital preservation focused
- **Risk Management**: Maximum 1% risk per trade, <20% drawdown limit
- **Transparency**: All decisions logged and explainable

## Features

### Data Pipeline
- Automated data download from yfinance
- Multi-ticker support with retry logic
- Data cleaning and alignment
- Quality validation and reporting

### Feature Engineering
- **Tier 1 Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, ADX
- **Multitimeframe Features**: Weekly trend alignment, monthly returns
- **Tier 2 Intermarket Features**: DXY correlation, VIX regime, Treasury yields, Oil
- **Derived Features**: Z-score, volatility percentile, trend strength

### Safeguards
- Look-ahead bias prevention
- OHLC relationship validation
- Reproducibility (versioned outputs)

## Installation

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd forex_signal_model
```

2. Create virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Full Pipeline
Run the complete data download and feature engineering pipeline:
```bash
python main.py --mode all
```

### Download Data Only
```bash
python main.py --mode download
```

### Generate Features Only
(Requires data to be downloaded first)
```bash
python main.py --mode features
```

### Update with Latest Data
```bash
python main.py --mode update
```

### Options
```bash
python main.py --help

Options:
  --mode {download,features,all,update}
  --config PATH       Path to config file (default: config/config.yaml)
  --verbose, -v       Enable debug logging
  --dry-run           Show what would be done without executing
```

## Project Structure

```
forex_signal_model/
├── config/
│   └── config.yaml          # All configuration parameters
├── data/
│   ├── raw/                  # Individual ticker CSVs
│   └── processed/            # Combined parquet files
├── src/
│   ├── __init__.py
│   ├── utils.py              # Utility functions
│   ├── data_pipeline.py      # Data acquisition and cleaning
│   └── feature_engineering.py # Feature creation
├── notebooks/
│   └── 01_data_exploration.ipynb  # EDA notebook
├── logs/                     # Log files
├── results/                  # Reports and outputs
├── tests/                    # Unit tests
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
└── README.md
```

## Configuration

All parameters are configured in `config/config.yaml`:

```yaml
data:
  tickers:
    main: [GBPUSD=X, EURUSD=X, XAUUSD=X]
    intermarket: [DX-Y.NYB, ^VIX, ^TNX, CL=F]
  period: '10y'
  interval: '1d'

features:
  technical:
    sma_periods: [20, 50, 200]
    rsi_period: 14
    macd: [12, 26, 9]
    # ... more settings

risk:
  max_risk_per_trade: 0.01  # 1%
  max_drawdown_pct: 20.0
```

## Testing

Run unit tests:
```bash
python -m pytest tests/ -v
```

## Development Roadmap

### Phase 1 (Current) - Foundation
- [x] Data pipeline
- [x] Feature engineering
- [x] EDA notebook
- [x] Configuration management

### Phase 2 - Model Development
- [ ] Baseline models (Logistic Regression, Random Forest)
- [ ] Advanced models (XGBoost, LightGBM)
- [ ] Walk-forward validation
- [ ] Hyperparameter tuning

### Phase 3 - Signal Generation
- [ ] Confidence scoring
- [ ] Dynamic stop loss calculation
- [ ] Position sizing
- [ ] Risk management

### Phase 4 - Production
- [ ] Telegram notifications
- [ ] Paper trading integration
- [ ] Performance monitoring
- [ ] Automated retraining

## Output Files

After running the pipeline:

| File | Description |
|------|-------------|
| `data/raw/*.csv` | Individual ticker data |
| `data/raw/metadata.json` | Download metadata |
| `data/processed/combined_data.parquet` | Unified price data |
| `data/processed/features.parquet` | ML-ready features |
| `data/processed/feature_metadata.json` | Feature descriptions |
| `logs/forex_model.log` | Execution logs |
| `results/pipeline_report.txt` | Summary report |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading forex and commodities involves substantial risk of loss. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.
