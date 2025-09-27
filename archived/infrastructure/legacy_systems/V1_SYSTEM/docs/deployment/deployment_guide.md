
# GPT-Trader Deployment Guide

## Prerequisites
- Python 3.9+
- Poetry for dependency management
- Git for version control

## Installation Steps

1. Clone repository:
   ```bash
   git clone <repository_url>
   cd GPT-Trader
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Configure environment:
   ```bash
   cp .env.template .env
   # Edit .env with your settings
   ```

4. Verify installation:
   ```bash
   poetry run python phase_1e_production_validation.py
   ```

## Running the System

### Backtesting
```bash
poetry run gpt-trader backtest --symbol AAPL --start 2024-01-01 --end 2024-03-01
```

### Paper Trading
```bash
poetry run python demos/alpaca_paper_trading_demo.py
```

### Dashboard
```bash
poetry run gpt-trader dashboard
```
