# 🚀 GPT-Trader Quick Start Guide

Get up and running with GPT-Trader in 5 minutes!

## 📋 Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- 4GB+ RAM
- Internet connection for market data

## 🎯 Quick Install

```bash
# Clone the repository
git clone https://github.com/your-org/gpt-trader.git
cd gpt-trader

# Install dependencies
poetry install

# Set up environment (choose one):
# Option 1: Demo mode (no API keys needed)
echo "DEMO_MODE=true" > .env.local

# Option 2: Real API keys (for paper/live trading)
cp .env.template .env.local
# Edit .env.local with your Alpaca API credentials
```

## 🏃 Your First Backtest (< 1 minute)

Run a simple moving average strategy on Apple stock:

```bash
# Using demo mode (no API keys required)
DEMO_MODE=true poetry run python -m bot.cli backtest \
  --symbol AAPL \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --strategy demo_ma
```

You should see output like:
```
╭───────────────────────────── Starting Backtest ──────────────────────────────╮
│ Backtest Configuration                                                       │
│ Strategy: demo_ma                                                            │
│ Period: 2024-01-01 to 2024-06-30                                             │
│ Risk per trade: 0.5%                                                         │
│ Max positions: 10                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯

Backtesting: 100%|██████████| 123/123 [00:01<00:00, 92.31it/s]

    Backtest Results
┏━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric       ┃ Value ┃
┡━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Return │ 1.18% │
│ CAGR         │ 2.41% │
│ Sharpe Ratio │ 1.206 │
│ Max Drawdown │ 0.73% │
│ Volatility   │ 1.99% │
│ Total Costs  │ $0.00 │
└──────────────┴───────┘
```

## 📊 Common Use Cases

### 1. Test Multiple Stocks

Create a file `universe.csv`:
```csv
symbol
AAPL
MSFT
GOOGL
AMZN
TSLA
```

Run backtest on all stocks:
```bash
poetry run python -m bot.cli backtest \
  --symbol-list universe.csv \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --strategy trend_breakout
```

### 2. Optimize Strategy Parameters

Find the best parameters for your strategy:
```bash
poetry run python -m bot.cli optimize \
  --symbol SPY \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --strategy demo_ma \
  --param window 10 50 5  # Test windows from 10 to 50, step 5
```

### 3. Test with Risk Management

Add position sizing and risk limits:
```bash
poetry run python -m bot.cli backtest \
  --symbol SPY \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --strategy trend_breakout \
  --risk-pct 1.0 \         # Risk 1% per trade
  --max-positions 5 \      # Maximum 5 positions
  --cost-bps 10            # 10 basis points transaction cost
```

### 4. Use Regime Filters

Only trade when market conditions are favorable:
```bash
poetry run python -m bot.cli backtest \
  --symbol AAPL \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --strategy trend_breakout \
  --regime on \            # Enable regime filter
  --regime-symbol SPY \    # Use SPY as market proxy
  --regime-window 200      # 200-day moving average
```

## 🎮 Demo Mode vs Real Mode

### Demo Mode (Testing & Learning)
```bash
DEMO_MODE=true poetry run python -m bot.cli <command>
```
- ✅ Backtesting with historical data
- ✅ Strategy optimization
- ✅ Walk-forward analysis
- ❌ Paper trading (requires API)
- ❌ Live trading (requires API)

### Real Mode (Full Features)
```bash
# Set up API keys in .env.local first
poetry run python -m bot.cli <command>
```
- ✅ All demo features
- ✅ Paper trading with real-time data
- ✅ Live trading (use with caution!)

## 📈 Understanding Results

### Key Metrics Explained

| Metric | What it Means | Good Value |
|--------|--------------|------------|
| **Total Return** | Overall profit/loss | > 0% |
| **CAGR** | Annualized return | > 10% |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 |
| **Max Drawdown** | Largest peak-to-trough loss | < 20% |
| **Volatility** | Price variation | < 20% |
| **Win Rate** | % of profitable trades | > 50% |

### Output Files

After each backtest, find your results in `data/backtests/`:
- `PORT_*.csv` - Daily portfolio values
- `*_summary.csv` - Performance metrics
- `*_trades.csv` - Individual trade records
- `*.png` - Performance chart

## 🛠️ Troubleshooting

### "No data available"
```bash
# Solution: Check date range and symbol
poetry run python -m bot.cli backtest \
  --symbol SPY \  # Use major index for testing
  --start 2023-01-01 \  # Use recent dates
  --end 2023-12-31
```

### "API credentials not found"
```bash
# Solution 1: Use demo mode
DEMO_MODE=true poetry run python -m bot.cli backtest ...

# Solution 2: Set up real credentials
# Sign up at https://alpaca.markets (free)
# Add to .env.local:
# ALPACA_API_KEY_ID=your_key_here
# ALPACA_API_SECRET_KEY=your_secret_here
```

### "Strategy not found"
```bash
# List available strategies:
poetry run python -m bot.cli backtest --help

# Current strategies:
# - demo_ma (Simple Moving Average)
# - trend_breakout (Donchian Channel Breakout)
```

## 🎯 Next Steps

1. **Explore Strategies**: Try different built-in strategies
2. **Adjust Parameters**: Experiment with risk and position settings
3. **Multiple Timeframes**: Test strategies across different periods
4. **Custom Strategies**: Learn to create your own strategies
5. **Paper Trade**: Test strategies with real-time data (requires API)

## 📚 Learn More

- [Full CLI Reference](CLI_REFERENCE.md)
- [Strategy Development Guide](STRATEGY_GUIDE.md)
- [Risk Management](RISK_MANAGEMENT.md)
- [API Setup Guide](API_SETUP.md)

## 💡 Pro Tips

1. **Start Small**: Test with 1-2 stocks before large universes
2. **Recent Data**: Use recent 1-2 years for more relevant results
3. **Transaction Costs**: Always include `--cost-bps` for realistic results
4. **Compare Strategies**: Run same period with different strategies
5. **Save Results**: Use `--run-tag` to organize experiments

## 🤝 Getting Help

- Run `gpt-trader --help` for command options
- Run `gpt-trader <command> --help` for specific command help
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Join our Discord community
- Open an issue on GitHub

---

**Ready to dive deeper?** Check out the [Complete Documentation](README.md)
