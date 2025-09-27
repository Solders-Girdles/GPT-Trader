# 📖 GPT-Trader CLI Command Reference

Complete reference for all GPT-Trader CLI commands with examples.

## Global Options

These options can be used with any command:

```bash
gpt-trader [GLOBAL_OPTIONS] <command> [COMMAND_OPTIONS]
```

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--help` | `-h` | Show help message | `gpt-trader --help` |
| `--version` | | Show version | `gpt-trader --version` |
| `--verbose` | `-v` | Increase verbosity (-vv for DEBUG) | `gpt-trader -vv backtest ...` |
| `--quiet` | `-q` | Suppress non-essential output | `gpt-trader -q backtest ...` |
| `--profile` | | Load configuration profile | `gpt-trader --profile aggressive ...` |
| `--data-strict` | | Data validation mode (strict/repair) | `gpt-trader --data-strict repair ...` |
| `--no-color` | | Disable colored output | `gpt-trader --no-color ...` |

## Commands

### 📊 backtest

Run historical backtests on trading strategies.

```bash
gpt-trader backtest [OPTIONS]
```

#### Required Options

| Option | Description | Example |
|--------|-------------|---------|
| `--start` | Start date (YYYY-MM-DD) | `--start 2024-01-01` |
| `--end` | End date (YYYY-MM-DD) | `--end 2024-12-31` |

#### Data Selection

| Option | Description | Example |
|--------|-------------|---------|
| `--symbol` | Single symbol to test | `--symbol AAPL` |
| `--symbol-list` | CSV file with symbols | `--symbol-list universe.csv` |

#### Strategy Configuration

| Option | Default | Description | Example |
|--------|---------|-------------|---------|
| `--strategy` | trend_breakout | Strategy to use | `--strategy demo_ma` |
| `--donchian` | 55 | Donchian channel period | `--donchian 20` |
| `--atr` | 20 | ATR period | `--atr 14` |
| `--atr-k` | 2.0 | ATR multiplier for stops | `--atr-k 3.0` |

#### Risk Management

| Option | Default | Description | Example |
|--------|---------|-------------|---------|
| `--risk-pct` | 0.5 | Risk per trade (%) | `--risk-pct 1.0` |
| `--max-positions` | 10 | Max concurrent positions | `--max-positions 5` |
| `--cost-bps` | 0 | Transaction costs (bps) | `--cost-bps 10` |

#### Regime Filter

| Option | Default | Description | Example |
|--------|---------|-------------|---------|
| `--regime` | off | Enable regime filter | `--regime on` |
| `--regime-symbol` | SPY | Symbol for regime | `--regime-symbol QQQ` |
| `--regime-window` | 200 | MA window for regime | `--regime-window 50` |

#### Examples

```bash
# Simple backtest
gpt-trader backtest \
  --symbol AAPL \
  --start 2024-01-01 \
  --end 2024-06-30

# Multiple symbols with risk management
gpt-trader backtest \
  --symbol-list universe.csv \
  --start 2024-01-01 \
  --end 2024-06-30 \
  --risk-pct 1.0 \
  --max-positions 5 \
  --cost-bps 10

# Trend following with regime filter
gpt-trader backtest \
  --symbol TSLA \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --strategy trend_breakout \
  --donchian 20 \
  --regime on \
  --regime-window 200

# Debug mode with verbose output
gpt-trader -vv backtest \
  --symbol SPY \
  --start 2024-01-01 \
  --end 2024-03-31 \
  --debug
```

### 🔧 optimize

Optimize strategy parameters using historical data.

```bash
gpt-trader optimize [OPTIONS]
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--symbol` | Symbol to optimize | `--symbol SPY` |
| `--start` | Start date | `--start 2023-01-01` |
| `--end` | End date | `--end 2023-12-31` |
| `--strategy` | Strategy to optimize | `--strategy demo_ma` |
| `--param` | Parameter range (name min max step) | `--param window 10 50 5` |
| `--metric` | Optimization metric | `--metric sharpe` |
| `--method` | Optimization method | `--method grid` |

#### Examples

```bash
# Optimize MA window
gpt-trader optimize \
  --symbol AAPL \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --strategy demo_ma \
  --param window 10 50 5

# Multi-parameter optimization
gpt-trader optimize \
  --symbol SPY \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --strategy trend_breakout \
  --param donchian 10 100 10 \
  --param atr 10 30 5 \
  --metric sharpe

# Evolutionary optimization
gpt-trader optimize \
  --symbol QQQ \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --strategy demo_ma \
  --method evolution \
  --generations 50 \
  --population 100
```

### 📈 walk-forward

Perform walk-forward analysis for robust strategy validation.

```bash
gpt-trader walk-forward [OPTIONS]
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--symbol` | Symbol to test | `--symbol SPY` |
| `--start` | Start date | `--start 2022-01-01` |
| `--end` | End date | `--end 2023-12-31` |
| `--in-sample` | In-sample period (months) | `--in-sample 12` |
| `--out-sample` | Out-sample period (months) | `--out-sample 3` |
| `--strategy` | Strategy to test | `--strategy trend_breakout` |

#### Examples

```bash
# Basic walk-forward
gpt-trader walk-forward \
  --symbol SPY \
  --start 2022-01-01 \
  --end 2023-12-31 \
  --in-sample 12 \
  --out-sample 3 \
  --strategy demo_ma

# With optimization
gpt-trader walk-forward \
  --symbol AAPL \
  --start 2022-01-01 \
  --end 2023-12-31 \
  --in-sample 6 \
  --out-sample 2 \
  --strategy trend_breakout \
  --optimize
```

### 📝 paper

Run paper trading with simulated execution.

```bash
gpt-trader paper [OPTIONS]
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--symbol` | Symbol(s) to trade | `--symbol AAPL,MSFT,GOOGL` |
| `--strategy` | Strategy to use | `--strategy trend_breakout` |
| `--capital` | Starting capital | `--capital 100000` |
| `--risk-pct` | Risk per trade | `--risk-pct 1.0` |

#### Examples

```bash
# Single symbol paper trading
gpt-trader paper \
  --symbol AAPL \
  --strategy demo_ma \
  --capital 100000

# Portfolio paper trading
gpt-trader paper \
  --symbol AAPL,MSFT,GOOGL,AMZN \
  --strategy trend_breakout \
  --capital 250000 \
  --risk-pct 2.0

# With monitoring
gpt-trader paper \
  --symbol SPY \
  --strategy demo_ma \
  --monitor \
  --alert-email your@email.com
```

### 🚀 live

Run live trading (use with extreme caution!).

```bash
gpt-trader live [OPTIONS]
```

⚠️ **WARNING**: Live trading uses real money. Test thoroughly with paper trading first!

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--symbol` | Symbol(s) to trade | `--symbol SPY` |
| `--strategy` | Strategy to use | `--strategy trend_breakout` |
| `--capital` | Capital to allocate | `--capital 10000` |
| `--risk-pct` | Risk per trade | `--risk-pct 0.5` |
| `--confirm` | Confirm live trading | `--confirm` |

#### Examples

```bash
# Live trading (requires confirmation)
gpt-trader live \
  --symbol SPY \
  --strategy demo_ma \
  --capital 10000 \
  --risk-pct 0.5 \
  --confirm

# With safety limits
gpt-trader live \
  --symbol QQQ \
  --strategy trend_breakout \
  --capital 25000 \
  --risk-pct 1.0 \
  --max-daily-loss 500 \
  --max-positions 3 \
  --confirm
```

### 📊 monitor

Monitor running strategies and positions.

```bash
gpt-trader monitor [OPTIONS]
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--refresh` | Refresh interval (seconds) | `--refresh 5` |
| `--metrics` | Metrics to display | `--metrics pnl,sharpe,drawdown` |
| `--export` | Export to file | `--export metrics.csv` |

#### Examples

```bash
# Basic monitoring
gpt-trader monitor

# Real-time dashboard
gpt-trader monitor \
  --refresh 5 \
  --metrics all

# Export metrics
gpt-trader monitor \
  --export daily_metrics.csv \
  --format csv
```

### 🎯 menu

Launch interactive menu system.

```bash
gpt-trader menu
```

Provides an interactive interface for:
- Running backtests
- Viewing results
- Managing strategies
- Configuration

### 📋 dashboard

Launch the trading dashboard.

```bash
gpt-trader dashboard [OPTIONS]
```

#### Options

| Option | Description | Example |
|--------|-------------|---------|
| `--port` | Dashboard port | `--port 8080` |
| `--host` | Dashboard host | `--host 0.0.0.0` |

#### Examples

```bash
# Launch dashboard
gpt-trader dashboard

# Custom port
gpt-trader dashboard --port 8888

# Allow external access
gpt-trader dashboard --host 0.0.0.0 --port 8080
```

### 🧙 wizard

Run the setup wizard for guided configuration.

```bash
gpt-trader wizard
```

Helps with:
- API credential setup
- Strategy selection
- Risk parameter configuration
- Data source configuration

### ⚡ shortcuts

Display available command shortcuts.

```bash
gpt-trader shortcuts
```

Common shortcuts:
- `bt` → `backtest`
- `opt` → `optimize`
- `wf` → `walk-forward`
- `dash` → `dashboard`

## Environment Variables

Control behavior via environment variables:

```bash
# Demo mode (no API keys required)
export DEMO_MODE=true

# Logging level
export LOG_LEVEL=DEBUG

# API credentials
export ALPACA_API_KEY_ID=your_key
export ALPACA_API_SECRET_KEY=your_secret

# Data directory
export DATA_DIR=/path/to/data

# Parallel processing
export NUM_WORKERS=4
```

## Configuration Files

### .env.local

```env
# API Configuration
ALPACA_API_KEY_ID=your_key_here
ALPACA_API_SECRET_KEY=your_secret_here
ALPACA_PAPER_BASE_URL=https://paper-api.alpaca.markets

# Settings
LOG_LEVEL=INFO
DEMO_MODE=false
NUM_WORKERS=4
```

### profiles/aggressive.yaml

```yaml
# ~/.gpt-trader/profiles/aggressive.yaml
risk_pct: 2.0
max_positions: 20
strategy: trend_breakout
regime: off
cost_bps: 5
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Configuration error |
| 4 | Data error |
| 5 | API error |
| 130 | Interrupted (Ctrl+C) |

## Tips & Tricks

### Batch Processing

```bash
# Run multiple backtests
for symbol in AAPL MSFT GOOGL; do
  gpt-trader backtest \
    --symbol $symbol \
    --start 2024-01-01 \
    --end 2024-06-30 \
    --run-tag $symbol
done
```

### Parallel Execution

```bash
# Use GNU parallel for multiple symbols
parallel -j 4 gpt-trader backtest \
  --symbol {} \
  --start 2024-01-01 \
  --end 2024-06-30 \
  ::: AAPL MSFT GOOGL AMZN
```

### Output Redirection

```bash
# Save results to file
gpt-trader backtest ... > results.txt 2>&1

# Pipe to analysis tool
gpt-trader backtest ... | grep "Sharpe Ratio"

# Silent mode with error capture
gpt-trader -q backtest ... 2> errors.log
```

### Automation

```bash
# Cron job for daily backtests
0 18 * * * /usr/bin/poetry run python -m bot.cli backtest \
  --symbol SPY \
  --start $(date -d "1 year ago" +%Y-%m-%d) \
  --end $(date +%Y-%m-%d) \
  >> /var/log/gpt-trader.log 2>&1
```

## Troubleshooting

### Command Not Found

```bash
# Ensure poetry environment is activated
poetry shell

# Or use poetry run prefix
poetry run python -m bot.cli --help
```

### Permission Denied

```bash
# Check file permissions
ls -la data/

# Fix permissions
chmod 755 data/
```

### Out of Memory

```bash
# Reduce data load
gpt-trader backtest \
  --symbol SPY \
  --start 2024-01-01 \
  --end 2024-03-31  # Shorter period

# Increase memory limit
ulimit -v unlimited
```

---

**Need more help?** Run `gpt-trader --help` or check the [FAQ](FAQ.md)
