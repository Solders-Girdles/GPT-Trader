# Enhanced CLI Documentation

## Overview

The GPT-Trader CLI has been completely redesigned with enhanced UX, organization, and visual appeal. The new CLI provides a modern, user-friendly interface with rich formatting, better help text, progress indicators, and improved error handling.

## Key Features

### 🎨 Rich Formatting & Visual Appeal
- **Color-coded output** with consistent theming
- **ASCII art banner** with system status
- **Formatted tables** for results display
- **Progress indicators** and spinners
- **Styled error messages** and warnings

### 📚 Better Help & Organization
- **Comprehensive help text** with examples
- **Logical argument grouping** by functionality
- **Command aliases** for convenience (e.g., `bt` for `backtest`)
- **Detailed descriptions** for each command

### ⚙️ Configuration Profiles
- **Profile-based configuration** from YAML files
- **Default settings** stored in `~/.gpt-trader/profiles/`
- **Profile merging** with command-line arguments
- **Easy profile management**

### 🔧 Enhanced Error Handling
- **Graceful error recovery**
- **Detailed error messages**
- **Environment validation**
- **Input validation** with helpful feedback

### 📊 System Status Display
- **Market status** (open/closed)
- **Data directory** information
- **Configuration status**
- **API key validation**

## Installation

The enhanced CLI requires additional dependencies:

```bash
pip install rich pytz pyyaml
```

Or update your `pyproject.toml`:

```toml
[tool.poetry.dependencies]
rich = "^13.7.0"
pytz = "^2024.1"
pyyaml = "^6.0.1"
```

## Usage

### Basic Usage

```bash
# Show help with banner
gpt-trader

# Show version
gpt-trader --version

# Disable colors
gpt-trader --no-color

# Increase verbosity
gpt-trader -v  # INFO level
gpt-trader -vv # DEBUG level

# Quiet mode
gpt-trader -q
```

### Configuration Profiles

Create a profile file at `~/.gpt-trader/profiles/my_profile.yaml`:

```yaml
# Strategy defaults
strategy: trend_breakout
donchian: 55
atr: 20
atr_k: 2.0

# Risk management
risk_pct: 0.5
max_positions: 10
cost_bps: 5.0

# Data validation
data_strict: repair

# Common symbols
symbols: AAPL,MSFT,GOOGL,SPY,QQQ

# Verbosity
verbose: 1
```

Use the profile:

```bash
gpt-trader --profile my_profile backtest --symbol AAPL --start 2023-01-01 --end 2023-12-31
```

#### Extended Profile Fields

```yaml
# Feature sets to compose for research datasets
features:
  - returns
  - volatility
  - trend
  - momentum
  - volume
  - microstructure
  - calendar

# Evolution settings (research mode)
evolution:
  seeds:
    from: data/optimization/my_run/seeds.json  # path to seeds file
    mode: merge  # merge | replace
    topk: 5
  generations: 50
  population_size: 24
  early_stopping: true

# Optimizer defaults
optimizer:
  method: grid  # grid | evolutionary | both
  max_workers: 4

# Risk/policy controls (paper/live)
risk:
  cost_adjusted_sizing: true
  slippage_bps: 5
  max_turnover_per_rebalance: 0.2

# Monitoring thresholds
monitoring:
  min_transition_smoothness: 0.6
```

### Command Aliases

The CLI provides convenient aliases for common commands:

- `bt` → `backtest`
- `opt` → `optimize`
- `wf` → `walk-forward`

```bash
# These are equivalent
gpt-trader backtest --help
gpt-trader bt --help
```

## Commands

### Backtest

Enhanced backtesting with comprehensive parameter organization:

```bash
# Basic backtest
gpt-trader backtest --symbol AAPL --start 2023-01-01 --end 2023-12-31

# Advanced backtest with custom parameters
gpt-trader backtest \
  --symbol-list universe.csv \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --donchian 55 \
  --atr 20 \
  --risk-pct 1.0 \
  --max-positions 10 \
  --regime on \
  --regime-window 200

# Using aliases
gpt-trader bt --symbol SPY --start 2023-01-01 --end 2023-12-31
```

**Features:**
- **Strategy Configuration** group for strategy parameters
- **Data Selection** group for symbol and date inputs
- **Risk Management** group for risk controls
- **Regime Filter** group for market regime detection
- **Execution Settings** group for trading logic
- **Output Options** group for results configuration

### Paper Trading

Enhanced paper trading with live market data:

```bash
# Basic paper trading
gpt-trader paper --symbols AAPL,MSFT,GOOGL --risk-pct 1.0

# Advanced paper trading
gpt-trader paper \
  --symbols SPY,QQQ \
  --risk-pct 0.5 \
  --max-positions 5 \
  --donchian 55 \
  --atr 20 \
  --atr-k 2.0 \
  --rebalance-interval 300
```

**Features:**
- **Real-time market data** from Alpaca
- **Live portfolio management**
- **Strategy execution** with risk controls
- **Performance monitoring**

### Optimization

Strategy parameter optimization:

```bash
# Basic optimization
gpt-trader optimize --symbol AAPL --start 2023-01-01 --end 2023-12-31

# Grid search optimization
gpt-trader opt \
  --symbol-list universe.csv \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --method grid \
  --param donchian:20:100:10 \
  --param atr:10:50:5
```

### Walk-Forward Validation

Strategy robustness testing:

```bash
# Basic walk-forward validation
gpt-trader walk-forward \
  --results optimization_results.csv \
  --symbols AAPL,MSFT,GOOGL

# Custom validation windows
gpt-trader wf \
  --results results.csv \
  --symbols SPY,QQQ \
  --train-months 18 \
  --test-months 3 \
  --step-months 3 \
  --min-windows 5
```

### Deployment

Deploy optimized strategies:

```bash
# Deploy from optimization results
gpt-trader deploy \
  --results optimization_results.csv \
  --symbols AAPL,MSFT,GOOGL

# Deploy with custom filters
gpt-trader deploy \
  --results results.csv \
  --symbols SPY,QQQ \
  --min-sharpe 1.5 \
  --max-drawdown 0.10 \
  --min-trades 50
```

### Monitoring

Monitor deployed strategies:

```bash
# Basic monitoring
gpt-trader monitor --min-sharpe 0.8 --max-drawdown 0.12

# Monitoring with webhook alerts
gpt-trader monitor \
  --min-sharpe 1.0 \
  --max-drawdown 0.10 \
  --webhook-url https://hooks.slack.com/... \
  --alert-cooldown 12
```

### Rapid Evolution

Evolutionary optimization:

```bash
# Rapid evolutionary optimization
gpt-trader rapid-evolution
```

## Output Examples

### Welcome Banner

```
╔═══════════════════════════════════════════╗
║       🤖 GPT-TRADER PLATFORM v2.0 🤖      ║
║   AI-Powered Trading Strategy Development  ║
╚═══════════════════════════════════════════╝

┌───────────────── System Status ─────────────────┐
│ 📊 Data Directory    /path/to/data              │
│ 🔧 Config Loaded     ✓ Default                  │
│ 📈 Market Status     Open                       │
└─────────────────────────────────────────────────┘
```

### Backtest Results

```
┌───────────────── Backtest Results ──────────────┐
│ Metric          │ Value                         │
├─────────────────┼───────────────────────────────┤
│ Total Return    │ 15.23%                        │
│ CAGR            │ 12.45%                        │
│ Sharpe Ratio    │ 1.234                         │
│ Max Drawdown    │ -8.76%                        │
│ Win Rate        │ 65.2%                         │
│ Total Trades    │ 142                           │
└─────────────────────────────────────────────────┘
```

### Progress Indicators

```
⠋ Running backtest...
✓ Backtest complete!
```

## Error Handling

The enhanced CLI provides clear, actionable error messages:

```
┌───────────────────── Error ─────────────────────┐
│ ✗ Alpaca credentials not found. Set             │
│   ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.  │
└─────────────────────────────────────────────────┘
```

## Environment Validation

The CLI automatically validates your environment:

- ✅ Data directory exists
- ⚠️ Alpaca API keys configured (for paper/live trading)
- ✅ Python version compatibility
- ✅ Required dependencies installed

## Tips & Best Practices

1. **Use profiles** for common configurations
2. **Use aliases** for faster command entry
3. **Enable verbose mode** for debugging (`-v` or `-vv`)
4. **Use quiet mode** for scripts (`-q`)
5. **Check market status** before paper trading
6. **Validate inputs** before running expensive operations

## Migration from v1

The enhanced CLI is fully backward compatible. Existing commands will work without changes:

```bash
# Old way (still works)
gpt-trader backtest --symbol AAPL --start 2023-01-01 --end 2023-12-31

# New way (enhanced)
gpt-trader bt --symbol AAPL --start 2023-01-01 --end 2023-12-31 --verbose
```

## Troubleshooting

### Common Issues

1. **Rich not installed**: `pip install rich pytz pyyaml`
2. **Profile not found**: Check `~/.gpt-trader/profiles/` directory
3. **Color issues**: Use `--no-color` flag
4. **Permission errors**: Check file/directory permissions

### Debug Mode

Enable debug mode for detailed information:

```bash
gpt-trader -vv backtest --symbol AAPL --start 2023-01-01 --end 2023-12-31
```

## Contributing

When adding new commands to the CLI:

1. Use the enhanced structure with argument groups
2. Add comprehensive help text with examples
3. Use the CLITheme for consistent styling
4. Add progress indicators for long-running operations
5. Include proper error handling and validation
6. Add command aliases if appropriate
