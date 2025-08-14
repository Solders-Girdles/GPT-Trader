# Usage Guide

This document explains how to **run backtests**, **perform parameter optimizations**, **deploy strategies**, and **monitor performance** using GPT-Trader.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ installed
- Poetry installed (`pip install poetry`)
- Alpaca account (for paper trading)

### Installation
```bash
# Clone and install
git clone https://github.com/your-username/GPT-Trader.git
cd GPT-Trader
poetry install

# Set up Alpaca credentials (optional, for paper trading)
export ALPACA_API_KEY_ID="your_api_key_here"
export ALPACA_API_SECRET_KEY="your_secret_key_here"
```

---

## 1. Backtesting a Strategy

### Basic Backtest
```bash
poetry run gpt-trader backtest \
  --strategy trend_breakout \
  --symbols "AAPL,MSFT,GOOGL" \
  --start-date 2022-01-01 \
  --end-date 2022-12-31 \
  --risk-pct 0.5 \
  --max-positions 5
```

### Advanced Backtest with Custom Parameters
```bash
poetry run gpt-trader backtest \
  --strategy trend_breakout \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --start-date 2022-01-01 \
  --end-date 2022-12-31 \
  --risk-pct 0.3 \
  --max-positions 8 \
  --donchian 55 \
  --atr 20 \
  --atr-k 2.0 \
  --entry-confirm 2 \
  --exit-mode stop \
  --cost-bps 5 \
  --regime on
```

### Backtest with Symbol List File
```bash
poetry run gpt-trader backtest \
  --strategy trend_breakout \
  --symbol-list data/universe/sp100.csv \
  --start-date 2022-01-01 \
  --end-date 2022-12-31 \
  --risk-pct 0.5 \
  --max-positions 10
```

---

## 2. Strategy Optimization

### Grid Search Optimization
```bash
poetry run gpt-trader optimize-new \
  --name "my_grid_optimization" \
  --strategy trend_breakout \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --start-date 2022-01-01 \
  --end-date 2022-12-31 \
  --method grid \
  --output-dir "data/optimization/my_grid_optimization"
```

### Evolutionary Optimization
```bash
poetry run gpt-trader optimize-new \
  --name "my_evolutionary_optimization" \
  --strategy trend_breakout \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --start-date 2022-01-01 \
  --end-date 2022-12-31 \
  --method evolutionary \
  --generations 50 \
  --population-size 24 \
  --output-dir "data/optimization/my_evolutionary_optimization"
```

### Multi-Objective Optimization
```bash
poetry run gpt-trader optimize-new \
  --name "my_multi_objective_optimization" \
  --strategy trend_breakout \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --start-date 2022-01-01 \
  --end-date 2022-12-31 \
  --method multi_objective \
  --generations 100 \
  --population-size 50 \
  --objectives "sharpe,drawdown,consistency,novelty" \
  --output-dir "data/optimization/my_multi_objective_optimization"
```

---

## 3. Walk-Forward Validation

### Basic Walk-Forward Testing
```bash
poetry run gpt-trader walk-forward \
  --results "data/optimization/my_optimization/all_results.csv" \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --train-months 12 \
  --test-months 6 \
  --step-months 6 \
  --output "data/optimization/my_optimization/wf_validated.csv"
```

### Advanced Walk-Forward with Filters
```bash
poetry run gpt-trader walk-forward \
  --results "data/optimization/my_optimization/all_results.csv" \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META" \
  --train-months 12 \
  --test-months 6 \
  --step-months 6 \
  --min-windows 5 \
  --min-mean-sharpe 0.8 \
  --max-mean-drawdown 0.15 \
  --output "data/optimization/my_optimization/wf_validated.csv"
```

---

## 4. Strategy Deployment

### Deploy Best Strategies to Paper Trading
```bash
poetry run gpt-trader deploy \
  --results "data/optimization/my_optimization/wf_validated.csv" \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META" \
  --min-sharpe 1.2 \
  --max-drawdown 0.12 \
  --min-trades 30 \
  --max-strategies 3 \
  --validation-days 30 \
  --output-dir "data/deployment"
```

### Manual Paper Trading
```bash
poetry run gpt-trader paper \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --strategy trend_breakout \
  --risk-pct 0.3 \
  --max-positions 5 \
  --rebalance-interval 300 \
  --donchian 55 \
  --atr 20 \
  --atr-k 2.0
```

---

## 5. Enhanced Strategy Evolution

### Basic Evolution
```bash
poetry run gpt-trader enhanced-evolution \
  --symbols "AAPL,MSFT,GOOGL" \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --generations 50 \
  --population-size 24 \
  --output-dir "data/evolution"
```

### Knowledge-Enhanced Evolution
```bash
poetry run gpt-trader enhanced-evolution \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --generations 100 \
  --population-size 50 \
  --use-knowledge-base \
  --novelty-weight 0.3 \
  --diversity-weight 0.2 \
  --output-dir "data/evolution/knowledge_enhanced"
```

---

## 6. Performance Monitoring

### Monitor Deployed Strategies
```bash
poetry run gpt-trader monitor \
  --min-sharpe 0.8 \
  --max-drawdown 0.12 \
  --min-cagr 0.08 \
  --webhook-url "https://hooks.slack.com/services/YOUR/WEBHOOK/URL" \
  --alert-cooldown 24
```

### One-Shot Summary
Run a single monitoring cycle and print a compact summary (selection metrics and turnover statistics):

```bash
poetry run gpt-trader monitor \
  --min-sharpe 0.8 \
  --max-drawdown 0.12 \
  --min-cagr 0.08 \
  --once --print-summary
```

### Monitor with Custom Alerts
```bash
poetry run gpt-trader monitor \
  --min-sharpe 1.0 \
  --max-drawdown 0.10 \
  --min-cagr 0.12 \
  --max-daily-loss 0.05 \
  --min-win-rate 0.55 \
  --webhook-url "https://hooks.slack.com/services/YOUR/WEBHOOK/URL" \
  --email "your-email@example.com" \
  --alert-cooldown 12
```

---

## 7. Interactive Mode

### Start Interactive Session
```bash
poetry run gpt-trader interactive
```

### Interactive Commands
```python
# Available in interactive mode
>>> help()  # Show available commands
>>> backtest("AAPL,MSFT", "2022-01-01", "2022-12-31")
>>> optimize("trend_breakout", "AAPL,MSFT", "2022-01-01", "2022-12-31")
>>> paper_trade("AAPL,MSFT", risk_pct=0.3)
>>> exit()
```

---

## 8. Configuration Profiles

### Create a Profile
Create `~/.gpt-trader/profiles/my_profile.yaml`:

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
```

### Use Profile
```bash
poetry run gpt-trader backtest --profile my_profile
```

---

## 9. Output Files

### Backtest Outputs
- `PORT_<strategy>_<timestamp>.csv` - Equity curve and positions
- `PORT_<strategy>_<timestamp>.png` - Performance chart
- `PORT_<strategy>_<timestamp>_summary.csv` - Performance metrics
- `PORT_<strategy>_<timestamp>_trades.csv` - Trade-by-trade log

### Optimization Outputs
- `all_results.csv` - All optimization results
- `best_results.csv` - Top performing configurations
- `parameter_sensitivity.csv` - Parameter sensitivity analysis
- `correlation_matrix.csv` - Parameter correlation analysis
- `optimization_report.html` - Interactive HTML dashboard

### Walk-Forward Outputs
- `wf_validated.csv` - Validated strategies
- `wf_summary.csv` - Walk-forward performance summary
- `wf_plots/` - Walk-forward visualization plots

---

## 10. Common Parameters

### Strategy Parameters
- `--strategy`: Strategy name (trend_breakout, demo_ma, etc.)
- `--symbols`: Comma-separated list of symbols
- `--start-date`: Start date (YYYY-MM-DD)
- `--end-date`: End date (YYYY-MM-DD)

### Risk Management
- `--risk-pct`: Risk per trade (0.1-1.0)
- `--max-positions`: Maximum concurrent positions
- `--cost-bps`: Transaction costs in basis points

### Strategy-Specific Parameters
- `--donchian`: Donchian channel lookback period
- `--atr`: ATR period for volatility calculation
- `--atr-k`: ATR multiplier for position sizing
- `--entry-confirm`: Entry confirmation periods
- `--exit-mode`: Exit mode (signal, stop, both)

### Optimization Parameters
- `--method`: Optimization method (grid, evolutionary, multi_objective)
- `--generations`: Number of generations for evolutionary
- `--population-size`: Population size for evolutionary
- `--output-dir`: Output directory for results

---

## 11. Best Practices

### Backtesting
1. **Use sufficient data**: At least 2-3 years for reliable results
2. **Include transaction costs**: Set realistic `--cost-bps`
3. **Test multiple symbols**: Diversify across different assets
4. **Use regime filters**: Enable `--regime on` for market-aware strategies

### Optimization
1. **Start with grid search**: For initial parameter exploration
2. **Use walk-forward validation**: To prevent overfitting
3. **Set reasonable bounds**: Don't over-optimize parameter ranges
4. **Monitor optimization progress**: Check intermediate results

### Paper Trading
1. **Start small**: Use conservative risk settings
2. **Monitor closely**: Set up alerts and regular monitoring
3. **Document performance**: Keep track of strategy behavior
4. **Have exit criteria**: Know when to stop or adjust

---

## 12. Troubleshooting

### Common Issues
- **Import errors**: Run from project root directory
- **Missing data**: Check symbol names and date ranges
- **Alpaca errors**: Verify API credentials and account status
- **Memory issues**: Reduce symbol count or date range

### Getting Help
- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review the [examples](../examples/) directory
- Open an issue on GitHub for specific problems

---

For more advanced features and detailed explanations, see the [Complete Pipeline Guide](COMPLETE_PIPELINE.md) and [Enhanced CLI Guide](ENHANCED_CLI.md).
