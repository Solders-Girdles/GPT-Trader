# GPT-Trader Paper Trading Setup Guide

## Overview

This guide walks you through setting up and using the GPT-Trader paper trading system with Alpaca Markets. Paper trading allows you to test your strategies with real market data without risking actual money.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Alpaca Account Setup](#alpaca-account-setup)
3. [Configuration](#configuration)
4. [Running Paper Trading](#running-paper-trading)
5. [Monitoring Performance](#monitoring-performance)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software
- Python 3.11+ installed
- Poetry dependency manager
- Git

### Required Dependencies
```bash
# Install if not already present
poetry add alpaca-py streamlit plotly pandas numpy
```

### System Requirements
- Minimum 4GB RAM
- Stable internet connection for real-time data
- Modern web browser for dashboard

## Alpaca Account Setup

### Step 1: Create Alpaca Account
1. Go to [https://app.alpaca.markets/signup](https://app.alpaca.markets/signup)
2. Complete registration with email verification
3. Enable paper trading mode (default for new accounts)

### Step 2: Generate API Keys
1. Log into Alpaca dashboard
2. Navigate to "API Keys" section
3. Generate new paper trading API keys
4. **Important**: Save both API Key and Secret Key securely

### Step 3: Verify Paper Trading Mode
- Ensure you're in "Paper Trading" mode (toggle in top-right corner)
- Paper trading uses simulated money ($100,000 default)
- All trades are simulated but use real market data

## Configuration

### Environment Variables
Create a `.env` file in project root:

```bash
# Alpaca Paper Trading Credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Optional Configuration
ALPACA_DATA_FEED=iex  # or 'sip' for premium data
PAPER_TRADING_MODE=true
MAX_POSITION_SIZE=10000  # Maximum $ per position
DAILY_TRADE_LIMIT=100   # Maximum trades per day
```

### Configuration File
Create `config/paper_trading.yaml`:

```yaml
paper_trading:
  # Account Settings
  initial_capital: 100000
  currency: USD
  
  # Risk Management
  max_position_size: 10000
  max_portfolio_risk: 0.02  # 2% max risk
  stop_loss_pct: 0.05       # 5% stop loss
  
  # Execution Settings
  order_types:
    - market
    - limit
  default_time_in_force: day
  
  # Data Settings
  data_frequency: 1min
  historical_days: 30
  
  # Strategies to Trade
  active_strategies:
    - demo_ma
    - trend_breakout
    - mean_reversion
    - momentum
    - volatility
```

## Running Paper Trading

### Step 1: Start the Trading System

```bash
# Navigate to project directory
cd /path/to/GPT-Trader

# Activate environment
poetry shell

# Run paper trading
poetry run python src/bot/brokers/alpaca/paper_trading_bridge.py
```

### Step 2: Using the Orchestrator

```python
# examples/run_paper_trading.py
from bot.integration.orchestrator import IntegratedOrchestrator
from bot.brokers.alpaca.paper_trading_bridge import PaperTradingBridge
from bot.brokers.alpaca.alpaca_client import AlpacaClient

# Initialize components
client = AlpacaClient()
bridge = PaperTradingBridge(client)
orchestrator = IntegratedOrchestrator()

# Configure strategies
strategies = ['demo_ma', 'trend_breakout', 'mean_reversion']
symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Run paper trading
orchestrator.run_paper_trading(
    strategies=strategies,
    symbols=symbols,
    broker=bridge,
    mode='paper'
)
```

### Step 3: Command Line Usage

```bash
# Start paper trading with specific strategy
poetry run gpt-trader paper-trade --strategy demo_ma --symbols AAPL,GOOGL

# Run all strategies
poetry run gpt-trader paper-trade --all-strategies --top-symbols 10

# Custom configuration
poetry run gpt-trader paper-trade --config config/paper_trading.yaml
```

## Monitoring Performance

### Start the Dashboard

```bash
# Run the Streamlit dashboard
poetry run streamlit run src/bot/dashboard/app.py

# Dashboard will open at http://localhost:8501
```

### Dashboard Features

#### Real-time Monitoring
- **Portfolio Overview**: Total value, daily P&L, returns
- **Position Tracking**: Current positions with unrealized P&L
- **Strategy Performance**: Individual strategy metrics
- **Risk Metrics**: VaR, max drawdown, Sharpe ratio

#### Historical Analysis
- **P&L Charts**: Daily, weekly, monthly performance
- **Trade History**: Complete log of all trades
- **Performance Attribution**: Breakdown by strategy/symbol
- **Risk Analytics**: Drawdown periods, volatility analysis

#### Export Options
- Download trade history as CSV
- Export performance reports as PDF
- Save charts as images

### Performance Metrics

Key metrics tracked:
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade**: Mean profit/loss per trade
- **Risk/Reward**: Average win vs average loss

## Validation & Testing

### Run Validation Tests

```bash
# Run paper trading validation suite
poetry run pytest tests/paper_trading/test_validation_suite.py -v

# Test specific components
poetry run pytest tests/paper_trading/ -k "alpaca" -v
poetry run pytest tests/paper_trading/ -k "position" -v
```

### Manual Validation Checklist

- [ ] Alpaca connection successful
- [ ] Orders submitted and filled
- [ ] Positions update correctly
- [ ] P&L calculations accurate
- [ ] Dashboard displays data
- [ ] WebSocket streaming works
- [ ] Stop losses trigger
- [ ] Daily limits enforced

## Troubleshooting

### Common Issues

#### Connection Errors
```
Error: Unable to connect to Alpaca
Solution: 
- Check API keys in .env file
- Verify paper trading mode is enabled
- Check internet connection
- Ensure API rate limits not exceeded
```

#### No Market Data
```
Error: No data received from WebSocket
Solution:
- Market may be closed (9:30 AM - 4:00 PM ET)
- Check data subscription (IEX vs SIP)
- Verify symbol is tradeable
```

#### Order Rejections
```
Error: Order rejected by broker
Solution:
- Check account buying power
- Verify symbol is tradeable
- Check position size limits
- Ensure market is open
```

#### Dashboard Not Loading
```
Error: Streamlit connection refused
Solution:
- Check port 8501 is not in use
- Restart Streamlit server
- Clear browser cache
- Check firewall settings
```

### Debug Mode

Enable detailed logging:

```python
import logging

# Set debug level
logging.basicConfig(level=logging.DEBUG)

# Or use environment variable
export LOG_LEVEL=DEBUG
```

### Support Resources

- **Alpaca Documentation**: [https://alpaca.markets/docs/](https://alpaca.markets/docs/)
- **API Status**: [https://status.alpaca.markets/](https://status.alpaca.markets/)
- **Community Forum**: [https://forum.alpaca.markets/](https://forum.alpaca.markets/)
- **GPT-Trader Issues**: [GitHub Issues](https://github.com/gpt-trader/issues)

## Best Practices

### Risk Management
1. Start with small position sizes
2. Use stop losses on all trades
3. Monitor daily trade limits
4. Review performance daily
5. Adjust strategy parameters based on results

### Strategy Testing
1. Test one strategy at a time initially
2. Run for at least 2 weeks before evaluation
3. Compare against buy-and-hold benchmark
4. Document parameter changes
5. Keep trade journal

### Production Preparation
1. Paper trade for minimum 30 days
2. Achieve consistent profitability
3. Test during different market conditions
4. Validate risk management works
5. Have emergency stop procedures

## Next Steps

After successful paper trading:

1. **Analyze Results**: Review performance metrics and identify improvements
2. **Optimize Strategies**: Tune parameters based on paper trading results
3. **Scale Testing**: Gradually increase position sizes
4. **Risk Assessment**: Ensure drawdowns are acceptable
5. **Live Trading**: Consider transitioning to live trading with small capital

---

**Important**: Always start with paper trading. Never trade with real money until you have validated your strategies and are comfortable with the risks involved.

For additional help, consult the main [README.md](../README.md) or raise an issue on GitHub.