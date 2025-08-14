# Paper Trading Guide

This guide explains how to use the GPT-Trader paper trading functionality to test your strategies in a live-like environment without financial risk.

## Overview

Paper trading allows you to:
- Test strategies with real market data
- Validate portfolio management logic
- Practice risk management
- Build confidence before live trading
- Collect performance data for strategy refinement

## Prerequisites

### 1. Alpaca Account Setup

1. **Create an Alpaca Account**: Sign up at [alpaca.markets](https://alpaca.markets)
2. **Get API Keys**: Navigate to your account dashboard and generate API keys
3. **Paper Trading**: Use the paper trading environment (no real money involved)

### 2. Environment Variables

Set your Alpaca credentials in your environment:

```bash
export ALPACA_API_KEY_ID="your_api_key_here"
export ALPACA_API_SECRET_KEY="your_secret_key_here"
```

Or create a `.env` file in your project root:

```env
ALPACA_API_KEY_ID=your_api_key_here
ALPACA_API_SECRET_KEY=your_secret_key_here
```

### 3. Install Dependencies

```bash
poetry install
```

## Quick Start

### Using the CLI

Start paper trading with the command line interface:

```bash
# Basic paper trading with default settings
poetry run gpt-trader paper \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA" \
  --strategy trend_breakout \
  --risk-pct 0.5 \
  --max-positions 5 \
  --rebalance-interval 300

# Advanced configuration
poetry run gpt-trader paper \
  --symbols "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NETFLIX" \
  --strategy trend_breakout \
  --risk-pct 0.3 \
  --max-positions 8 \
  --rebalance-interval 600 \
  --donchian 55 \
  --atr 20 \
  --atr-k 2.0
```

### Using the Example Script

Run the provided example script:

```bash
python examples/paper_trading_example.py
```

### Programmatic Usage

```python
import asyncio
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.live.trading_engine import LiveTradingEngine
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.trend_breakout import TrendBreakoutStrategy, TrendBreakoutParams

async def run_paper_trading():
    # Initialize broker
    broker = AlpacaPaperBroker(
        api_key="your_api_key",
        secret_key="your_secret_key"
    )
    
    # Initialize strategy
    strategy = TrendBreakoutStrategy(
        TrendBreakoutParams(
            donchian_lookback=55,
            atr_period=20,
            atr_k=2.0,
        )
    )
    
    # Initialize portfolio rules
    rules = PortfolioRules(
        per_trade_risk_pct=0.005,  # 0.5%
        atr_k=2.0,
        max_positions=10,
        max_gross_exposure_pct=0.60,
        cost_bps=5.0,
    )
    
    # Initialize trading engine
    engine = LiveTradingEngine(
        broker=broker,
        strategy=strategy,
        rules=rules,
        symbols=["AAPL", "MSFT", "GOOGL"],
        rebalance_interval=300,  # 5 minutes
        max_positions=10,
    )
    
    # Start trading
    await engine.start()

# Run the trading engine
asyncio.run(run_paper_trading())
```

## Configuration Options

### Strategy Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--donchian` | Donchian channel lookback period | 55 | 10-200 |
| `--atr` | ATR calculation period | 20 | 5-50 |
| `--atr-k` | ATR multiplier for stops | 2.0 | 0.5-5.0 |

### Portfolio Management

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--risk-pct` | Risk per trade as % of equity | 0.5 | 0.1-2.0 |
| `--max-positions` | Maximum concurrent positions | 10 | 1-50 |
| `--rebalance-interval` | Rebalancing frequency (seconds) | 300 | 60-3600 |

### Risk Management

The system includes several risk management features:

- **Position Sizing**: ATR-based position sizing to limit risk per trade
- **Maximum Positions**: Limits the number of concurrent positions
- **Gross Exposure**: Limits total portfolio exposure
- **Circuit Breakers**: Automatic stops when risk limits are exceeded

## Monitoring and Control

### Real-time Monitoring

The trading engine provides real-time information:

```python
# Get trading summary
summary = engine.get_trading_summary()
print(f"Portfolio value: ${summary['portfolio_summary']['portfolio_value']:,.2f}")
print(f"Unrealized P&L: ${summary['portfolio_summary']['unrealized_pl']:,.2f}")
print(f"Position count: {summary['portfolio_summary']['position_count']}")

# Get trading history
history = engine.get_trading_history()
for decision in history:
    print(f"{decision.timestamp}: {decision.action} {decision.symbol} {decision.quantity} shares")
```

### Portfolio Analysis

```python
# Get position summary
positions_df = engine.portfolio_manager.get_position_summary()
print(positions_df)

# Get portfolio history
history_df = engine.portfolio_manager.get_portfolio_history(hours=24)
print(history_df)
```

### Stopping the Engine

```python
# Graceful shutdown
await engine.stop()

# Or use Ctrl+C in the CLI
```

## Best Practices

### 1. Start Small

- Begin with a small number of symbols (3-5)
- Use conservative risk parameters
- Monitor performance closely

### 2. Validate Strategy

- Run extensive backtests before paper trading
- Compare paper trading results with backtest expectations
- Identify and address any discrepancies

### 3. Monitor Performance

- Track key metrics: Sharpe ratio, drawdown, win rate
- Monitor execution quality and slippage
- Watch for strategy drift or regime changes

### 4. Risk Management

- Set appropriate position limits
- Use stop losses and position sizing
- Monitor correlation between positions
- Have emergency stop procedures

### 5. Data Quality

- Verify market data accuracy
- Monitor for data gaps or delays
- Have fallback data sources if needed

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API keys are correct
   - Check network connectivity
   - Ensure Alpaca service is available

2. **No Trading Activity**
   - Check if market is open
   - Verify strategy is generating signals
   - Review risk management settings

3. **Unexpected Trades**
   - Review strategy logic
   - Check signal generation
   - Verify position sizing calculations

### Debug Mode

Enable debug logging for detailed information:

```bash
export LOG_LEVEL=DEBUG
poetry run gpt-trader paper --symbols "AAPL" --risk-pct 0.5
```

### Performance Monitoring

Monitor system performance:

```python
# Check data update frequency
data_manager = engine.data_manager
print(f"Data cache size: {len(data_manager.data_cache)}")

# Check portfolio refresh rate
portfolio_manager = engine.portfolio_manager
print(f"Last refresh: {portfolio_manager.state.timestamp}")
```

## Next Steps

After successful paper trading:

1. **Analyze Results**: Review performance metrics and trading patterns
2. **Optimize Strategy**: Use insights to refine strategy parameters
3. **Scale Up**: Gradually increase position sizes and symbol count
4. **Live Trading**: Consider transitioning to live trading with small amounts

## Support

For issues or questions:

1. Check the logs for error messages
2. Review the configuration settings
3. Test with a single symbol first
4. Verify Alpaca account status and API access

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Regularly rotate API keys
- Monitor account activity for unauthorized access
- Use paper trading for testing, not live accounts
