# Paper Trading Guide

---
status: current
last-updated: 2025-03-01
consolidates:
  - PAPER_TRADING_IMPLEMENTATION.md
  - PAPER_ENGINE_DECOUPLING.md
  - PAPER_TRADING_PROGRESS.md
  - PAPER_TRADING_SESSION_REPORT.md
---

## Overview

Paper trading provides risk-free simulation of trading strategies using real market data but simulated execution.

## Implementation

### Mock Broker
The system uses a deterministic broker stub for paper trading that provides:
- Deterministic fills for testing
- Random walk price generation
- Configurable slippage simulation
- Order tracking and PnL calculation

### Configuration
```bash
# Enable paper trading mode
poetry run perps-bot --profile dev

# The dev profile automatically uses:
# - Mock broker with simulated fills
# - Tiny positions (0.001 BTC)
# - Extensive logging
# - No real money at risk
```

### Slice Quick Start (Python API)

Run the slice directly from Python when you do not need the full CLI profile:

    from features.paper_trade import start_paper_trading, get_status, stop_paper_trading

    start_paper_trading(
        strategy="SimpleMAStrategy",
        symbols=["AAPL", "MSFT", "GOOGL"],
        initial_capital=100_000,
        fast_period=10,
        slow_period=30,
    )

    status = get_status()
    print(status.summary())

    results = stop_paper_trading()
    print(results.summary())

The module is completely self-contained: strategies, data adapters, execution, risk controls, and dataclasses all live inside the slice.

### Module Layout

    paper_trade/
    ├── paper_trade.py   # Orchestration helpers
    ├── strategies.py    # Local strategy implementations
    ├── data.py          # Live/historical data adapters
    ├── execution.py     # Commission/slippage-aware fills
    ├── risk.py          # Position limits, drawdown guards
    └── types.py         # Slice-local dataclasses and responses

### Strategy Catalog

Five built-in strategies ship with the slice:
1. `SimpleMAStrategy` – moving-average crossover
2. `MomentumStrategy` – momentum signal band
3. `MeanReversionStrategy` – Bollinger-band mean reversion
4. `VolatilityStrategy` – low-volatility filter
5. `BreakoutStrategy` – breakout detector

### Feature Highlights

- Real-time data with configurable polling interval and market-hours awareness
- Execution simulator supporting commission, slippage, maximum concurrent positions, and trade logging
- Risk controls for position size, daily loss limits, drawdown guard, and cash reserve enforcement
- Performance tracking with equity curve, trade log, and summary metrics

### Configuration Example

    start_paper_trading(
        strategy="MomentumStrategy",
        symbols=["SPY", "QQQ"],
        initial_capital=50_000,
        commission=0.001,
        slippage=0.0005,
        position_size=0.95,
        max_positions=10,
        update_interval=60,
    )

## Features

### Market Simulation
- Generates realistic price movements
- Simulates order book depth
- Includes spread and slippage
- Provides fill notifications

### Risk-Free Testing
- Test strategies without capital
- Validate order logic
- Debug execution paths
- Performance benchmarking

## Usage

### Quick Start
```bash
# Run with deterministic broker
poetry run perps-bot --profile dev --dev-fast

# Monitor performance
tail -f var/logs/perps_bot.log | grep "PnL"
```

### Advanced Configuration
```python
# Custom paper trading settings
config = {
    "mock_broker": {
        "initial_balance": 100000,
        "spread_bps": 10,
        "slippage_bps": 5,
        "fill_probability": 0.95
    }
}
```

## Transition to Live Trading

### Validation Steps
1. Run paper trading for minimum 1 week
2. Achieve consistent profitability
3. Verify risk metrics stay within limits
4. Review all error logs

### Migration Process
```bash
# Step 1: Canary testing (tiny real positions)
poetry run perps-bot --profile canary --dry-run

# Step 2: Limited live trading
poetry run perps-bot --profile canary

# Step 3: Full production
poetry run perps-bot --profile prod
```

## Performance Metrics

Track these metrics during paper trading:
- Win rate (target > 55%)
- Sharpe ratio (target > 1.0)
- Maximum drawdown (limit < 10%)
- Average trade duration
- Risk/reward ratio

## Best Practices

1. **Extended Testing**: Run paper trading for at least 100 trades
2. **Market Conditions**: Test across different market regimes
3. **Stress Testing**: Simulate extreme market conditions
4. **Logging**: Keep detailed logs for analysis
5. **Gradual Scaling**: Start with tiny positions when going live
