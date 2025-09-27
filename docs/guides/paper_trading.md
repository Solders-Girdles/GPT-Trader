# Paper Trading Guide

---
status: current
last-updated: 2025-01-01
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
The system uses a mock broker for paper trading that provides:
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
# Run with mock broker
poetry run perps-bot --profile dev --dev-fast

# Monitor performance
tail -f logs/perps_bot.log | grep "PnL"
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