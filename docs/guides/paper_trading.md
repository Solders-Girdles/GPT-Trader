# Paper Trading Guide

---
status: current
last-updated: 2025-10-07
consolidates:
  - PAPER_TRADING_IMPLEMENTATION.md
  - PAPER_ENGINE_DECOUPLING.md
  - PAPER_TRADING_PROGRESS.md
  - PAPER_TRADING_SESSION_REPORT.md
---

## Overview

Paper trading provides risk-free simulation of trading strategies using simulated execution.
The `paper` and `dev` profiles run with `mock_broker` enabled, so no real orders
or API calls are made.

## Implementation

### Deterministic Broker
The default paper workflow uses the deterministic broker stub:
- Deterministic fills for testing
- Synthetic quotes (no external market data calls)
- Immediate execution with predictable order IDs

Implementation: `src/gpt_trader/features/brokerages/mock/deterministic.py`.

### Hybrid Paper Broker (experimental)
`src/gpt_trader/features/brokerages/paper/hybrid.py` supports real market data
with simulated execution. It is **not wired** in the default broker factory.
Use it only for experiments (custom container/broker factory).

### Configuration
```bash
# Paper profile (mock broker + dry run)
uv run gpt-trader run --profile paper

# TUI paper mode
uv run gpt-trader tui --mode paper

# Single-cycle smoke test
uv run gpt-trader run --profile paper --dev-fast
```

### Programmatic Entry (Python)

```python
from gpt_trader.app.container import ApplicationContainer
from gpt_trader.cli.services import load_config_from_yaml

config = load_config_from_yaml("config/profiles/paper.yaml")
bot = ApplicationContainer(config).create_bot()
```

### Module Layout

    config/profiles/paper.yaml        # Paper profile settings
    src/gpt_trader/features/brokerages/mock/deterministic.py
    src/gpt_trader/features/brokerages/paper/hybrid.py  # Experimental
    src/gpt_trader/features/live_trade/strategies/      # Strategy implementations

### Strategy Catalog

Paper mode uses the same strategies as live trading:
1. `baseline` – MA + RSI baseline
2. `mean_reversion` – Z-score mean reversion
3. `ensemble` – signal ensemble architecture

## Features

### Market Simulation
- Synthetic quotes from the deterministic broker
- Immediate fills with predictable IDs
- No external API calls

### Risk-Free Testing
- Test strategies without capital
- Validate order logic
- Debug execution paths
- Performance benchmarking

## Usage

### Quick Start
```bash
# Run with deterministic broker
uv run gpt-trader run --profile paper --dev-fast

# Monitor performance
tail -f ${COINBASE_TRADER_LOG_DIR:-var/logs}/coinbase_trader.log | grep "PnL"
```

### Advanced Configuration
```python
# Custom paper trading settings
from decimal import Decimal
from gpt_trader.features.brokerages.mock import DeterministicBroker

broker = DeterministicBroker(equity=Decimal("100000"))
broker.set_mark("BTC-PERP", Decimal("50000"))
# Set container._broker = broker before calling container.create_bot()
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
uv run gpt-trader run --profile canary --dry-run

# Step 2: Limited live trading
uv run gpt-trader run --profile canary

# Step 3: Full production
uv run gpt-trader run --profile prod
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
