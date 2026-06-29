# Paper Trading Guide

---
status: current
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
from gpt_trader.cli.services import load_profile_config
from gpt_trader.config.types import Profile

config = load_profile_config(Profile.PAPER)
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

## Readiness Evidence Inputs

Paper trading produces evidence that feeds the readiness checklist; it does not
itself authorize live execution. Live profiles only run after the gates in
[Live Operations](production.md) and the
[Direction](DIRECTION.md) have been
satisfied with explicit human approval.

### What paper runs should produce
1. Multi-day paper sessions with daily reports archived
2. Strategy/risk metrics measured against the readiness pillars
3. Reviewed error and guard logs

### Dry-run validation of profile wiring
```bash
# Validate canary profile settings without exchange orders
uv run gpt-trader run --profile canary --dry-run
```

For any live profile run, follow the gate sequence in
[Live Operations](production.md#live-gate-sequence). Do not promote past
`--dry-run` without recorded approval.

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
