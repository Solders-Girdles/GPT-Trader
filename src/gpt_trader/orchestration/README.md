# Orchestration

The orchestration layer coordinates all trading system components, managing the lifecycle
from configuration loading through order execution and runtime monitoring.

## Philosophy

This package follows **composition over inheritance**. Components are assembled at runtime
via the `ApplicationContainer` (preferred) or legacy `ServiceRegistry`. This enables:

- Easy testing via mock injection
- Profile-based configuration switching
- Clean separation between spot and perpetuals trading modes

## Key Components

| Module | Purpose |
|--------|---------|
| `configuration/` | Config loading, profiles, risk parameters |
| `execution/` | Order validation, submission, guard management |
| `live_execution.py` | Main live trading engine |
| `bootstrap.py` | Application initialization |
| `protocols.py` | Interface definitions (EventStore, OrdersStore) |

## Entry Points

- **Live Trading**: `LiveExecutionEngine` in `live_execution.py`
- **Strategy Orchestration**: `strategy_orchestrator/` package
- **Bot Lifecycle**: `trading_bot/` package

## Configuration Flow

```
profiles.yaml → ProfileLoader → BotConfig → ApplicationContainer → Services
```

## Related Packages

- `features/live_trade/` - Live trading strategies and risk management
- `features/brokerages/` - Brokerage adapters (Coinbase, etc.)
- `persistence/` - Event and order storage
