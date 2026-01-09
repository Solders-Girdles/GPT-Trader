# Migration Status (Composition Root / Orchestration)

The DI migration to `ApplicationContainer` is **complete** (v3.0). All legacy
`ServiceRegistry` code has been removed. `TradingBot` now receives all services
directly from the container.

## Current Source of Truth

- **Canonical wiring**: `src/gpt_trader/app/container.py` (`ApplicationContainer`)
- **CLI bot creation**: `src/gpt_trader/cli/services.py` (`instantiate_bot()` → container → `create_bot()`)
- **TradingBot**: Receives `broker`, `risk_manager`, `event_store`, etc. directly from container

## Supported Functions

### ApplicationContainer (Primary)

```python
from gpt_trader.app.container import ApplicationContainer

container = ApplicationContainer(config)
bot = container.create_bot()
```

### Convenience Functions

These functions use `ApplicationContainer` internally:

- `app/bootstrap.py` → `build_bot(config)` - Canonical way to create a bot
- `app/bootstrap.py` → `bot_from_profile(profile)` - Create bot from profile name

```python
from gpt_trader.app.bootstrap import build_bot, bot_from_profile

bot = build_bot(config)
bot = bot_from_profile("demo")
```

> **Note:** The old `orchestration.bootstrap` path still works via re-export shim but emits a `DeprecationWarning`.

## v3.0 Changes (Completed)

The following legacy items were removed in v3.0:

| Item | Action Taken |
|------|--------------|
| `ServiceRegistry` | Deleted `orchestration/service_registry.py` |
| `ServiceRegistryProtocol` | Removed from `orchestration/protocols.py` |
| `create_service_registry()` | Removed from `ApplicationContainer` |
| `prepare_bot()` | Removed from `orchestration/bootstrap.py` |
| `prepare_bot_with_container()` | Removed from `orchestration/bootstrap.py` |
| `BootstrapResult` | Removed from `orchestration/bootstrap.py` |
| `CoordinatorContext.registry` | Field removed |
| Legacy test marker | Removed from `pytest.ini` |
| `features/brokerages/core/interfaces.py` | Deleted (was deprecated re-export shim) |
| `orchestration/storage.py` | Deleted (StorageBootstrapper was unused) |
| `orchestration/read_only_broker.py` | Deleted (ReadOnlyBroker was unused) |

## Status

All deprecated modules have been removed. The migration is complete.
