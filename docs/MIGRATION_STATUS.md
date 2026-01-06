# Migration Status (Composition Root / Orchestration)

The DI migration to `ApplicationContainer` is **complete**. `TradingBot` now receives all
services directly from the container—no intermediate `ServiceRegistry`.

## Current Source of Truth

- **Canonical wiring**: `src/gpt_trader/app/container.py` (`ApplicationContainer`)
- **CLI bot creation**: `src/gpt_trader/cli/services.py` (`instantiate_bot()` → container → `create_bot()`)
- **TradingBot**: Receives `broker`, `risk_manager`, `event_store`, etc. directly from container

## Deprecated (Avoid for New Code)

| File | Item | Status | Notes |
|------|------|--------|-------|
| `orchestration/service_registry.py` | `ServiceRegistry` | Deprecated | Use `ApplicationContainer` (removal: v3.0) |
| `app/container.py` | `create_service_registry()` | Deprecated | Emits `DeprecationWarning` (removal: v3.0) |
| `orchestration/storage.py` | `StorageBootstrapper` | Deprecated | Use container properties |
| `orchestration/bootstrap.py` | `prepare_bot()` | Deprecated | Emits warning; use `ApplicationContainer` |
| `orchestration/bootstrap.py` | `prepare_bot_with_container()` | Deprecated | Use `ApplicationContainer` |
| `orchestration/bootstrap.py` | `BootstrapResult` | Deprecated | Returns `ServiceRegistry` |
| `orchestration/read_only_broker.py` | `ReadOnlyBroker` | Deprecated | Unused; use `BotConfig.read_only` |

## Supported Convenience Functions

These functions use `ApplicationContainer` internally and are NOT deprecated:

- `orchestration/bootstrap.py` → `build_bot(config)` - Canonical way to create a bot
- `orchestration/bootstrap.py` → `bot_from_profile(profile)` - Create bot from profile name

## Compatibility Layers (Still Present)

- `orchestration/bootstrap.py` uses `ApplicationContainer` internally but returns a `ServiceRegistry` in `BootstrapResult` for backwards compatibility with legacy callers.

## Migration Rules of Thumb

1. Add new services/dependencies to `ApplicationContainer` first.
2. Do not introduce new imports of `ServiceRegistry`/`StorageBootstrapper` in production code.
3. Prefer passing dependencies explicitly (or pass the container) instead of expanding legacy registries.
4. Use `build_bot()` or `bot_from_profile()` for simple bot creation without touching `ServiceRegistry`.

## v3.0 Removal Plan

The following items are scheduled for removal in v3.0:

| Item | Current State | Removal Action |
|------|--------------|----------------|
| `ServiceRegistry` | Unused at runtime | Delete `orchestration/service_registry.py` |
| `create_service_registry()` | Emits `DeprecationWarning` | Remove method from `ApplicationContainer` |
| `BootstrapResult.registry` | Returns `ServiceRegistry` | Remove field from dataclass |
| `StorageBootstrapper` | Unused | Delete `orchestration/storage.py` |

**Pre-removal checklist:**
- [ ] Grep codebase for `ServiceRegistry` imports—migrate or delete
- [ ] Delete tests marked with `@pytest.mark.legacy` (run `pytest -m legacy --collect-only` to list)
- [ ] Remove `registry` parameter from `CoordinatorContext`

**Legacy-marked tests** (delete during v3.0 removal):
- `tests/unit/gpt_trader/orchestration/test_service_registry.py` (entire module)
- `test_create_service_registry` in `tests/unit/app/test_container.py`
- `test_create_service_registry_emits_deprecation_warning` in `tests/unit/app/test_container.py`
- `test_prepare_bot_with_existing_registry` in `tests/unit/gpt_trader/orchestration/test_bootstrap.py`
- `test_build_bot_container_creates_registry` in `tests/unit/gpt_trader/orchestration/test_bootstrap.py`
