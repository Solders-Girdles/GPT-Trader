# Migration Status (Composition Root / Orchestration)

This repo is mid-transition toward `ApplicationContainer` as the single composition root.

## Current Source of Truth

- **Canonical wiring**: `src/gpt_trader/app/container.py` (`ApplicationContainer`)
- **CLI bot creation**: `src/gpt_trader/cli/services.py` (`instantiate_bot()` → container → `create_bot()`)

## Deprecated (Avoid for New Code)

| File | Item | Status | Notes |
|------|------|--------|-------|
| `orchestration/service_registry.py` | `ServiceRegistry` | Deprecated | Use `ApplicationContainer` |
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
2. Only mirror into `ServiceRegistry` when a legacy call site cannot be migrated yet.
3. Do not introduce new imports of `ServiceRegistry`/`StorageBootstrapper` in production code.
4. Prefer passing dependencies explicitly (or pass the container) instead of expanding legacy registries.
5. Use `build_bot()` or `bot_from_profile()` for simple bot creation without touching `ServiceRegistry`.

