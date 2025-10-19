from __future__ import annotations

import pytest

import bot_v2.orchestration.runtime_settings as runtime_settings_module
from bot_v2.config.types import Profile
from bot_v2.orchestration.configuration.core import BotConfig
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.storage import StorageBootstrapper
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore


def _make_config(profile: Profile = Profile.DEV, **overrides) -> BotConfig:
    payload: dict[str, object] = {"profile": profile}
    if profile == Profile.SPOT and "max_leverage" not in overrides:
        payload["max_leverage"] = 1
    payload.update(overrides)
    return BotConfig(**payload)


def test_bootstrapper_creates_context_with_defaults(
    patched_runtime_settings,
) -> None:
    config = _make_config()
    registry = ServiceRegistry(config=config)

    bootstrapper = StorageBootstrapper(config=config, registry=registry)
    context = bootstrapper.bootstrap()

    expected_storage_root = (
        patched_runtime_settings.runtime_root / f"perps_bot/{config.profile.value}"
    )
    assert context.storage_dir == expected_storage_root
    assert context.orders_store.storage_path == expected_storage_root

    assert context.registry.runtime_settings is patched_runtime_settings
    assert context.event_store_root == expected_storage_root
    assert context.event_store.path.parent == expected_storage_root


def test_bootstrapper_respects_event_store_override(
    monkeypatch: pytest.MonkeyPatch,
    runtime_settings_factory,
) -> None:
    config = _make_config(Profile.SPOT)
    settings = runtime_settings_factory(event_store_override=True)

    real_loader = runtime_settings_module.load_runtime_settings
    monkeypatch.setattr(
        runtime_settings_module,
        "load_runtime_settings",
        lambda env=None: settings if env is None else real_loader(env),
    )
    monkeypatch.setattr(
        "bot_v2.orchestration.storage.load_runtime_settings",
        lambda: settings,
    )

    registry = ServiceRegistry(config=config)
    bootstrapper = StorageBootstrapper(config=config, registry=registry)

    context = bootstrapper.bootstrap()

    expected_root = settings.event_store_root_override
    assert expected_root is not None
    if "perps_bot" not in expected_root.parts:
        expected_root = expected_root / "perps_bot" / config.profile.value
    assert context.event_store_root == expected_root
    assert context.event_store.path.parent == expected_root


def test_bootstrapper_reuses_registry_components(runtime_settings_factory, tmp_path) -> None:
    config = _make_config(Profile.DEMO)
    storage_path = tmp_path / "storage"
    storage_path.mkdir()
    orders_store = OrdersStore(storage_path=storage_path)

    event_root = tmp_path / "events"
    event_root.mkdir()
    event_store = EventStore(root=event_root)

    registry = ServiceRegistry(
        config=config,
        runtime_settings=runtime_settings_factory(),
        orders_store=orders_store,
        event_store=event_store,
    )

    bootstrapper = StorageBootstrapper(config=config, registry=registry)
    context = bootstrapper.bootstrap()

    assert context.orders_store is orders_store
    assert context.event_store is event_store
