"""Tests for the orchestration service registry scaffolding."""

from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.service_registry import ServiceRegistry, empty_registry


def test_empty_registry_initialises_with_config():
    cfg = BotConfig.from_profile("dev")
    registry = empty_registry(cfg)
    assert registry.config is cfg
    assert registry.event_store is None
    assert registry.orders_store is None


def test_with_updates_returns_new_instance():
    cfg = BotConfig.from_profile("dev")
    registry = empty_registry(cfg)
    updated = registry.with_updates(extras={"key": "value"})
    assert updated is not registry
    assert updated.extras == {"key": "value"}
    # Original registry remains unchanged (frozen dataclass semantics)
    assert registry.extras == {}
