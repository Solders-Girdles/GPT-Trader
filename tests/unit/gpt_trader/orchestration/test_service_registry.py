"""Tests for the orchestration service registry scaffolding.

LEGACY: ServiceRegistry is deprecated and scheduled for removal in v3.0.
These tests are marked with @pytest.mark.legacy for easy identification
during the removal process. See docs/MIGRATION_STATUS.md for details.
"""

import pytest

from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.service_registry import empty_registry

# Mark entire module as legacy for v3.0 removal
pytestmark = pytest.mark.legacy


def test_empty_registry_initialises_with_config():
    config = BotConfig.from_profile("dev")
    registry = empty_registry(config)
    assert registry.config is config
    assert registry.event_store is None
    assert registry.orders_store is None


def test_with_updates_returns_new_instance():
    config = BotConfig.from_profile("dev")
    registry = empty_registry(config)
    updated = registry.with_updates(extras={"key": "value"})
    assert updated is not registry
    assert updated.extras == {"key": "value"}
    # Original registry remains unchanged (frozen dataclass semantics)
    assert registry.extras == {}
