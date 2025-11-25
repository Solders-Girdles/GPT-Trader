"""Tests to verify backward compatibility of the StateManager implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.engines.runtime import RuntimeEngine
from gpt_trader.orchestration.config_controller import ConfigController
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.service_registry import ServiceRegistry
from gpt_trader.persistence.event_store import EventStore


class TestReduceOnlyBackwardCompatibility:
    """Tests to ensure backward compatibility is maintained."""

    @pytest.mark.skip(
        reason="TODO: Fix StateManager fallback - RuntimeEngine needs config_controller=None handling"
    )
    def test_runtime_coordinator_without_state_manager(self) -> None:
        """Test that RuntimeEngine works without StateManager."""
        event_store = EventStore()
        config = BotConfig(symbols=["BTC-USD"])

        registry = ServiceRegistry(
            config=config,
            event_store=event_store,
        )

        from gpt_trader.features.live_trade.engines.base import CoordinatorContext

        context = CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            symbols=(),
            bot_id="test",
        )

        runtime_coordinator = RuntimeEngine(
            context,
            config_controller=None,
        )

        # Test basic functionality
        assert runtime_coordinator.is_reduce_only_mode() is False

        # Change state
        runtime_coordinator.set_reduce_only_mode(True, "test")
        assert runtime_coordinator.is_reduce_only_mode() is True

    @pytest.mark.skip(
        reason="TODO: Fix StateManager fallback - mixed ConfigController/RuntimeEngine interaction"
    )
    def test_mixed_environment_compatibility(self) -> None:
        """Test that components work in a mixed environment with and without StateManager."""
        event_store = EventStore()
        config = BotConfig(symbols=["BTC-USD"])

        # Create config controller with StateManager
        state_manager = MagicMock()  # Mocking StateManager
        config_controller = ConfigController(
            config,
            reduce_only_state_manager=state_manager,
        )

        # Create runtime coordinator without StateManager
        registry = ServiceRegistry(
            config=config,
            event_store=event_store,
        )

        from gpt_trader.features.live_trade.engines.base import CoordinatorContext

        context = CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            symbols=(),
            bot_id="test",
        )

        runtime_coordinator = RuntimeEngine(
            context,
            config_controller=config_controller,
        )

        # Both should work independently
        assert config_controller.is_reduce_only_mode() is False
        assert runtime_coordinator.is_reduce_only_mode() is False

        # Change through config controller
        config_controller.set_reduce_only_mode(
            enabled=True,
            reason="test",
            risk_manager=None,
        )

        # Config controller should see the change
        assert config_controller.is_reduce_only_mode() is True

        # Runtime coordinator should still work (fallback behavior)
        runtime_coordinator.set_reduce_only_mode(False, "runtime_test")
        assert runtime_coordinator.is_reduce_only_mode() is False
