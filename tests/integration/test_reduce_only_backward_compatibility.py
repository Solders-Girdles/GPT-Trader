"""Tests to verify backward compatibility of the reduce-only state management."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import ApplicationContainer
from gpt_trader.features.live_trade.engines.runtime import RuntimeEngine
from gpt_trader.orchestration.config_controller import ConfigController


class TestReduceOnlyBackwardCompatibility:
    """Tests to ensure backward compatibility is maintained."""

    @pytest.mark.skip(
        reason="Test uses parameters (config_controller) not supported by current API - needs redesign"
    )
    def test_runtime_coordinator_without_state_manager(self) -> None:
        """Test that RuntimeEngine works without StateManager."""
        config = BotConfig(symbols=["BTC-USD"], mock_broker=True)
        container = ApplicationContainer(config)

        from gpt_trader.features.live_trade.engines.base import CoordinatorContext

        context = CoordinatorContext(
            config=config,
            container=container,
            event_store=container.event_store,
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
        reason="Test uses parameters (reduce_only_state_manager) not supported by current API - needs redesign"
    )
    def test_mixed_environment_compatibility(self) -> None:
        """Test that components work in a mixed environment with and without StateManager."""
        config = BotConfig(symbols=["BTC-USD"], mock_broker=True)
        container = ApplicationContainer(config)

        # Create config controller with StateManager
        state_manager = MagicMock()  # Mocking StateManager
        config_controller = ConfigController(
            config,
            reduce_only_state_manager=state_manager,
        )

        from gpt_trader.features.live_trade.engines.base import CoordinatorContext

        context = CoordinatorContext(
            config=config,
            container=container,
            event_store=container.event_store,
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
