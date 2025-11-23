"""Tests to verify backward compatibility of the StateManager implementation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.engines.runtime import RuntimeEngine
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.persistence.event_store import EventStore


class TestReduceOnlyBackwardCompatibility:
    """Tests to ensure backward compatibility is maintained."""

    def test_config_controller_without_state_manager(self) -> None:
        """Test that ConfigController works without StateManager."""
        config = BotConfig.from_profile("dev")

        # Create config controller without StateManager
        config_controller = ConfigController(config)

        # Test basic functionality
        assert config_controller.is_reduce_only_mode() is False

        # Change state
        changed = config_controller.set_reduce_only_mode(
            enabled=True,
            reason="test",
            risk_manager=None,
        )

        assert changed is True
        assert config_controller.is_reduce_only_mode() is True

        # Apply risk update
        changed = config_controller.apply_risk_update(False)
        assert changed is True
        assert config_controller.is_reduce_only_mode() is False

    @pytest.mark.xfail(reason="State Manager fallback logic needs update")
    def test_runtime_coordinator_without_state_manager(self) -> None:
        """Test that RuntimeEngine works without StateManager."""
        event_store = EventStore()
        config = BotConfig.from_profile("dev")

        registry = ServiceRegistry(
            config=config,
            event_store=event_store,
        )

        from bot_v2.orchestration.engines.base import CoordinatorContext

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

    @pytest.mark.xfail(reason="State Manager fallback logic needs update")
    def test_mixed_environment_compatibility(self) -> None:
        """Test that components work in a mixed environment with and without StateManager."""
        event_store = EventStore()
        config = BotConfig.from_profile("dev")

        # Create config controller with StateManager
        from bot_v2.orchestration.state.unified_state import create_reduce_only_state_manager

        state_manager = create_reduce_only_state_manager(event_store)
        config_controller = ConfigController(
            config,
            reduce_only_state_manager=state_manager,
        )

        # Create runtime coordinator without StateManager
        registry = ServiceRegistry(
            config=config,
            event_store=event_store,
        )

        from bot_v2.orchestration.engines.base import CoordinatorContext

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

    def test_legacy_api_compatibility(self) -> None:
        """Test that legacy APIs still work as expected."""
        config = BotConfig.from_profile("dev")

        # Create components without StateManager
        config_controller = ConfigController(config)

        # Test all legacy methods
        assert hasattr(config_controller, "reduce_only_mode")
        assert hasattr(config_controller, "set_reduce_only_mode")
        assert hasattr(config_controller, "is_reduce_only_mode")
        assert hasattr(config_controller, "apply_risk_update")
        assert hasattr(config_controller, "sync_with_risk_manager")

        # Test that they work
        assert config_controller.reduce_only_mode is False

        changed = config_controller.set_reduce_only_mode(
            enabled=True,
            reason="test",
            risk_manager=None,
        )
        assert changed is True
        assert config_controller.reduce_only_mode is True

        # Test sync with risk manager
        risk_manager = MagicMock()
        risk_manager.is_reduce_only_mode.return_value = False
        config_controller.sync_with_risk_manager(risk_manager)

        # Should have called the risk manager
        risk_manager.is_reduce_only_mode.assert_called_once()

    @pytest.mark.xfail(reason="Property access compatibility mismatch")
    def test_property_access_compatibility(self) -> None:
        """Test that property access patterns still work."""
        event_store = EventStore()
        config = BotConfig.from_profile("dev")

        # Create config controller with StateManager
        from bot_v2.orchestration.state.unified_state import create_reduce_only_state_manager

        state_manager = create_reduce_only_state_manager(event_store)
        config_controller = ConfigController(
            config,
            reduce_only_state_manager=state_manager,
        )

        # Test property access
        assert hasattr(config_controller, "reduce_only_mode")
        assert isinstance(config_controller.reduce_only_mode, bool)

        # Change state
        config_controller.set_reduce_only_mode(
            enabled=True,
            reason="test",
            risk_manager=None,
        )

        # Property should reflect the change
        assert config_controller.reduce_only_mode is True

    @pytest.mark.xfail(reason="Method signature compatibility mismatch")
    def test_method_signature_compatibility(self) -> None:
        """Test that method signatures remain compatible."""
        event_store = EventStore()
        config = BotConfig.from_profile("dev")

        # Create config controller with StateManager
        from bot_v2.orchestration.state.unified_state import create_reduce_only_state_manager

        state_manager = create_reduce_only_state_manager(event_store)
        config_controller = ConfigController(
            config,
            reduce_only_state_manager=state_manager,
        )

        # Test method signatures
        # set_reduce_only_mode should accept enabled, reason, and risk_manager
        result = config_controller.set_reduce_only_mode(
            enabled=True,
            reason="test",
            risk_manager=None,
        )
        assert isinstance(result, bool)

        # is_reduce_only_mode should accept optional risk_manager
        result = config_controller.is_reduce_only_mode(risk_manager=None)
        assert isinstance(result, bool)

        # apply_risk_update should accept enabled
        result = config_controller.apply_risk_update(enabled=False)
        assert isinstance(result, bool)
