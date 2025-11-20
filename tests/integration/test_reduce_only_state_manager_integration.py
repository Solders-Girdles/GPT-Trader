"""Integration tests for the ReduceOnlyModeStateManager."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.coordinators.runtime import RuntimeCoordinator
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.state.unified_state import (
    ReduceOnlyModeSource,
    SystemState,
)
from bot_v2.persistence.event_store import EventStore


class TestReduceOnlyStateManagerIntegration:
    """Integration tests for the StateManager with other components."""

    def test_state_manager_with_config_controller(self) -> None:
        """Test StateManager integration with ConfigController."""
        event_store = EventStore()
        state_manager = create_reduce_only_state_manager(
            event_store=event_store,
            initial_state=False,
        )

        config = BotConfig.from_profile("dev")
        config_controller = ConfigController(
            config,
            reduce_only_state_manager=state_manager,
        )

        # Test initial state
        assert config_controller.is_reduce_only_mode() is False
        assert state_manager.is_reduce_only_mode is False

        # Change state through config controller
        changed = config_controller.set_reduce_only_mode(
            enabled=True,
            reason="test",
            risk_manager=None,
        )

        assert changed is True
        assert config_controller.is_reduce_only_mode() is True
        assert state_manager.is_reduce_only_mode is True

        # Verify audit log
        audit_log = state_manager.get_audit_log()
        assert len(audit_log) == 1
        assert audit_log[0].source == ReduceOnlyModeSource.CONFIG
        assert audit_log[0].reason == "test"

    def test_state_manager_with_runtime_coordinator(self) -> None:
        """Test StateManager integration with RuntimeCoordinator."""
        event_store = EventStore()
        state_manager = create_reduce_only_state_manager(
            event_store=event_store,
            initial_state=False,
        )

        config = BotConfig.from_profile("dev")
        registry = ServiceRegistry(
            config=config,
            event_store=event_store,
            reduce_only_state_manager=state_manager,
        )

        # Create a mock coordinator context
        from bot_v2.orchestration.coordinators.base import CoordinatorContext

        context = CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            symbols=(),
            bot_id="test",
        )

        runtime_coordinator = RuntimeCoordinator(
            context,
            config_controller=None,
        )

        # Test initial state
        assert runtime_coordinator.is_reduce_only_mode() is False
        assert state_manager.is_reduce_only_mode is False

        # Change state through runtime coordinator
        runtime_coordinator.set_reduce_only_mode(True, "test")

        assert runtime_coordinator.is_reduce_only_mode() is True
        assert state_manager.is_reduce_only_mode is True

        # Verify audit log
        audit_log = state_manager.get_audit_log()
        assert len(audit_log) == 1
        assert audit_log[0].source == ReduceOnlyModeSource.RUNTIME_COORDINATOR
        assert audit_log[0].reason == "test"

    def test_multiple_components_using_state_manager(self) -> None:
        """Test multiple components using the same StateManager."""
        event_store = EventStore()
        state_manager = create_reduce_only_state_manager(
            event_store=event_store,
            initial_state=False,
        )

        config = BotConfig.from_profile("dev")

        # Create components
        config_controller = ConfigController(
            config,
            reduce_only_state_manager=state_manager,
        )

        registry = ServiceRegistry(
            config=config,
            event_store=event_store,
            reduce_only_state_manager=state_manager,
        )

        from bot_v2.orchestration.coordinators.base import CoordinatorContext

        context = CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            symbols=(),
            bot_id="test",
        )

        runtime_coordinator = RuntimeCoordinator(
            context,
            config_controller=config_controller,
        )

        # Change state through config controller
        config_controller.set_reduce_only_mode(
            enabled=True,
            reason="config_change",
            risk_manager=None,
        )

        # Verify all components see the change
        assert config_controller.is_reduce_only_mode() is True
        assert runtime_coordinator.is_reduce_only_mode() is True
        assert state_manager.is_reduce_only_mode is True

        # Change state through runtime coordinator
        runtime_coordinator.set_reduce_only_mode(False, "runtime_change")

        # Verify all components see the change
        assert config_controller.is_reduce_only_mode() is False
        assert runtime_coordinator.is_reduce_only_mode() is False
        assert state_manager.is_reduce_only_mode is False

        # Verify audit log has both changes
        audit_log = state_manager.get_audit_log()
        assert len(audit_log) == 2
        assert audit_log[1].source == ReduceOnlyModeSource.CONFIG
        assert audit_log[1].reason == "config_change"
        assert audit_log[0].source == ReduceOnlyModeSource.RUNTIME_COORDINATOR
        assert audit_log[0].reason == "runtime_change"

    def test_state_listeners_notification(self) -> None:
        """Test that state listeners are properly notified."""
        event_store = EventStore()
        state_manager = create_reduce_only_state_manager(
            event_store=event_store,
            initial_state=False,
        )

        # Add a listener
        listener_calls = []

        def test_listener(state) -> None:
            listener_calls.append(state)

        state_manager.add_state_listener(test_listener)

        # Change state
        state_manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        # Verify listener was called
        assert len(listener_calls) == 1
        assert listener_calls[0].enabled is True
        assert listener_calls[0].reason == "test"

    def test_backward_compatibility_fallback(self) -> None:
        """Test that components work without StateManager (backward compatibility)."""
        # Create config controller without StateManager
        config = BotConfig.from_profile("dev")
        config_controller = ConfigController(config)

        # Test that it still works
        assert config_controller.is_reduce_only_mode() is False

        changed = config_controller.set_reduce_only_mode(
            enabled=True,
            reason="test",
            risk_manager=None,
        )

        assert changed is True
        assert config_controller.is_reduce_only_mode() is True

    def test_state_persistence_through_event_store(self) -> None:
        """Test that state changes are persisted to the event store."""
        event_store = EventStore()
        state_manager = create_reduce_only_state_manager(
            event_store=event_store,
            initial_state=False,
        )

        # Change state
        state_manager.set_reduce_only_mode(
            enabled=True,
            reason="test",
            source=ReduceOnlyModeSource.CONFIG,
        )

        # Verify event was added to event store
        events = event_store.get_events(limit=10)
        state_change_events = [
            e for e in events if e.get("event_type") == "reduce_only_mode_changed"
        ]

        assert len(state_change_events) > 0
        assert state_change_events[-1]["enabled"] is True
        assert state_change_events[-1]["reason"] == "test"
        assert state_change_events[-1]["source"] == "config"

    def test_validation_integration(self) -> None:
        """Test that validation works in the integrated context."""
        event_store = EventStore()
        state_manager = create_reduce_only_state_manager(
            event_store=event_store,
            initial_state=False,
            validation_enabled=True,
        )

        config = BotConfig.from_profile("dev")
        config_controller = ConfigController(
            config,
            reduce_only_state_manager=state_manager,
        )

        # Test validation with empty reason (should fail)
        with pytest.raises(ValueError):
            config_controller.set_reduce_only_mode(
                enabled=True,
                reason="",
                risk_manager=None,
            )

        # Verify state didn't change
        assert config_controller.is_reduce_only_mode() is False
        assert state_manager.is_reduce_only_mode is False

    def test_critical_source_logging(self) -> None:
        """Test that critical sources are logged as warnings."""
        event_store = EventStore()
        state_manager = create_reduce_only_state_manager(
            event_store=event_store,
            initial_state=False,
        )

        with patch("bot_v2.orchestration.state_manager.logger") as mock_logger:
            # Change state with critical source
            state_manager.set_reduce_only_mode(
                enabled=True,
                reason="guard_failure",
                source=ReduceOnlyModeSource.GUARD_FAILURE,
            )

            # Verify warning was logged
            mock_logger.warning.assert_called()

            # Check the call arguments
            call_args = mock_logger.warning.call_args
            assert "guard_failure" in str(call_args)
            assert "Reduce-only mode enabled" in str(call_args)

    def test_metadata_handling_in_integration(self) -> None:
        """Test that metadata is properly handled in integration."""
        event_store = EventStore()
        state_manager = create_reduce_only_state_manager(
            event_store=event_store,
            initial_state=False,
        )

        # Change state with metadata
        metadata = {"loss_amount": "1000", "symbol": "BTC-PERP"}
        state_manager.set_reduce_only_mode(
            enabled=True,
            reason="daily_loss_limit",
            source=ReduceOnlyModeSource.DAILY_LOSS_LIMIT,
            metadata=metadata,
        )

        # Verify metadata in audit log
        audit_log = state_manager.get_audit_log()
        assert len(audit_log) == 1
        assert audit_log[0].metadata == metadata

        # Verify metadata in event store
        events = event_store.get_events(limit=10)
        state_change_events = [
            e for e in events if e.get("event_type") == "reduce_only_mode_changed"
        ]

        assert len(state_change_events) > 0
        assert state_change_events[-1]["metadata"] == metadata
