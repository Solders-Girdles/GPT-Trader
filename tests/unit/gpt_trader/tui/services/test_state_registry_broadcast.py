"""
Tests for StateRegistry broadcast behavior.

Focus: one observer failing must not block the rest.
"""

from unittest.mock import MagicMock

from gpt_trader.tui.services.state_registry import StateRegistry
from gpt_trader.tui.state import TuiState


class TestStateRegistryBroadcast:
    def test_broadcast_continues_after_observer_exception(self) -> None:
        registry = StateRegistry()

        failing_observer = MagicMock()
        failing_observer.on_state_updated = MagicMock(side_effect=ValueError("boom"))

        working_observer = MagicMock()
        working_observer.on_state_updated = MagicMock()

        registry.register(failing_observer)
        registry.register(working_observer)

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        registry.broadcast(state)

        failing_observer.on_state_updated.assert_called_once_with(state)
        working_observer.on_state_updated.assert_called_once_with(state)
