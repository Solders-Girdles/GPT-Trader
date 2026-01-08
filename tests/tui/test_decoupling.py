"""
Tests for decoupled UICoordinator and StateRegistry observer pattern.

Verifies that UICoordinator no longer depends on specific screen implementations
and that screens properly register/unregister as StateObservers.
"""

from unittest.mock import MagicMock

from gpt_trader.tui.services.state_registry import StateRegistry
from gpt_trader.tui.state import TuiState


class MockObserver:
    """Mock observer for testing StateRegistry."""

    def __init__(self, priority: int = 0):
        self._priority = priority
        self.updates_received: list[TuiState] = []

    @property
    def observer_priority(self) -> int:
        return self._priority

    def on_state_updated(self, state: TuiState) -> None:
        self.updates_received.append(state)


class TestStateRegistryPriority:
    """Tests for StateRegistry observer priority ordering."""

    def test_observers_receive_updates_in_priority_order(self):
        """Higher priority observers should be updated first."""
        registry = StateRegistry()
        state = TuiState()

        # Track call order
        call_order: list[str] = []

        low_priority = MockObserver(priority=0)
        high_priority = MockObserver(priority=100)
        medium_priority = MockObserver(priority=50)

        # Patch on_state_updated to track call order
        def make_tracker(name: str, original):
            def track(s):
                call_order.append(name)
                original(s)

            return track

        low_priority.on_state_updated = make_tracker("low", low_priority.on_state_updated)
        high_priority.on_state_updated = make_tracker("high", high_priority.on_state_updated)
        medium_priority.on_state_updated = make_tracker("medium", medium_priority.on_state_updated)

        # Register in random order
        registry.register(low_priority)
        registry.register(high_priority)
        registry.register(medium_priority)

        # Broadcast
        registry.broadcast(state)

        # Verify order: high (100) -> medium (50) -> low (0)
        assert call_order == ["high", "medium", "low"]

    def test_observer_without_priority_defaults_to_zero(self):
        """Observers without observer_priority should default to 0."""
        registry = StateRegistry()
        state = TuiState()

        class NoPriorityObserver:
            def __init__(self):
                self.received = False

            def on_state_updated(self, state: TuiState) -> None:
                self.received = True

        observer = NoPriorityObserver()
        registry.register(observer)
        registry.broadcast(state)

        assert observer.received is True

    def test_all_observers_receive_update(self):
        """All registered observers should receive updates regardless of priority."""
        registry = StateRegistry()
        state = TuiState()

        observers = [MockObserver(priority=i * 10) for i in range(5)]
        for obs in observers:
            registry.register(obs)

        registry.broadcast(state)

        for obs in observers:
            assert len(obs.updates_received) == 1


class TestUICoordinatorDecoupling:
    """Tests that UICoordinator works without screen dependencies."""

    def test_update_main_screen_works_without_screens_mounted(self):
        """UICoordinator.update_main_screen should not fail if no screens are mounted."""
        from gpt_trader.tui.managers.ui_coordinator import UICoordinator

        # Create mock app with state_registry but no screens
        mock_app = MagicMock()
        mock_app.tui_state = TuiState()
        mock_app.state_registry = StateRegistry()
        mock_app._pulse_heartbeat = MagicMock()

        coordinator = UICoordinator(mock_app)

        # Should not raise any exceptions
        coordinator.update_main_screen()

        # Heartbeat should still be pulsed
        mock_app._pulse_heartbeat.assert_called_once()

    def test_update_main_screen_broadcasts_to_registry(self):
        """UICoordinator should broadcast to StateRegistry."""
        from gpt_trader.tui.managers.ui_coordinator import UICoordinator

        mock_app = MagicMock()
        mock_app.tui_state = TuiState()
        mock_app.state_registry = StateRegistry()
        mock_app._pulse_heartbeat = MagicMock()

        observer = MockObserver()
        mock_app.state_registry.register(observer)

        coordinator = UICoordinator(mock_app)
        coordinator.update_main_screen()

        assert len(observer.updates_received) == 1
        assert observer.updates_received[0] is mock_app.tui_state

    def test_update_main_screen_handles_missing_registry(self):
        """UICoordinator should handle missing state_registry gracefully."""
        from gpt_trader.tui.managers.ui_coordinator import UICoordinator

        mock_app = MagicMock(spec=[])  # No state_registry attribute
        mock_app._pulse_heartbeat = MagicMock()

        coordinator = UICoordinator(mock_app)

        # Should not raise
        coordinator.update_main_screen()


class TestScreenObserverRegistration:
    """Tests that screens properly implement StateObserver pattern."""

    def test_main_screen_has_observer_priority(self):
        """MainScreen should define observer_priority."""
        from gpt_trader.tui.screens.main_screen import MainScreen

        screen = MainScreen()
        assert hasattr(screen, "observer_priority")
        assert screen.observer_priority == 100

    def test_system_details_screen_has_observer_priority(self):
        """SystemDetailsScreen should define observer_priority."""
        from gpt_trader.tui.screens.system_details_screen import SystemDetailsScreen

        screen = SystemDetailsScreen()
        assert hasattr(screen, "observer_priority")
        assert screen.observer_priority == 100

    def test_main_screen_has_on_state_updated(self):
        """MainScreen should implement on_state_updated."""
        from gpt_trader.tui.screens.main_screen import MainScreen

        screen = MainScreen()
        assert hasattr(screen, "on_state_updated")
        assert callable(screen.on_state_updated)

    def test_system_details_screen_has_on_state_updated(self):
        """SystemDetailsScreen should implement on_state_updated."""
        from gpt_trader.tui.screens.system_details_screen import SystemDetailsScreen

        screen = SystemDetailsScreen()
        assert hasattr(screen, "on_state_updated")
        assert callable(screen.on_state_updated)
