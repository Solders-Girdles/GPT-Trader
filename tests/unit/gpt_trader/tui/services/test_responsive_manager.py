"""Tests for ResponsiveManager."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.tui.services.responsive_manager import ResponsiveManager


class TestResponsiveManager:
    """Test ResponsiveManager functionality."""

    def test_init_sets_defaults(self):
        """Test that initialization sets default values."""
        mock_app = MagicMock()
        manager = ResponsiveManager(mock_app)

        assert manager.current_state == ResponsiveState.STANDARD
        assert manager.current_width == 120
        assert manager._resize_timer is None

    def test_initialize_sets_state_from_width(self):
        """Test initialize calculates state from width."""
        mock_app = MagicMock()
        manager = ResponsiveManager(mock_app)

        # Test compact width
        state = manager.initialize(80)
        assert state == ResponsiveState.COMPACT
        assert manager.current_width == 80

    def test_initialize_returns_comfortable_for_wide_terminal(self):
        """Test initialize returns comfortable for wide terminals."""
        mock_app = MagicMock()
        manager = ResponsiveManager(mock_app)

        state = manager.initialize(160)
        assert state in (ResponsiveState.COMFORTABLE, ResponsiveState.WIDE)

    def test_handle_resize_schedules_timer(self):
        """Test handle_resize schedules a debounced timer."""
        mock_app = MagicMock()
        mock_timer = MagicMock()
        mock_app.set_timer.return_value = mock_timer

        manager = ResponsiveManager(mock_app)
        manager.handle_resize(100)

        mock_app.set_timer.assert_called_once()
        # First argument should be delay, second should be callback
        assert mock_app.set_timer.call_args[0][0] == 0.1

    def test_handle_resize_cancels_pending_timer(self):
        """Test handle_resize cancels any pending timer."""
        mock_app = MagicMock()
        mock_timer = MagicMock()
        mock_app.set_timer.return_value = mock_timer

        manager = ResponsiveManager(mock_app)
        manager._resize_timer = mock_timer

        manager.handle_resize(100)

        mock_timer.stop.assert_called_once()

    def test_propagate_to_screen_sets_responsive_state(self):
        """Test propagate_to_screen sets state on screen."""
        mock_app = MagicMock()
        mock_screen = MagicMock()
        mock_screen.responsive_state = ResponsiveState.STANDARD
        mock_app.screen = mock_screen

        manager = ResponsiveManager(mock_app)
        manager.current_state = ResponsiveState.COMFORTABLE
        manager.propagate_to_screen()

        assert mock_screen.responsive_state == ResponsiveState.COMFORTABLE

    def test_propagate_to_screen_handles_no_screen(self):
        """Test propagate_to_screen handles missing screen gracefully."""
        mock_app = MagicMock()
        mock_app.screen = None

        manager = ResponsiveManager(mock_app)

        # Should not raise
        manager.propagate_to_screen()
        assert manager.current_state == ResponsiveState.STANDARD

    def test_cleanup_stops_timer(self):
        """Test cleanup stops any pending timer."""
        mock_app = MagicMock()
        mock_timer = MagicMock()

        manager = ResponsiveManager(mock_app)
        manager._resize_timer = mock_timer

        manager.cleanup()

        mock_timer.stop.assert_called_once()
        assert manager._resize_timer is None

    def test_cleanup_handles_no_timer(self):
        """Test cleanup handles no pending timer gracefully."""
        mock_app = MagicMock()
        manager = ResponsiveManager(mock_app)

        # Should not raise
        manager.cleanup()
        assert manager._resize_timer is None
