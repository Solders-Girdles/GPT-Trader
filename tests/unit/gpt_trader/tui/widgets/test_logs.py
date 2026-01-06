"""Tests for LogWidget filter functionality."""

import logging
from unittest.mock import MagicMock, PropertyMock, patch

from gpt_trader.tui.widgets.logs import LogWidget


class TestLogWidgetFilterChips:
    """Tests for LogWidget level filter chip functionality."""

    def test_filter_bindings_present(self):
        """Test 1-5 and f/F keybinds for filters are present."""
        widget = LogWidget()
        binding_keys = [b.key for b in widget.BINDINGS]

        # Number keys for direct filter selection
        for key in ["1", "2", "3", "4", "5"]:
            assert key in binding_keys, f"Key '{key}' not in bindings"

        # Cycle/clear filter keys
        assert "f" in binding_keys  # Cycle filter
        assert "F" in binding_keys  # Clear filter

    def test_action_filter_all_sets_debug_level(self):
        """Test '1' key sets DEBUG level (show all)."""
        widget = LogWidget()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(widget, "_set_level_filter") as mock_set:
                widget.action_filter_all()
                mock_set.assert_called_once_with(logging.DEBUG, "all")

    def test_action_filter_error_sets_error_level(self):
        """Test '2' key sets ERROR level."""
        widget = LogWidget()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(widget, "_set_level_filter") as mock_set:
                widget.action_filter_error()
                mock_set.assert_called_once_with(logging.ERROR, "error")

    def test_action_filter_warning_sets_warning_level(self):
        """Test '3' key sets WARNING level."""
        widget = LogWidget()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(widget, "_set_level_filter") as mock_set:
                widget.action_filter_warning()
                mock_set.assert_called_once_with(logging.WARNING, "warn")

    def test_action_filter_info_sets_info_level(self):
        """Test '4' key sets INFO level."""
        widget = LogWidget()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(widget, "_set_level_filter") as mock_set:
                widget.action_filter_info()
                mock_set.assert_called_once_with(logging.INFO, "info")

    def test_action_filter_debug_sets_debug_level(self):
        """Test '5' key sets DEBUG level."""
        widget = LogWidget()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(widget, "_set_level_filter") as mock_set:
                widget.action_filter_debug()
                mock_set.assert_called_once_with(logging.DEBUG, "debug")


class TestLogWidgetCycleFilter:
    """Tests for LogWidget filter cycling (aligned with AlertHistory pattern)."""

    def test_cycle_level_filter_cycles_through_levels(self):
        """Test 'f' key cycles through filter levels."""
        widget = LogWidget()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(widget, "_set_level_filter") as mock_set:
                # Start at info, should cycle to warn
                widget._current_level_filter = "info"
                widget.action_cycle_level_filter()
                mock_set.assert_called_once_with(logging.WARNING, "warn")

    def test_cycle_level_filter_wraps_around(self):
        """Test filter cycling wraps from last to first."""
        widget = LogWidget()
        mock_app = MagicMock()
        mock_app.notify = MagicMock()

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(widget, "_set_level_filter") as mock_set:
                # At 'all' (last), should wrap to 'info'
                widget._current_level_filter = "all"
                widget.action_cycle_level_filter()
                mock_set.assert_called_once_with(logging.INFO, "info")

    def test_clear_filters_binding_exists(self):
        """Test 'F' key binding exists for clear filters."""
        widget = LogWidget()
        binding_keys = [b.key for b in widget.BINDINGS]
        assert "F" in binding_keys

        # Verify the action method exists and is callable
        assert hasattr(widget, "action_clear_filters")
        assert callable(widget.action_clear_filters)

    def test_clear_filters_resets_state(self):
        """Test action_clear_filters sets expected state values."""
        widget = LogWidget()
        widget._min_level = logging.ERROR
        widget._logger_filter = "some_logger"
        widget._current_level_filter = "error"

        mock_app = MagicMock()
        mock_app.notify = MagicMock()

        # Create mock handler
        mock_handler = MagicMock()

        with patch.object(type(widget), "app", new_callable=PropertyMock, return_value=mock_app):
            with patch.object(widget, "_update_level_chips"):
                with patch.object(widget, "query_one") as mock_query:
                    # Set up query_one to return mocks for different queries
                    mock_input = MagicMock()
                    mock_log_stream = MagicMock()
                    mock_query.side_effect = [mock_input, mock_log_stream, mock_log_stream]

                    # Patch the import inside the method
                    with patch(
                        "gpt_trader.tui.widgets.logs.get_tui_log_handler",
                        create=True,
                    ) as mock_get_handler:
                        mock_get_handler.return_value = mock_handler
                        # Import and patch directly in the module's namespace
                        import gpt_trader.tui.widgets.logs as logs_module

                        original = getattr(logs_module, "get_tui_log_handler", None)
                        try:
                            # Manually inject the mock
                            logs_module.get_tui_log_handler = lambda: mock_handler
                            widget.action_clear_filters()
                        finally:
                            if original:
                                logs_module.get_tui_log_handler = original

        # Verify state was reset
        assert widget._min_level == logging.INFO
        assert widget._logger_filter == ""
        assert widget._current_level_filter == "info"


class TestLogWidgetSetLevel:
    """Tests for LogWidget.set_level method."""

    def test_set_level_delegates_to_set_level_filter(self):
        """Test set_level uses _set_level_filter internally."""
        widget = LogWidget()

        with patch.object(widget, "_set_level_filter") as mock_set:
            widget.set_level(logging.WARNING)
            mock_set.assert_called_once_with(logging.WARNING, "warn")

    def test_set_level_maps_all_standard_levels(self):
        """Test set_level maps all standard logging levels correctly."""
        widget = LogWidget()

        level_map = [
            (logging.DEBUG, "debug"),
            (logging.INFO, "info"),
            (logging.WARNING, "warn"),
            (logging.ERROR, "error"),
        ]

        for level, expected_name in level_map:
            with patch.object(widget, "_set_level_filter") as mock_set:
                widget.set_level(level)
                mock_set.assert_called_once_with(level, expected_name)
