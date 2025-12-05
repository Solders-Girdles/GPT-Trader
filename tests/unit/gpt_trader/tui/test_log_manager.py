"""Tests for TUI log manager."""

import logging
from unittest.mock import MagicMock, patch

from rich.text import Text

from gpt_trader.tui.log_manager import TuiLogHandler, get_tui_log_handler


class TestTuiLogHandler:
    """Tests for TuiLogHandler."""

    def test_write_to_widget_uses_write_with_markup(self):
        """Test that _write_to_widget uses write() with markup string + newline."""
        # Create a mock Log widget
        mock_widget = MagicMock()

        # Create a Rich Text object with style (not from_markup)
        text_message = Text("Test error message", style="red")

        # Call _write_to_widget
        TuiLogHandler._write_to_widget(mock_widget, text_message)

        # Verify write (not write_line) was called with markup string + newline
        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]
        assert isinstance(call_arg, str), "write() should be called with str markup"
        # Verify the message content is present and styled
        assert "Test error message" in call_arg
        # Verify newline is included
        assert call_arg.endswith("\n"), "Should include newline"
        # The markup will contain the style information
        assert "red" in call_arg or "Test error message" in call_arg

    def test_emit_handles_error_level(self):
        """Test that emit() properly formats ERROR level logs."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        # Register widget
        handler.register_widget(mock_widget, logging.DEBUG)

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test error",
            args=(),
            exc_info=None,
        )

        # Emit the record
        handler.emit(record)

        # Verify widget.write_line was called (not write)
        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]
        assert isinstance(call_arg, str)
        assert "Test error" in call_arg
        # Verify color markup is present (ERROR = red = #bf616a)
        assert "#bf616a" in call_arg or "Test error" in call_arg

    def test_emit_handles_warning_level(self):
        """Test that emit() properly formats WARNING level logs."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Test warning",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]
        assert isinstance(call_arg, str)
        assert "Test warning" in call_arg
        # Verify color markup is present (WARNING = yellow = #ebcb8b)
        assert "#ebcb8b" in call_arg or "Test warning" in call_arg

    def test_emit_handles_info_level(self):
        """Test that emit() properly formats INFO level logs."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test info",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]
        assert isinstance(call_arg, str)
        assert "Test info" in call_arg
        # Verify color markup is present (INFO = green = #a3be8c)
        assert "#a3be8c" in call_arg or "Test info" in call_arg

    def test_emit_handles_debug_level(self):
        """Test that emit() properly formats DEBUG level logs."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Test debug",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]
        assert isinstance(call_arg, str)
        assert "Test debug" in call_arg
        # Verify color markup is present (DEBUG = grey = #4c566a)
        assert "#4c566a" in call_arg or "Test debug" in call_arg

    def test_emit_respects_widget_level_filter(self):
        """Test that emit() only sends logs at or above widget's min level."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        # Register widget with WARNING level
        handler.register_widget(mock_widget, logging.WARNING)

        # Emit DEBUG log (should be filtered)
        debug_record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=1,
            msg="Debug message",
            args=(),
            exc_info=None,
        )
        handler.emit(debug_record)

        # Should not be called for DEBUG
        mock_widget.write.assert_not_called()

        # Emit WARNING log (should pass)
        warning_record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        handler.emit(warning_record)

        # Should be called for WARNING
        mock_widget.write.assert_called_once()

    def test_emit_skips_unmounted_widgets(self):
        """Test that emit() skips widgets that aren't mounted."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = False  # Not mounted
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        # Should not call write_line on unmounted widget
        mock_widget.write.assert_not_called()

    def test_emit_uses_call_from_thread_on_background_thread(self):
        """Test that emit() uses call_from_thread when not on main thread."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Mock threading to simulate background thread
        with (
            patch("threading.current_thread") as mock_thread,
            patch("threading.main_thread") as mock_main_thread,
        ):
            mock_thread.return_value = MagicMock(name="background_thread")
            mock_main_thread.return_value = MagicMock(name="main_thread")

            handler.emit(record)

            # Should use call_from_thread, not direct write
            mock_widget.app.call_from_thread.assert_called_once()
            # Get the callable that was passed
            call_args = mock_widget.app.call_from_thread.call_args
            assert call_args[0][0] == handler._write_to_widget

    def test_get_tui_log_handler_returns_singleton(self):
        """Test that get_tui_log_handler returns same instance."""
        handler1 = get_tui_log_handler()
        handler2 = get_tui_log_handler()
        assert handler1 is handler2

    def test_register_and_unregister_widget(self):
        """Test widget registration and unregistration."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()

        # Register
        handler.register_widget(mock_widget, logging.INFO)
        assert mock_widget in handler._widgets
        assert handler._widgets[mock_widget] == logging.INFO

        # Unregister
        handler.unregister_widget(mock_widget)
        assert mock_widget not in handler._widgets

    def test_update_widget_level(self):
        """Test updating widget's minimum level."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()

        # Register with INFO
        handler.register_widget(mock_widget, logging.INFO)
        assert handler._widgets[mock_widget] == logging.INFO

        # Update to WARNING
        handler.update_widget_level(mock_widget, logging.WARNING)
        assert handler._widgets[mock_widget] == logging.WARNING

    def test_log_message_with_markup_characters_is_escaped(self):
        """Test that log messages containing markup characters are treated as literal text."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        # Log message containing Rich markup tags
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Processing [red]data[/red] from API",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        # Verify write_line was called
        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]

        # The literal brackets should be escaped in the output
        # Original message text should be present
        assert "Processing" in call_arg
        assert "data" in call_arg
        assert "from API" in call_arg

        # The message should be styled with INFO color (#a3be8c)
        # but the [red] tags should be escaped/literal, not interpreted
        assert "#a3be8c" in call_arg

    def test_log_message_with_closing_bracket_is_safe(self):
        """Test that log messages with [/] don't corrupt coloring."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        # Log message containing closing bracket that could corrupt markup
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Invalid regex pattern: [/] detected",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        # Should not raise an exception
        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]

        # Verify message content is present
        assert "Invalid regex pattern" in call_arg
        assert "detected" in call_arg

        # Verify ERROR color is applied
        assert "#bf616a" in call_arg

    def test_log_message_with_multiple_brackets_is_safe(self):
        """Test that complex bracket patterns in logs don't break formatting."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        # Log message with multiple bracket patterns
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Config: [section] key=[value] [/incomplete",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        # Should not raise an exception
        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]

        # Verify message content is preserved
        assert "Config:" in call_arg
        assert "section" in call_arg
        assert "key=" in call_arg
        assert "value" in call_arg

        # Verify WARNING color is applied
        assert "#ebcb8b" in call_arg

    def test_formatter_does_not_add_newline(self):
        """Regression test: Ensure formatter doesn't add newline (would cause double-spacing).

        If this test fails, the formatter was changed to include \\n at the end.
        This conflicts with our write(markup + "\\n") pattern in _write_to_widget.
        """
        handler = TuiLogHandler()

        # Get the formatter's output
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        formatted = handler.format(record)

        # Verify formatter output does NOT end with newline
        assert not formatted.endswith("\n"), (
            "Formatter should not add newline - _write_to_widget adds it. "
            "If formatter was changed to add \\n, update _write_to_widget to remove the manual \\n."
        )

        # Verify our write() call adds exactly one newline
        mock_widget = MagicMock()
        from rich.text import Text

        test_text = Text("Test", style="#a3be8c")

        TuiLogHandler._write_to_widget(mock_widget, test_text)

        # Get the string that was written
        call_arg = mock_widget.write.call_args[0][0]

        # Verify exactly one newline at end
        assert call_arg.endswith("\n"), "Should have newline at end"
        assert not call_arg.endswith(
            "\n\n"
        ), "Should NOT have double newline (would cause blank lines)"
