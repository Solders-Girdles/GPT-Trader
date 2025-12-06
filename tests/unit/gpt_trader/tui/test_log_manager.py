"""Tests for TUI log manager."""

import logging
from unittest.mock import MagicMock, patch

from rich.text import Text

from gpt_trader.tui.log_manager import TuiLogHandler, get_tui_log_handler
from gpt_trader.tui.theme import THEME


class TestTuiLogHandler:
    """Tests for TuiLogHandler."""

    def test_write_to_widget_passes_text(self):
        """_write_to_widget should hand Rich Text straight to the widget."""
        mock_widget = MagicMock()
        text_message = Text("Test error message", style="red")

        TuiLogHandler._write_to_widget(mock_widget, text_message)

        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]
        assert call_arg is text_message
        assert call_arg.plain == "Test error message"
        assert str(call_arg.style) == "red"

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

        # Verify widget.write was called with styled Text
        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]
        assert isinstance(call_arg, Text)
        assert "Test error" in call_arg.plain
        assert str(call_arg.style) == THEME.colors.error

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
        assert isinstance(call_arg, Text)
        assert "Test warning" in call_arg.plain
        assert str(call_arg.style) == THEME.colors.warning

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
        assert isinstance(call_arg, Text)
        assert "Test info" in call_arg.plain
        assert str(call_arg.style) == THEME.colors.success

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
        assert isinstance(call_arg, Text)
        assert "Test debug" in call_arg.plain
        assert str(call_arg.style) == THEME.colors.text_muted

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
            assert call_args[0][1] is mock_widget
            assert isinstance(call_args[0][2], Text)

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
        assert "Processing" in call_arg.plain
        assert "data" in call_arg.plain
        assert "from API" in call_arg.plain

        # The message should be styled with INFO color but markup remains literal
        assert str(call_arg.style) == THEME.colors.success

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
        assert "Invalid regex pattern" in call_arg.plain
        assert "detected" in call_arg.plain

        # Verify ERROR color is applied
        assert str(call_arg.style) == THEME.colors.error

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
        assert "Config:" in call_arg.plain
        assert "section" in call_arg.plain
        assert "key=" in call_arg.plain
        assert "value" in call_arg.plain

        # Verify WARNING color is applied
        assert str(call_arg.style) == THEME.colors.warning

    def test_formatter_output_has_no_extra_newline(self):
        """Ensure formatter output and widget writes don't append stray newlines."""
        handler = TuiLogHandler()

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
        assert not formatted.endswith("\n")

        mock_widget = MagicMock()
        test_text = Text("Test", style=THEME.colors.success)

        TuiLogHandler._write_to_widget(mock_widget, test_text)

        call_arg = mock_widget.write.call_args[0][0]
        assert call_arg is test_text
        assert call_arg.plain == "Test"

    def test_pause_widget_buffers_logs(self):
        """Test that paused widgets buffer logs instead of displaying them."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        # Pause the widget
        handler.pause_widget(mock_widget)
        assert handler.is_widget_paused(mock_widget) is True

        # Emit a log - should be buffered, not written
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Buffered message",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        # Widget.write should not be called for the new log
        # (it may have been called during registration for replay)
        initial_call_count = mock_widget.write.call_count
        handler.emit(record)
        assert mock_widget.write.call_count == initial_call_count

        # Verify buffered count
        assert handler.get_paused_count(mock_widget) >= 1

    def test_resume_widget_flushes_buffered_logs(self):
        """Test that resuming a widget flushes buffered logs."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)

        # Pause and emit some logs
        handler.pause_widget(mock_widget)
        for i in range(3):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        # Verify logs are buffered
        assert handler.get_paused_count(mock_widget) == 3

        # Resume widget
        handler.resume_widget(mock_widget)

        # Widget should no longer be paused
        assert handler.is_widget_paused(mock_widget) is False

        # Buffered logs should have been flushed
        assert handler.get_paused_count(mock_widget) == 0
        assert mock_widget.write.call_count >= 3

    def test_error_tracking_counts_errors(self):
        """Test that error entries are tracked for navigation."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)

        # Emit some logs including errors
        for level in [logging.INFO, logging.ERROR, logging.WARNING, logging.ERROR]:
            record = logging.LogRecord(
                name="test",
                level=level,
                pathname="test.py",
                lineno=1,
                msg=f"Message at level {level}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        # Should have tracked 2 errors
        assert handler.get_error_count() == 2

    def test_log_entry_stores_enhanced_metadata(self):
        """Test that LogEntry stores raw_message and is_multiline flags."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)

        # Emit a log with JSON
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg='Processing data: {"key": "value"}',
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        # Check the stored log entry
        assert len(handler._log_buffer) == 1
        entry = handler._log_buffer[0]
        assert "Processing data" in entry.raw_message
        assert entry.is_json is True
        assert entry.json_data == {"key": "value"}

    def test_log_entry_detects_multiline_content(self):
        """Test that is_multiline flag is set for long or newline content."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)

        # Emit a multiline log
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred:\nTraceback line 1\nTraceback line 2",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        entry = handler._log_buffer[0]
        assert entry.is_multiline is True

    def test_unregister_cleans_up_pause_state(self):
        """Test that unregistering a widget cleans up pause state."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)
        handler.pause_widget(mock_widget)

        assert handler.is_widget_paused(mock_widget) is True

        handler.unregister_widget(mock_widget)

        # Pause state should be cleaned up
        assert handler.is_widget_paused(mock_widget) is False

    def test_compact_formatter_produces_short_output(self):
        """Test that CompactTuiFormatter produces compact output with icons."""
        from gpt_trader.tui.log_manager import CompactTuiFormatter

        formatter = CompactTuiFormatter()
        record = logging.LogRecord(
            name="gpt_trader.tui.managers.bot_lifecycle",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Mode switch completed",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        # Should have short logger name and icon
        assert "[bot_lifecycle]" in formatted
        assert "✓" in formatted
        assert "Mode switch completed" in formatted
        # Should NOT have full logger path
        assert "gpt_trader.tui.managers" not in formatted

    def test_compact_formatter_uses_correct_icons(self):
        """Test that CompactTuiFormatter uses correct icons for each level."""
        from gpt_trader.tui.log_manager import CompactTuiFormatter

        formatter = CompactTuiFormatter()

        levels_and_icons = [
            (logging.ERROR, "✗"),
            (logging.WARNING, "⚠"),
            (logging.INFO, "✓"),
            (logging.DEBUG, "·"),
        ]

        for level, expected_icon in levels_and_icons:
            record = logging.LogRecord(
                name="test.module",
                level=level,
                pathname="test.py",
                lineno=1,
                msg="Test message",
                args=(),
                exc_info=None,
            )
            formatted = formatter.format(record)
            assert expected_icon in formatted, f"Expected '{expected_icon}' for level {level}"

    def test_format_mode_property(self):
        """Test that format mode property works correctly."""
        handler = TuiLogHandler()

        # Default should be compact
        assert handler.format_mode == "compact"

        # Can set to valid modes
        handler.format_mode = "verbose"
        assert handler.format_mode == "verbose"

        handler.format_mode = "structured"
        assert handler.format_mode == "structured"

        # Invalid mode should be ignored
        handler.format_mode = "invalid"
        assert handler.format_mode == "structured"  # Should remain unchanged

    def test_log_entry_has_structured_fields(self):
        """Test that LogEntry now has AI-friendly structured fields."""
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)

        record = logging.LogRecord(
            name="gpt_trader.tui.managers.bot_lifecycle",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        handler.emit(record)

        entry = handler._log_buffer[0]

        # Check new structured fields
        assert entry.short_logger == "bot_lifecycle"
        assert entry.level_name == "INFO"
        assert entry.category == "trading"  # bot_lifecycle is in trading category
        assert entry.compact_message != ""

    def test_detect_category_function(self):
        """Test that detect_category correctly categorizes loggers."""
        from gpt_trader.tui.log_manager import detect_category

        # Test various logger names (detect_category checks the LAST component)
        assert detect_category("gpt_trader.tui.app") == "startup"
        assert detect_category("gpt_trader.tui.managers.bot_lifecycle") == "trading"
        assert detect_category("gpt_trader.features.risk") == "risk"  # Last component is "risk"
        assert detect_category("gpt_trader.features.market_data") == "market"  # Contains "market"
        assert detect_category("gpt_trader.unknown.module") == "general"
