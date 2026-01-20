"""Tests for TUI log manager entries and message safety."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from rich.text import Text

from gpt_trader.tui.log_manager import TuiLogHandler
from gpt_trader.tui.theme import THEME


class TestTuiLogManagerEntries:
    """Tests for log entry handling and metadata."""

    def test_error_tracking_counts_errors(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)

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

        assert handler.get_error_count() == 2

    def test_log_entry_stores_enhanced_metadata(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)

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

        assert len(handler._log_buffer) == 1
        entry = handler._log_buffer[0]
        assert "Processing data" in entry.raw_message
        assert entry.is_json is True
        assert entry.json_data == {"key": "value"}

    def test_log_entry_detects_multiline_content(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)

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

    def test_log_entry_has_structured_fields(self) -> None:
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
        assert entry.short_logger == "bot_lifecycle"
        assert entry.level_name == "INFO"
        assert entry.category == "trading"
        assert entry.compact_message != ""


class TestTuiLogHandlerMessageSafety:
    """Tests for log message safety and markup handling."""

    def test_log_message_with_markup_characters_is_escaped(self) -> None:
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
            msg="Processing [red]data[/red] from API",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]

        assert "Processing" in call_arg.plain
        assert "data" in call_arg.plain
        assert "from API" in call_arg.plain
        assert str(call_arg.style) == THEME.colors.success

    def test_log_message_with_closing_bracket_is_safe(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

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

        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]

        assert "Invalid regex pattern" in call_arg.plain
        assert "detected" in call_arg.plain
        assert str(call_arg.style) == THEME.colors.error

    def test_log_message_with_multiple_brackets_is_safe(self) -> None:
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
            msg="Config: [section] key=[value] [/incomplete",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]

        assert "Config:" in call_arg.plain
        assert "section" in call_arg.plain
        assert "key=" in call_arg.plain
        assert "value" in call_arg.plain
        assert str(call_arg.style) == THEME.colors.warning

    def test_formatter_output_has_no_extra_newline(self) -> None:
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
