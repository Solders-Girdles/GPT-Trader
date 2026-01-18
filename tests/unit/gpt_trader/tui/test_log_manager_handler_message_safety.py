from __future__ import annotations

import logging
from unittest.mock import MagicMock

from rich.text import Text

from gpt_trader.tui.log_manager import TuiLogHandler
from gpt_trader.tui.theme import THEME


class TestTuiLogHandlerMessageSafety:
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
