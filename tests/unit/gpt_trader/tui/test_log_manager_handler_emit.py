from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest
from rich.text import Text

from gpt_trader.tui.log_manager import TuiLogHandler
from gpt_trader.tui.theme import THEME


class TestTuiLogHandlerEmit:
    def test_write_to_widget_passes_text(self) -> None:
        mock_widget = MagicMock()
        text_message = Text("Test error message", style="red")

        TuiLogHandler._write_to_widget(mock_widget, text_message)

        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]
        assert call_arg is text_message
        assert call_arg.plain == "Test error message"
        assert str(call_arg.style) == "red"

    def test_emit_handles_error_level(self) -> None:
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
            msg="Test error",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        mock_widget.write.assert_called_once()
        call_arg = mock_widget.write.call_args[0][0]
        assert isinstance(call_arg, Text)
        assert "Test error" in call_arg.plain
        assert str(call_arg.style) == THEME.colors.error

    def test_emit_handles_warning_level(self) -> None:
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

    def test_emit_handles_info_level(self) -> None:
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

    def test_emit_handles_debug_level(self) -> None:
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

    def test_emit_respects_widget_level_filter(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.WARNING)

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

        mock_widget.write.assert_not_called()

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

        mock_widget.write.assert_called_once()

    def test_emit_skips_unmounted_widgets(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = False
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

        mock_widget.write.assert_not_called()

    def test_emit_uses_call_from_thread_on_background_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

        mock_current = MagicMock(name="background_thread")
        mock_main = MagicMock(name="main_thread")
        monkeypatch.setattr(threading, "current_thread", lambda: mock_current)
        monkeypatch.setattr(threading, "main_thread", lambda: mock_main)

        handler.emit(record)

        mock_widget.app.call_from_thread.assert_called_once()
        call_args = mock_widget.app.call_from_thread.call_args
        assert call_args[0][0] == handler._write_to_widget
        assert call_args[0][1] is mock_widget
        assert isinstance(call_args[0][2], Text)
