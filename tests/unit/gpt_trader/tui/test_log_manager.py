"""Tests for the TUI log manager (handler behavior + widget controls)."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest
from rich.text import Text

from gpt_trader.tui.log_manager import TuiLogHandler
from gpt_trader.tui.theme import THEME


def _make_record(level: int, msg: str, *, name: str = "test") -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


@pytest.fixture
def widget() -> MagicMock:
    mock_widget = MagicMock()
    mock_widget.is_mounted = True
    mock_widget.app = MagicMock()
    return mock_widget


@pytest.fixture
def handler() -> TuiLogHandler:
    return TuiLogHandler()


class TestTuiLogHandler:
    def test_register_unregister_and_update_widget_level(self, handler) -> None:
        mock_widget = MagicMock()

        handler.register_widget(mock_widget, logging.INFO)
        assert handler._widgets[mock_widget] == logging.INFO

        handler.update_widget_level(mock_widget, logging.WARNING)
        assert handler._widgets[mock_widget] == logging.WARNING

        handler.unregister_widget(mock_widget)
        assert mock_widget not in handler._widgets

    @pytest.mark.parametrize("mode", ["verbose", "compact", "structured", "json"])
    def test_format_mode_accepts_known_values(self, handler, mode: str) -> None:
        handler.format_mode = mode
        assert handler.format_mode == mode

    def test_format_mode_rejects_unknown_values(self, handler) -> None:
        handler.format_mode = "json"
        handler.format_mode = "invalid"
        assert handler.format_mode == "json"

    def test_pause_and_resume_widget_buffers_and_flushes(self, handler, widget) -> None:
        handler.register_widget(widget, logging.DEBUG, replay_logs=False)
        handler.pause_widget(widget)
        assert handler.is_widget_paused(widget) is True

        for i in range(3):
            handler.emit(_make_record(logging.INFO, f"Message {i}"))

        assert widget.write.call_count == 0
        assert handler.get_paused_count(widget) == 3

        handler.resume_widget(widget)
        assert handler.is_widget_paused(widget) is False
        assert handler.get_paused_count(widget) == 0
        assert widget.write.call_count >= 3

    def test_unregister_cleans_up_pause_state(self, handler, widget) -> None:
        handler.register_widget(widget, logging.DEBUG, replay_logs=False)
        handler.pause_widget(widget)
        assert handler.is_widget_paused(widget) is True

        handler.unregister_widget(widget)
        assert handler.is_widget_paused(widget) is False

    def test_error_tracking_counts_errors(self, handler, widget) -> None:
        handler.register_widget(widget, logging.DEBUG, replay_logs=False)

        for level in [logging.INFO, logging.ERROR, logging.WARNING, logging.ERROR]:
            handler.emit(_make_record(level, f"Message at level {level}"))

        assert handler.get_error_count() == 2

    def test_log_entry_metadata_json_and_multiline(self, handler, widget) -> None:
        handler.register_widget(widget, logging.DEBUG, replay_logs=False)

        handler.emit(_make_record(logging.INFO, 'Processing data: {"key": "value"}'))
        handler.emit(
            _make_record(logging.ERROR, "Error occurred:\nTraceback line 1\nTraceback line 2")
        )

        assert len(handler._log_buffer) == 2

        entry_json = handler._log_buffer[0]
        assert entry_json.is_json is True
        assert entry_json.json_data == {"key": "value"}

        entry_multiline = handler._log_buffer[1]
        assert entry_multiline.is_multiline is True

    def test_log_entry_has_structured_fields(self, handler, widget) -> None:
        handler.register_widget(widget, logging.DEBUG, replay_logs=False)

        handler.emit(
            _make_record(
                logging.INFO,
                "Test message",
                name="gpt_trader.tui.managers.bot_lifecycle",
            )
        )

        entry = handler._log_buffer[0]
        assert entry.short_logger == "bot_lifecycle"
        assert entry.level_name == "INFO"
        assert entry.category == "trading"
        assert entry.compact_message

    @pytest.mark.parametrize(
        ("level", "expected_style"),
        [
            (logging.ERROR, THEME.colors.error),
            (logging.WARNING, THEME.colors.warning),
            (logging.INFO, THEME.colors.success),
            (logging.DEBUG, THEME.colors.text_muted),
        ],
    )
    def test_emit_writes_styled_text(
        self,
        handler,
        widget,
        level: int,
        expected_style,
    ) -> None:
        handler.register_widget(widget, logging.DEBUG, replay_logs=False)
        handler.emit(_make_record(level, "Test message"))

        widget.write.assert_called_once()
        call_arg = widget.write.call_args[0][0]
        assert isinstance(call_arg, Text)
        assert "Test message" in call_arg.plain
        assert str(call_arg.style) == expected_style

    @pytest.mark.parametrize(
        ("msg", "level", "expected_style", "expected_literal"),
        [
            ("Processing [red]data[/red] from API", logging.INFO, THEME.colors.success, "[red]"),
            ("Invalid regex pattern: [/] detected", logging.ERROR, THEME.colors.error, "[/]"),
            (
                "Config: [section] key=[value] [/incomplete",
                logging.WARNING,
                THEME.colors.warning,
                "[section]",
            ),
        ],
    )
    def test_log_message_is_treated_as_literal_text(
        self,
        handler,
        widget,
        msg: str,
        level: int,
        expected_style,
        expected_literal: str,
    ) -> None:
        handler.register_widget(widget, logging.DEBUG, replay_logs=False)
        handler.emit(_make_record(level, msg))

        widget.write.assert_called_once()
        call_arg = widget.write.call_args[0][0]
        assert expected_literal in call_arg.plain
        assert str(call_arg.style) == expected_style

    def test_emit_respects_widget_min_level(self, handler, widget) -> None:
        handler.register_widget(widget, logging.WARNING, replay_logs=False)
        handler.emit(_make_record(logging.DEBUG, "Debug message"))
        widget.write.assert_not_called()

        handler.emit(_make_record(logging.WARNING, "Warning message"))
        widget.write.assert_called_once()

    def test_emit_skips_unmounted_widgets(self, handler) -> None:
        mock_widget = MagicMock()
        mock_widget.is_mounted = False
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)
        handler.emit(_make_record(logging.INFO, "Test message"))

        mock_widget.write.assert_not_called()

    def test_emit_uses_call_from_thread_off_main_thread(
        self,
        handler,
        widget,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler.register_widget(widget, logging.DEBUG, replay_logs=False)

        mock_current = MagicMock(name="background_thread")
        mock_main = MagicMock(name="main_thread")
        monkeypatch.setattr(threading, "current_thread", lambda: mock_current)
        monkeypatch.setattr(threading, "main_thread", lambda: mock_main)

        handler.emit(_make_record(logging.INFO, "Test message"))

        widget.app.call_from_thread.assert_called_once()
        call_args = widget.app.call_from_thread.call_args
        assert call_args[0][0] == handler._write_to_widget
        assert call_args[0][1] is widget
        assert isinstance(call_args[0][2], Text)

    def test_handler_format_has_no_extra_newline(self, handler) -> None:
        formatted = handler.format(_make_record(logging.INFO, "Test message"))
        assert not formatted.endswith("\n")

    def test_write_to_widget_passes_rich_text(self) -> None:
        mock_widget = MagicMock()
        test_text = Text("Test", style=THEME.colors.success)

        TuiLogHandler._write_to_widget(mock_widget, test_text)

        call_arg = mock_widget.write.call_args[0][0]
        assert call_arg is test_text
        assert call_arg.plain == "Test"
