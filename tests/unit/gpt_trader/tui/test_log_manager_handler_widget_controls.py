from __future__ import annotations

import logging
from unittest.mock import MagicMock

from gpt_trader.tui.log_manager import TuiLogHandler, get_tui_log_handler


class TestTuiLogHandlerWidgetControls:
    def test_get_tui_log_handler_returns_singleton(self) -> None:
        handler1 = get_tui_log_handler()
        handler2 = get_tui_log_handler()
        assert handler1 is handler2

    def test_register_and_unregister_widget(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()

        handler.register_widget(mock_widget, logging.INFO)
        assert mock_widget in handler._widgets
        assert handler._widgets[mock_widget] == logging.INFO

        handler.unregister_widget(mock_widget)
        assert mock_widget not in handler._widgets

    def test_update_widget_level(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()

        handler.register_widget(mock_widget, logging.INFO)
        assert handler._widgets[mock_widget] == logging.INFO

        handler.update_widget_level(mock_widget, logging.WARNING)
        assert handler._widgets[mock_widget] == logging.WARNING

    def test_format_mode_property(self) -> None:
        handler = TuiLogHandler()

        assert handler.format_mode == "structured"

        handler.format_mode = "verbose"
        assert handler.format_mode == "verbose"

        handler.format_mode = "compact"
        assert handler.format_mode == "compact"

        handler.format_mode = "json"
        assert handler.format_mode == "json"

        handler.format_mode = "invalid"
        assert handler.format_mode == "json"

    def test_pause_widget_buffers_logs(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)

        handler.pause_widget(mock_widget)
        assert handler.is_widget_paused(mock_widget) is True

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

        initial_call_count = mock_widget.write.call_count
        handler.emit(record)
        assert mock_widget.write.call_count == initial_call_count

        assert handler.get_paused_count(mock_widget) >= 1

    def test_resume_widget_flushes_buffered_logs(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG, replay_logs=False)

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

        assert handler.get_paused_count(mock_widget) == 3

        handler.resume_widget(mock_widget)

        assert handler.is_widget_paused(mock_widget) is False
        assert handler.get_paused_count(mock_widget) == 0
        assert mock_widget.write.call_count >= 3

    def test_unregister_cleans_up_pause_state(self) -> None:
        handler = TuiLogHandler()
        mock_widget = MagicMock()
        mock_widget.is_mounted = True
        mock_widget.app = MagicMock()

        handler.register_widget(mock_widget, logging.DEBUG)
        handler.pause_widget(mock_widget)

        assert handler.is_widget_paused(mock_widget) is True

        handler.unregister_widget(mock_widget)

        assert handler.is_widget_paused(mock_widget) is False
