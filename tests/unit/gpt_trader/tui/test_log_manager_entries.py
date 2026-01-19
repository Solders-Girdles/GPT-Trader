from __future__ import annotations

import logging
from unittest.mock import MagicMock

from gpt_trader.tui.log_manager import TuiLogHandler


class TestTuiLogManagerEntries:
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
