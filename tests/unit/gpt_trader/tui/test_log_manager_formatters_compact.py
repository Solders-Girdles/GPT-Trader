from __future__ import annotations

import logging


class TestCompactTuiFormatter:
    def test_compact_formatter_produces_short_output(self) -> None:
        from gpt_trader.tui.log_manager import LEVEL_ICONS, CompactTuiFormatter

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

        assert "[bot_lifecycle]" in formatted
        assert LEVEL_ICONS[logging.INFO] in formatted
        assert "Mode switch completed" in formatted
        assert "gpt_trader.tui.managers" not in formatted

    def test_compact_formatter_uses_correct_icons(self) -> None:
        from gpt_trader.tui.log_manager import LEVEL_ICONS, CompactTuiFormatter

        formatter = CompactTuiFormatter()

        levels_and_icons = [
            (logging.ERROR, LEVEL_ICONS[logging.ERROR]),
            (logging.WARNING, LEVEL_ICONS[logging.WARNING]),
            (logging.INFO, LEVEL_ICONS[logging.INFO]),
            (logging.DEBUG, LEVEL_ICONS[logging.DEBUG]),
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
