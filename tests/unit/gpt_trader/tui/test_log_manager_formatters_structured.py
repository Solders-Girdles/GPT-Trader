from __future__ import annotations

import logging


class TestStructuredTuiFormatter:
    def test_detect_category_function(self) -> None:
        from gpt_trader.tui.log_manager import detect_category

        assert detect_category("gpt_trader.tui.app") == "startup"
        assert detect_category("gpt_trader.tui.managers.bot_lifecycle") == "trading"
        assert detect_category("gpt_trader.features.risk") == "risk"
        assert detect_category("gpt_trader.features.market_data") == "market"
        assert detect_category("gpt_trader.unknown.module") == "general"

    def test_structured_formatter_basic_output(self) -> None:
        from gpt_trader.tui.log_manager import LEVEL_ICONS, StructuredTuiFormatter

        formatter = StructuredTuiFormatter()
        record = logging.LogRecord(
            name="gpt_trader.tui.app",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Application started",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert LEVEL_ICONS[logging.INFO] in formatted
        assert "app" in formatted
        assert "Application started" in formatted

    def test_structured_formatter_abbreviates_loggers(self) -> None:
        from gpt_trader.tui.log_manager import _abbreviate_logger

        assert _abbreviate_logger("gpt_trader.strategy").strip() == "strat"
        assert _abbreviate_logger("gpt_trader.portfolio").strip() == "port"
        assert _abbreviate_logger("gpt_trader.bot_lifecycle").strip() == "bot"

        long_name = "very_long_logger_name"
        result = _abbreviate_logger(long_name, max_len=10)
        assert len(result) == 10
        assert result.endswith("…") or result.strip() == long_name[:10]

    def test_structured_formatter_formats_strategy_debug(self) -> None:
        from gpt_trader.tui.log_manager import StructuredTuiFormatter

        formatter = StructuredTuiFormatter()
        record = logging.LogRecord(
            name="gpt_trader.features.strategy",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Strategy decision debug: symbol=BTC-USD marks=30 short_ma=115.25507938085695 long_ma=113.3961448256771715 bullish=False bearish=False label=neutral force=False",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "BTC-USD" in formatted
        assert "NEUTRAL" in formatted
        assert "115.26" in formatted or "115.25" in formatted
        assert "113.40" in formatted or "113.39" in formatted
        assert "bullish=False" not in formatted
        assert "bearish=False" not in formatted

    def test_structured_formatter_formats_strategy_decisions_with_reason(self) -> None:
        from gpt_trader.tui.log_manager import StructuredTuiFormatter

        formatter = StructuredTuiFormatter()
        record = logging.LogRecord(
            name="gpt_trader.features.strategy",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Strategy Decision for BTC-USD: BUY (momentum crossover)",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "BTC-USD" in formatted
        assert "BUY" in formatted
        assert "momentum crossover" in formatted

    def test_structured_formatter_formats_numbers(self) -> None:
        from gpt_trader.tui.log_manager import _format_number

        assert _format_number("115.25507938085695") == "115.26"
        assert _format_number("0.00012345") == "0.00"
        assert _format_number("98450.50") == "98,450.50"
        assert _format_number("not_a_number") == "not_a_number"

    def test_structured_formatter_handles_exceptions(self) -> None:
        from gpt_trader.tui.log_manager import StructuredTuiFormatter

        formatter = StructuredTuiFormatter()

        try:
            raise ValueError("Test error message")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="gpt_trader.tui.app",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="An error occurred",
            args=(),
            exc_info=exc_info,
        )

        formatted = formatter.format(record)

        assert "ValueError" in formatted
        assert "Test error" in formatted
