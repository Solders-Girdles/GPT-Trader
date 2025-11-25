from __future__ import annotations

import logging

from gpt_trader.utilities import logging_patterns


class ListHandler(logging.Handler):
    """Handler that captures log records for inspection."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_log_execution_includes_args_and_result() -> None:
    """Test that log_execution decorator captures args and results."""
    # Set up a logger with our custom handler
    exec_logger = logging.getLogger("execution.test.logger")
    handler = ListHandler()
    exec_logger.handlers = [handler]
    exec_logger.setLevel(logging.DEBUG)
    exec_logger.propagate = False

    try:

        @logging_patterns.log_execution(
            operation="compute_sum",
            logger=exec_logger,
            include_args=True,
            include_result=True,
        )
        def compute(a: int, b: int) -> int:
            return a + b

        result = compute(3, 4)

        assert result == 7
        assert len(handler.records) == 3

        start_rec, result_rec, end_rec = handler.records

        # Check start message
        assert "Started compute_sum" in start_rec.getMessage()
        assert hasattr(start_rec, "operation")
        assert start_rec.operation == "compute_sum"
        assert hasattr(start_rec, "arg_0")
        assert start_rec.arg_0 == "3"
        assert hasattr(start_rec, "arg_1")
        assert start_rec.arg_1 == "4"

        # Check result message
        assert "Result: 7" in result_rec.getMessage()

        # Check end message
        assert "Completed compute_sum" in end_rec.getMessage()
        assert hasattr(end_rec, "duration_ms")
        assert float(end_rec.duration_ms) >= 0
    finally:
        exec_logger.handlers = []
