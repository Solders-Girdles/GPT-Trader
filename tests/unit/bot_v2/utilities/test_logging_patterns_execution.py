from __future__ import annotations

import logging

import pytest

from bot_v2.utilities import logging_patterns


def test_log_execution_includes_args_and_result(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.clear()
    exec_logger = logging.getLogger("execution.logger")
    caplog.set_level(logging.DEBUG, "execution.logger")

    captured_kwargs: list[dict[str, object]] = []
    original_format = logging_patterns.StructuredLogger._format_message

    def spy_format(self, message: str, **kwargs: object) -> str:  # type: ignore[override]
        captured_kwargs.append(dict(kwargs))
        return original_format(self, message, **kwargs)

    monkeypatch.setattr(logging_patterns.StructuredLogger, "_format_message", spy_format)

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
    assert len(caplog.records) == 3
    start, debug_msg, end = (record.message for record in caplog.records)
    assert start.startswith("Started compute_sum")
    assert end.startswith("Completed compute_sum")
    duration_token = next(part for part in end.split() if part.startswith("duration_ms="))
    assert float(duration_token.split("=")[1]) >= 0
    assert debug_msg == "Result: 7"
    assert captured_kwargs[0]["operation"] == "compute_sum"
    assert captured_kwargs[0]["arg_0"] == "3"
    assert captured_kwargs[2]["operation"] == "compute_sum"
    assert captured_kwargs[2]["arg_1"] == "4"
