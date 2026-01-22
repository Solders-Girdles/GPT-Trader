"""Tests for execution telemetry metrics and issue tracking."""

from __future__ import annotations

import pytest

from gpt_trader.tui.services.execution_telemetry import (
    ExecutionTelemetryCollector,
    clear_execution_telemetry,
)
from gpt_trader.tui.types import ExecutionMetrics


@pytest.fixture(autouse=True)
def _clear_telemetry():
    clear_execution_telemetry()
    yield
    clear_execution_telemetry()


def _record_rejection(collector: ExecutionTelemetryCollector, reason: str, **kwargs) -> None:
    collector.record_submission(
        latency_ms=30.0,
        success=False,
        rejected=True,
        rejection_reason=reason,
        **kwargs,
    )


def _record_retry(collector: ExecutionTelemetryCollector, reason: str, **kwargs) -> None:
    collector.record_retry(reason=reason, **kwargs)


class TestExecutionMetrics:
    def test_default_values(self):
        metrics = ExecutionMetrics()
        assert metrics.submissions_total == 0
        assert metrics.submissions_success == 0
        assert metrics.success_rate == 100.0
        assert metrics.is_healthy is True
        assert metrics.rejection_reasons == {}
        assert metrics.retry_reasons == {}
        assert metrics.recent_rejections == []
        assert metrics.recent_retries == []

    @pytest.mark.parametrize(
        ("total", "success", "expected"),
        [(10, 8, 80.0), (0, 0, 100.0)],
    )
    def test_success_rate_calculation(self, total: int, success: int, expected: float):
        metrics = ExecutionMetrics(submissions_total=total, submissions_success=success)
        assert metrics.success_rate == expected

    @pytest.mark.parametrize(
        ("total", "success", "retry_rate", "expected"),
        [(100, 98, 0.2, True), (100, 80, 0.2, False), (100, 98, 0.6, False)],
    )
    def test_is_healthy(self, total: int, success: int, retry_rate: float, expected: bool):
        metrics = ExecutionMetrics(
            submissions_total=total,
            submissions_success=success,
            retry_rate=retry_rate,
        )
        assert metrics.is_healthy is expected

    def test_top_rejection_reasons_sorted(self):
        metrics = ExecutionMetrics(
            rejection_reasons={"rate_limit": 5, "insufficient_funds": 2, "timeout": 8}
        )
        assert metrics.top_rejection_reasons == [
            ("timeout", 8),
            ("rate_limit", 5),
            ("insufficient_funds", 2),
        ]

    def test_top_rejection_reasons_empty(self):
        metrics = ExecutionMetrics()
        assert metrics.top_rejection_reasons == []

    def test_top_retry_reasons_sorted(self):
        metrics = ExecutionMetrics(retry_reasons={"timeout": 3, "network": 7, "rate_limit": 1})
        assert metrics.top_retry_reasons == [("network", 7), ("timeout", 3), ("rate_limit", 1)]


class TestIssueTracking:
    @pytest.mark.parametrize(
        ("kwargs", "expected_reason"),
        [
            (
                {
                    "rejected": True,
                    "rejection_reason": "rate_limit",
                    "symbol": "BTC-USD",
                    "side": "BUY",
                    "quantity": 0.5,
                    "price": 30000.0,
                },
                "rate_limit",
            ),
            (
                {
                    "failure_reason": "Timeout",
                    "symbol": "ETH-USD",
                    "side": "SELL",
                    "quantity": 1.0,
                    "price": 2000.0,
                },
                "Timeout",
            ),
        ],
    )
    def test_submission_issue_records_context(self, kwargs, expected_reason):
        collector = ExecutionTelemetryCollector()
        collector.record_submission(latency_ms=30.0, success=False, **kwargs)

        issue = collector.get_metrics().recent_rejections[0]
        assert issue.symbol == kwargs["symbol"]
        assert issue.side == kwargs["side"]
        assert issue.quantity == kwargs["quantity"]
        assert issue.price == kwargs["price"]
        assert issue.reason == expected_reason
        assert issue.is_retry is False

    def test_retry_issue_records_context(self):
        collector = ExecutionTelemetryCollector()
        _record_retry(
            collector,
            reason="timeout",
            symbol="SOL-USD",
            side="SELL",
            quantity=2.0,
            price=95.0,
        )
        collector.record_submission(latency_ms=20.0, success=True)

        issue = collector.get_metrics().recent_retries[0]
        assert issue.symbol == "SOL-USD"
        assert issue.side == "SELL"
        assert issue.quantity == 2.0
        assert issue.price == 95.0
        assert issue.reason == "timeout"
        assert issue.is_retry is True

    def test_issue_ordering_is_most_recent_first(self):
        collector = ExecutionTelemetryCollector()
        _record_rejection(collector, "older")
        _record_rejection(collector, "newer")

        metrics = collector.get_metrics()
        assert [issue.reason for issue in metrics.recent_rejections] == ["newer", "older"]

    def test_empty_retry_reason_skips_issue(self):
        collector = ExecutionTelemetryCollector()
        _record_retry(collector, reason="")
        collector.record_submission(latency_ms=50.0, success=True)

        metrics = collector.get_metrics()
        assert metrics.recent_retries == []


class TestReasonTracking:
    def test_rejection_reasons_aggregate_and_sorted(self):
        collector = ExecutionTelemetryCollector()
        for reason in ["rate_limit", "insufficient_funds", "rate_limit"]:
            _record_rejection(collector, reason)

        metrics = collector.get_metrics()
        assert metrics.rejection_reasons == {"rate_limit": 2, "insufficient_funds": 1}
        assert metrics.top_rejection_reasons == [
            ("rate_limit", 2),
            ("insufficient_funds", 1),
        ]

    def test_retry_reasons_aggregate_and_sorted(self):
        collector = ExecutionTelemetryCollector()
        for reason in ["network", "network", "network", "timeout"]:
            _record_retry(collector, reason)
        collector.record_submission(latency_ms=50.0, success=True)

        metrics = collector.get_metrics()
        assert metrics.retry_reasons == {"network": 3, "timeout": 1}
        assert metrics.top_retry_reasons == [("network", 3), ("timeout", 1)]

    def test_empty_reasons_not_tracked(self):
        collector = ExecutionTelemetryCollector()
        _record_rejection(collector, reason="")
        _record_retry(collector, reason="")
        collector.record_submission(latency_ms=50.0, success=True)

        metrics = collector.get_metrics()
        assert metrics.rejection_reasons == {}
        assert metrics.retry_reasons == {}

    def test_reasons_respect_rolling_window(self):
        collector = ExecutionTelemetryCollector(window_size=3)
        for _ in range(3):
            _record_rejection(collector, "rate_limit")

        metrics = collector.get_metrics()
        assert metrics.rejection_reasons == {"rate_limit": 3}

        for _ in range(3):
            _record_rejection(collector, "timeout")

        metrics = collector.get_metrics()
        assert metrics.rejection_reasons == {"timeout": 3}
        assert "rate_limit" not in metrics.rejection_reasons

    def test_clear_clears_reasons(self):
        collector = ExecutionTelemetryCollector()
        _record_rejection(collector, "rate_limit")
        _record_retry(collector, "timeout")

        collector.clear()
        collector.record_submission(latency_ms=50.0, success=True)
        metrics = collector.get_metrics()

        assert metrics.rejection_reasons == {}
        assert metrics.retry_reasons == {}
        assert metrics.recent_rejections == []
        assert metrics.recent_retries == []
