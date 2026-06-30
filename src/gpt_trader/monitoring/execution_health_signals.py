"""Execution health signal computation.

Pure functions that derive health signals (order error/retry rate, broker call
latency p95, guard trip frequency, missing decision ids) from the metrics
collector's counter/histogram summary. Separated from health_checks.py so the
metric math is isolated from broker/ticker probing and the HealthCheckRunner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_trader.monitoring.health_signals import (
        HealthSignal,
        HealthSummary,
        HealthThresholds,
    )


def compute_execution_health_signals(
    thresholds: HealthThresholds | None = None,
) -> HealthSummary:
    """Compute health signals from execution metrics.

    Uses the metrics collector to derive health signals for:
    - Order submission error rate
    - Order retry rate
    - Broker call latency (p95)
    - Guard trip frequency
    - Missing decision_id count

    Args:
        thresholds: Optional custom thresholds. Uses defaults if None.

    Returns:
        HealthSummary with computed signals.
    """
    from gpt_trader.monitoring.health_signals import (
        HealthSummary,
        HealthThresholds,
    )
    from gpt_trader.monitoring.metrics_collector import get_metrics_collector

    if thresholds is None:
        thresholds = HealthThresholds()

    counters, histograms = _extract_execution_metrics(get_metrics_collector().get_metrics_summary())
    signals = _build_execution_health_signals(counters, histograms, thresholds)
    return HealthSummary.from_signals(signals)


def _extract_execution_metrics(
    summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    counters_raw = summary.get("counters", {})
    histograms_raw = summary.get("histograms", {})
    counters = counters_raw if isinstance(counters_raw, dict) else {}
    histograms = histograms_raw if isinstance(histograms_raw, dict) else {}
    return counters, histograms


def _build_execution_health_signals(
    counters: dict[str, Any],
    histograms: dict[str, Any],
    thresholds: HealthThresholds,
) -> list[HealthSignal]:
    from gpt_trader.monitoring.health_signals import HealthSignal

    total_submissions, failed_submissions = _count_order_submissions(counters)
    error_rate = _ratio_or_zero(failed_submissions, total_submissions)

    broker_call_count, broker_call_failures = _count_broker_calls(histograms)
    retry_rate = _ratio_or_zero(broker_call_count - total_submissions, total_submissions)
    if broker_call_count <= total_submissions:
        retry_rate = 0.0

    latency_p95_ms = _compute_broker_latency_p95_ms(histograms)
    guard_trip_count = _count_guard_trips(counters)
    missing_decision_id_count = _count_missing_decision_ids(counters)

    return [
        HealthSignal.from_value(
            name="order_error_rate",
            value=error_rate,
            threshold_warn=thresholds.order_error_rate_warn,
            threshold_crit=thresholds.order_error_rate_crit,
            unit="ratio",
            details={
                "total_submissions": total_submissions,
                "failed_submissions": failed_submissions,
            },
        ),
        HealthSignal.from_value(
            name="order_retry_rate",
            value=retry_rate,
            threshold_warn=thresholds.order_retry_rate_warn,
            threshold_crit=thresholds.order_retry_rate_crit,
            unit="ratio",
            details={
                "broker_call_count": broker_call_count,
                "broker_call_failures": broker_call_failures,
                "submission_count": total_submissions,
            },
        ),
        HealthSignal.from_value(
            name="broker_latency_p95",
            value=latency_p95_ms,
            threshold_warn=thresholds.broker_latency_ms_warn,
            threshold_crit=thresholds.broker_latency_ms_crit,
            unit="ms",
            details={},
        ),
        HealthSignal.from_value(
            name="guard_trip_count",
            value=float(guard_trip_count),
            threshold_warn=float(thresholds.guard_trip_count_warn),
            threshold_crit=float(thresholds.guard_trip_count_crit),
            unit="count",
            details={},
        ),
        HealthSignal.from_value(
            name="missing_decision_id_count",
            value=float(missing_decision_id_count),
            threshold_warn=float(thresholds.missing_decision_id_count_warn),
            threshold_crit=float(thresholds.missing_decision_id_count_crit),
            unit="count",
            details={},
        ),
    ]


def _count_order_submissions(counters: dict[str, Any]) -> tuple[int, int]:
    total_submissions = 0
    failed_submissions = 0
    for key, count in counters.items():
        if key.startswith("gpt_trader_order_submission_total"):
            total_submissions += count
            if "result=failed" in key or "result=rejected" in key:
                failed_submissions += count
    return total_submissions, failed_submissions


def _count_broker_calls(histograms: dict[str, Any]) -> tuple[int, int]:
    broker_call_count = 0
    broker_call_failures = 0
    for key, hist_data in histograms.items():
        if key.startswith("gpt_trader_broker_call_latency_seconds"):
            if isinstance(hist_data, dict):
                broker_call_count += hist_data.get("count", 0)
                if "outcome=failure" in key:
                    broker_call_failures += hist_data.get("count", 0)
    return broker_call_count, broker_call_failures


def _compute_broker_latency_p95_ms(histograms: dict[str, Any]) -> float:
    for key, hist_data in histograms.items():
        if key.startswith("gpt_trader_broker_call_latency_seconds") and "outcome=success" in key:
            if isinstance(hist_data, dict):
                return _approximate_p95_ms(hist_data)
            break
    return 0.0


def _approximate_p95_ms(hist_data: dict[str, Any]) -> float:
    total_count = hist_data.get("count", 0)
    if total_count <= 0:
        return 0.0

    buckets = hist_data.get("buckets", {})
    target = total_count * 0.95
    for bucket_str, bucket_count in sorted(buckets.items(), key=lambda x: float(x[0])):
        if bucket_count >= target:
            return float(bucket_str) * 1000

    mean = hist_data.get("mean", 0.0)
    return float(mean) * 1000 * 1.5


def _count_guard_trips(counters: dict[str, Any]) -> int:
    guard_trip_count = 0
    for key, count in counters.items():
        lowered = key.lower()
        if "guard" in lowered and ("failure" in lowered or "trip" in lowered):
            guard_trip_count += count
    return guard_trip_count


def _count_missing_decision_ids(counters: dict[str, Any]) -> int:
    missing_count = 0
    for key, count in counters.items():
        if key.startswith("gpt_trader_order_missing_decision_id_total"):
            missing_count += count
    return missing_count


def _ratio_or_zero(numerator: int | float, denominator: int | float) -> float:
    if denominator > 0:
        return float(numerator) / float(denominator)
    return 0.0
