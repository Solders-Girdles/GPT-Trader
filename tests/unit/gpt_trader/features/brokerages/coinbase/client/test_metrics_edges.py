"""Edge coverage for Coinbase client metrics helpers."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.client.metrics import (
    EndpointMetrics,
    categorize_endpoint,
)


def test_categorize_endpoint_case_insensitive() -> None:
    assert categorize_endpoint("/API/V3/BROKERAGE/ORDERS") == "orders"
    assert categorize_endpoint("/api/v3/brokerage/MARKET/TICKER") == "market"
    assert categorize_endpoint("/api/v3/brokerage/positions") == "positions"
    assert categorize_endpoint("/api/v3/brokerage/unknown") == "other"


def test_endpoint_metrics_to_dict_zeroed_defaults() -> None:
    metrics = EndpointMetrics()

    data = metrics.to_dict()

    assert data["total_calls"] == 0
    assert data["total_errors"] == 0
    assert data["error_rate"] == 0.0
    assert data["avg_latency_ms"] == 0.0
    assert data["last_latency_ms"] == 0.0
    assert data["min_latency_ms"] == 0.0
    assert data["max_latency_ms"] == 0.0
    assert data["p50_latency_ms"] == 0.0
    assert data["p95_latency_ms"] == 0.0
    assert data["p99_latency_ms"] == 0.0
