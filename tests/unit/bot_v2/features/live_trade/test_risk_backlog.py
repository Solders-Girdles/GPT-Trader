"""Placeholders tracking pending risk features."""

import pytest


@pytest.mark.xfail(reason="TODO(2025-01-31): circuit breaker APIs still pending", strict=False)
def test_circuit_breakers_placeholder():
    raise NotImplementedError("Circuit breaker methods not yet implemented")


@pytest.mark.xfail(reason="TODO(2025-01-31): impact cost modelling pending", strict=False)
def test_impact_cost_placeholder():
    raise NotImplementedError("Impact cost methods not yet implemented")


@pytest.mark.xfail(reason="TODO(2025-01-31): dynamic position sizing backlog", strict=False)
def test_position_sizing_placeholder():
    raise NotImplementedError("Position sizing methods not yet implemented")


@pytest.mark.xfail(reason="TODO(2025-01-31): risk metrics aggregation backlog", strict=False)
def test_risk_metrics_placeholder():
    raise NotImplementedError("Risk metrics methods not yet implemented")
