"""Tests for runtime logging helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

import gpt_trader.logging.runtime_helpers as runtime_helpers
from gpt_trader.logging.runtime_helpers import (
    get_runtime_logger,
    log_execution_error,
    log_market_data_update,
    log_order_event,
    log_risk_event,
    log_strategy_decision,
    log_trading_operation,
)


@dataclass(slots=True)
class RuntimeLogHarness:
    logger: MagicMock
    base_context: dict[str, Any]
    add_domain_field: MagicMock


@pytest.fixture
def runtime_log_harness(monkeypatch: pytest.MonkeyPatch) -> RuntimeLogHarness:
    logger = MagicMock(name="runtime_logger")
    base_context: dict[str, Any] = {}

    def get_log_context() -> dict[str, Any]:
        return dict(base_context)

    monkeypatch.setattr(runtime_helpers, "get_log_context", get_log_context)
    monkeypatch.setattr(runtime_helpers, "get_runtime_logger", lambda *_args, **_kwargs: logger)

    add_domain_field = MagicMock(name="add_domain_field")
    monkeypatch.setattr(runtime_helpers, "add_domain_field", add_domain_field)

    return RuntimeLogHarness(
        logger=logger,
        base_context=base_context,
        add_domain_field=add_domain_field,
    )


def _assert_logged(
    logger: MagicMock, *, level: int, message: str, exc_info: bool | None = None
) -> dict[str, Any]:
    logger.log.assert_called_once()
    args, kwargs = logger.log.call_args

    assert args[0] == level
    assert args[1] == message
    if exc_info is not None:
        assert kwargs["exc_info"] is exc_info

    extra = kwargs.get("extra")
    assert isinstance(extra, dict)
    return extra


def test_get_runtime_logger() -> None:
    """Test getting a runtime logger."""
    logger = get_runtime_logger("test_component")
    assert logger.name == "gpt_trader.json.test_component"

    # Test with JSON disabled
    logger = get_runtime_logger("test_component", enable_json=False)
    assert logger.name == "test_component"


def test_log_trading_operation(runtime_log_harness: RuntimeLogHarness) -> None:
    """Test logging a trading operation."""
    runtime_log_harness.base_context.update({"correlation_id": "test-123", "symbol": "BTC-USD"})

    log_trading_operation(
        operation="test_operation",
        symbol="BTC-USD",
        level=logging.INFO,
        test_field="test_value",
    )

    runtime_log_harness.add_domain_field.assert_called_once_with("symbol", "BTC-USD")

    extra = _assert_logged(
        runtime_log_harness.logger,
        level=logging.INFO,
        message="test_operation",
    )
    assert extra["symbol"] == "BTC-USD"
    assert extra["operation"] == "test_operation"
    assert extra["test_field"] == "test_value"
    assert extra["correlation_id"] == "test-123"


def test_log_order_event(runtime_log_harness: RuntimeLogHarness) -> None:
    """Test logging an order event."""
    runtime_log_harness.base_context.update({"correlation_id": "order-456"})

    log_order_event(
        event_type="order_placed",
        order_id="order-123",
        symbol="ETH-USD",
        side="buy",
        quantity=Decimal("1.5"),
        price=Decimal("3000.25"),
        level=logging.INFO,
    )

    extra = _assert_logged(
        runtime_log_harness.logger,
        level=logging.INFO,
        message="Order event: order_placed",
    )
    assert extra["event_type"] == "order_placed"
    assert extra["order_id"] == "order-123"
    assert extra["symbol"] == "ETH-USD"
    assert extra["side"] == "buy"
    assert extra["quantity"] == 1.5
    assert extra["price"] == 3000.25
    assert extra["correlation_id"] == "order-456"


def test_log_strategy_decision(runtime_log_harness: RuntimeLogHarness) -> None:
    """Test logging a strategy decision."""
    runtime_log_harness.base_context.update({"correlation_id": "strategy-789"})

    log_strategy_decision(
        symbol="BTC-USD",
        decision="BUY",
        reason="RSI oversold",
        confidence=0.85,
        level=logging.INFO,
    )

    extra = _assert_logged(
        runtime_log_harness.logger,
        level=logging.INFO,
        message="Strategy decision for BTC-USD: BUY",
    )
    assert extra["symbol"] == "BTC-USD"
    assert extra["decision"] == "BUY"
    assert extra["reason"] == "RSI oversold"
    assert extra["confidence"] == 0.85
    assert extra["correlation_id"] == "strategy-789"


def test_log_execution_error(runtime_log_harness: RuntimeLogHarness) -> None:
    """Test logging an execution error."""
    runtime_log_harness.base_context.update({"correlation_id": "error-123"})

    test_error = ValueError("Test error message")

    log_execution_error(
        error=test_error,
        operation="place_order",
        symbol="ETH-USD",
        order_id="order-456",
        level=logging.ERROR,
    )

    extra = _assert_logged(
        runtime_log_harness.logger,
        level=logging.ERROR,
        message="Execution error in place_order: Test error message",
        exc_info=True,
    )
    assert extra["operation"] == "place_order"
    assert extra["error_type"] == "ValueError"
    assert extra["error_message"] == "Test error message"
    assert extra["symbol"] == "ETH-USD"
    assert extra["order_id"] == "order-456"
    assert extra["correlation_id"] == "error-123"


def test_log_risk_event(runtime_log_harness: RuntimeLogHarness) -> None:
    """Test logging a risk event."""
    runtime_log_harness.base_context.update({"correlation_id": "risk-456"})

    log_risk_event(
        event_type="circuit_breaker_triggered",
        symbol="BTC-USD",
        trigger_value=0.15,
        threshold=0.10,
        action="reduce_only",
        level=logging.WARNING,
    )

    extra = _assert_logged(
        runtime_log_harness.logger,
        level=logging.WARNING,
        message="Risk event: circuit_breaker_triggered",
    )
    assert extra["event_type"] == "circuit_breaker_triggered"
    assert extra["symbol"] == "BTC-USD"
    assert extra["trigger_value"] == "0.15"
    assert extra["threshold"] == "0.10"
    assert extra["action"] == "reduce_only"
    assert extra["correlation_id"] == "risk-456"


def test_log_market_data_update(runtime_log_harness: RuntimeLogHarness) -> None:
    """Test logging a market data update."""
    runtime_log_harness.base_context.update({"correlation_id": "market-789"})

    log_market_data_update(
        symbol="BTC-USD",
        price=Decimal("50000.50"),
        volume=Decimal("100.25"),
        timestamp=1640995200.123,
        level=logging.DEBUG,
    )

    extra = _assert_logged(
        runtime_log_harness.logger,
        level=logging.DEBUG,
        message="Market data update: BTC-USD",
    )
    assert extra["symbol"] == "BTC-USD"
    assert extra["price"] == 50000.50
    assert extra["volume"] == 100.25
    assert extra["timestamp"] == 1640995200.123
    assert extra["correlation_id"] == "market-789"
