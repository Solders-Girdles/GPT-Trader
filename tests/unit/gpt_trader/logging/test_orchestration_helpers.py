"""Tests for orchestration logging helpers."""

from __future__ import annotations

import logging
from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.logging.orchestration_helpers import (
    get_orchestration_logger,
    log_execution_error,
    log_market_data_update,
    log_order_event,
    log_risk_event,
    log_strategy_decision,
    log_trading_operation,
    with_order_context,
    with_symbol_context,
    with_trading_context,
)


def test_get_orchestration_logger():
    """Test getting an orchestration logger."""
    logger = get_orchestration_logger("test_component")
    assert logger.name == "gpt_trader.json.test_component"

    # Test with JSON disabled
    logger = get_orchestration_logger("test_component", enable_json=False)
    assert logger.name == "test_component"


@patch("gpt_trader.logging.orchestration_helpers.get_log_context")
def test_log_trading_operation(mock_get_log_context):
    """Test logging a trading operation."""
    # Mock get_log_context to return correlation_id and the expected symbol
    mock_get_log_context.return_value = {"correlation_id": "test-123", "symbol": "BTC-USD"}

    with patch("gpt_trader.logging.orchestration_helpers.get_orchestration_logger") as mock_logger:
        mock_log = MagicMock()
        mock_logger.return_value = mock_log

        log_trading_operation(
            operation="test_operation",
            symbol="BTC-USD",
            level=logging.INFO,
            test_field="test_value",
        )

        # Check that the logger was called with the correct context
        mock_log.log.assert_called_once()
        args, kwargs = mock_log.log.call_args
        assert args[0] == logging.INFO  # level
        assert args[1] == "test_operation"  # message
        assert "extra" in kwargs
        assert kwargs["extra"]["symbol"] == "BTC-USD"
        assert kwargs["extra"]["operation"] == "test_operation"
        assert kwargs["extra"]["test_field"] == "test_value"
        assert kwargs["extra"]["correlation_id"] == "test-123"


@patch("gpt_trader.logging.orchestration_helpers.get_log_context")
def test_log_order_event(mock_get_log_context):
    """Test logging an order event."""
    mock_get_log_context.return_value = {"correlation_id": "order-456"}

    with patch("gpt_trader.logging.orchestration_helpers.get_orchestration_logger") as mock_logger:
        mock_log = MagicMock()
        mock_logger.return_value = mock_log

        log_order_event(
            event_type="order_placed",
            order_id="order-123",
            symbol="ETH-USD",
            side="buy",
            quantity=Decimal("1.5"),
            price=Decimal("3000.25"),
            level=logging.INFO,
        )

        # Check that the logger was called with the correct context
        mock_log.log.assert_called_once()
        args, kwargs = mock_log.log.call_args
        assert args[0] == logging.INFO  # level
        assert args[1] == "Order event: order_placed"  # message
        assert "extra" in kwargs
        assert kwargs["extra"]["event_type"] == "order_placed"
        assert kwargs["extra"]["order_id"] == "order-123"
        assert kwargs["extra"]["symbol"] == "ETH-USD"
        assert kwargs["extra"]["side"] == "buy"
        assert kwargs["extra"]["quantity"] == 1.5
        assert kwargs["extra"]["price"] == 3000.25
        assert kwargs["extra"]["correlation_id"] == "order-456"


@patch("gpt_trader.logging.orchestration_helpers.get_log_context")
def test_log_strategy_decision(mock_get_log_context):
    """Test logging a strategy decision."""
    mock_get_log_context.return_value = {"correlation_id": "strategy-789"}

    with patch("gpt_trader.logging.orchestration_helpers.get_orchestration_logger") as mock_logger:
        mock_log = MagicMock()
        mock_logger.return_value = mock_log

        log_strategy_decision(
            symbol="BTC-USD",
            decision="BUY",
            reason="RSI oversold",
            confidence=0.85,
            level=logging.INFO,
        )

        # Check that the logger was called with the correct context
        mock_log.log.assert_called_once()
        args, kwargs = mock_log.log.call_args
        assert args[0] == logging.INFO  # level
        assert args[1] == "Strategy decision for BTC-USD: BUY"  # message
        assert "extra" in kwargs
        assert kwargs["extra"]["symbol"] == "BTC-USD"
        assert kwargs["extra"]["decision"] == "BUY"
        assert kwargs["extra"]["reason"] == "RSI oversold"
        assert kwargs["extra"]["confidence"] == 0.85
        assert kwargs["extra"]["correlation_id"] == "strategy-789"


@patch("gpt_trader.logging.orchestration_helpers.get_log_context")
def test_log_execution_error(mock_get_log_context):
    """Test logging an execution error."""
    mock_get_log_context.return_value = {"correlation_id": "error-123"}

    with patch("gpt_trader.logging.orchestration_helpers.get_orchestration_logger") as mock_logger:
        mock_log = MagicMock()
        mock_logger.return_value = mock_log

        test_error = ValueError("Test error message")

        log_execution_error(
            error=test_error,
            operation="place_order",
            symbol="ETH-USD",
            order_id="order-456",
            level=logging.ERROR,
        )

        # Check that the logger was called with the correct context
        mock_log.log.assert_called_once()
        args, kwargs = mock_log.log.call_args
        assert args[0] == logging.ERROR  # level
        assert args[1] == "Execution error in place_order: Test error message"  # message
        assert kwargs["exc_info"] is True
        assert "extra" in kwargs
        assert kwargs["extra"]["operation"] == "place_order"
        assert kwargs["extra"]["error_type"] == "ValueError"
        assert kwargs["extra"]["error_message"] == "Test error message"
        assert kwargs["extra"]["symbol"] == "ETH-USD"
        assert kwargs["extra"]["order_id"] == "order-456"
        assert kwargs["extra"]["correlation_id"] == "error-123"


@patch("gpt_trader.logging.orchestration_helpers.get_log_context")
def test_log_risk_event(mock_get_log_context):
    """Test logging a risk event."""
    mock_get_log_context.return_value = {"correlation_id": "risk-456"}

    with patch("gpt_trader.logging.orchestration_helpers.get_orchestration_logger") as mock_logger:
        mock_log = MagicMock()
        mock_logger.return_value = mock_log

        log_risk_event(
            event_type="circuit_breaker_triggered",
            symbol="BTC-USD",
            trigger_value=0.15,
            threshold=0.10,
            action="reduce_only",
            level=logging.WARNING,
        )

        # Check that the logger was called with the correct context
        mock_log.log.assert_called_once()
        args, kwargs = mock_log.log.call_args
        assert args[0] == logging.WARNING  # level
        assert args[1] == "Risk event: circuit_breaker_triggered"  # message
        assert "extra" in kwargs
        assert kwargs["extra"]["event_type"] == "circuit_breaker_triggered"
        assert kwargs["extra"]["symbol"] == "BTC-USD"
        assert kwargs["extra"]["trigger_value"] == "0.15"
        assert kwargs["extra"]["threshold"] == "0.10"
        assert kwargs["extra"]["action"] == "reduce_only"
        assert kwargs["extra"]["correlation_id"] == "risk-456"


@patch("gpt_trader.logging.orchestration_helpers.get_log_context")
def test_log_market_data_update(mock_get_log_context):
    """Test logging a market data update."""
    mock_get_log_context.return_value = {"correlation_id": "market-789"}

    with patch("gpt_trader.logging.orchestration_helpers.get_orchestration_logger") as mock_logger:
        mock_log = MagicMock()
        mock_logger.return_value = mock_log

        log_market_data_update(
            symbol="BTC-USD",
            price=Decimal("50000.50"),
            volume=Decimal("100.25"),
            timestamp=1640995200.123,
            level=logging.DEBUG,
        )

        # Check that the logger was called with the correct context
        mock_log.log.assert_called_once()
        args, kwargs = mock_log.log.call_args
        assert args[0] == logging.DEBUG  # level
        assert args[1] == "Market data update: BTC-USD"  # message
        assert "extra" in kwargs
        assert kwargs["extra"]["symbol"] == "BTC-USD"
        assert kwargs["extra"]["price"] == 50000.50
        assert kwargs["extra"]["volume"] == 100.25
        assert kwargs["extra"]["timestamp"] == 1640995200.123
        assert kwargs["extra"]["correlation_id"] == "market-789"


def test_with_trading_context_decorator():
    """Test the trading context decorator."""

    @with_trading_context("test_operation")
    def test_function():
        from gpt_trader.logging.correlation import get_correlation_id, get_domain_context

        return {
            "correlation_id": get_correlation_id(),
            "operation": get_domain_context().get("operation"),
        }

    result = test_function()
    assert result["operation"] == "test_operation"
    assert result["correlation_id"] != ""


def test_with_symbol_context_decorator():
    """Test the symbol context decorator."""

    @with_symbol_context("BTC-USD")
    def test_function():
        from gpt_trader.logging.correlation import get_domain_context

        return get_domain_context().get("symbol")

    result = test_function()
    assert result == "BTC-USD"


def test_with_order_context_decorator():
    """Test the order context decorator."""

    @with_order_context("order-123", "ETH-USD")
    def test_function():
        from gpt_trader.logging.correlation import get_domain_context

        return {
            "order_id": get_domain_context().get("order_id"),
            "symbol": get_domain_context().get("symbol"),
        }

    result = test_function()
    assert result["order_id"] == "order-123"
    assert result["symbol"] == "ETH-USD"
