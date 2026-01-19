"""Tests for runtime logging helper context decorators."""

from __future__ import annotations

from gpt_trader.logging.runtime_helpers import (
    with_order_context,
    with_symbol_context,
    with_trading_context,
)


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
