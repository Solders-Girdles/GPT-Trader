"""Tests for correlation ID management."""

from __future__ import annotations

from gpt_trader.logging.correlation import (
    add_domain_field,
    correlation_context,
    generate_correlation_id,
    get_correlation_id,
    get_domain_context,
    get_log_context,
    order_context,
    set_correlation_id,
    set_domain_context,
    symbol_context,
    update_domain_context,
)


def test_generate_correlation_id():
    """Test that correlation IDs are generated correctly."""
    correlation_id = generate_correlation_id()
    assert isinstance(correlation_id, str)
    assert len(correlation_id) > 0

    # Test that multiple calls generate different IDs
    correlation_id2 = generate_correlation_id()
    assert correlation_id != correlation_id2


def test_set_get_correlation_id():
    """Test setting and getting correlation IDs."""
    # Test with a specific ID
    test_id = "test-correlation-id-123"
    set_correlation_id(test_id)
    assert get_correlation_id() == test_id

    # Test with empty context (should return empty string)
    set_correlation_id("")
    assert get_correlation_id() == ""


def test_domain_context():
    """Test domain context management."""
    # Test setting domain context
    test_context = {"symbol": "BTC-USD", "side": "buy"}
    set_domain_context(test_context)
    assert get_domain_context() == test_context

    # Test updating domain context
    update_domain_context(quantity=100, price=50000)
    expected = {"symbol": "BTC-USD", "side": "buy", "quantity": 100, "price": 50000}
    assert get_domain_context() == expected

    # Test adding a single field
    add_domain_field("order_id", "order-123")
    expected["order_id"] = "order-123"
    assert get_domain_context() == expected


def test_get_log_context():
    """Test getting complete log context."""
    # Set correlation ID and domain context
    set_correlation_id("test-id-456")
    set_domain_context({"symbol": "ETH-USD", "side": "sell"})

    context = get_log_context()
    expected = {"correlation_id": "test-id-456", "symbol": "ETH-USD", "side": "sell"}
    assert context == expected

    # Test with empty context
    set_correlation_id("")
    set_domain_context({})
    assert get_log_context() == {}


def test_correlation_context_manager():
    """Test the correlation context manager."""
    # Set initial context
    set_correlation_id("initial-id")
    set_domain_context({"initial": "value"})

    # Test context manager
    with correlation_context("test-context-id", operation="test"):
        assert get_correlation_id() == "test-context-id"
        context = get_domain_context()
        assert context["operation"] == "test"

    # Context should be restored after exiting
    assert get_correlation_id() == "initial-id"
    context = get_domain_context()
    assert context["initial"] == "value"
    assert "operation" not in context


def test_correlation_context_without_id():
    """Test correlation context manager without providing an ID."""
    with correlation_context():
        correlation_id = get_correlation_id()
        assert correlation_id != ""
        assert len(correlation_id) > 0


def test_symbol_context_manager():
    """Test the symbol context manager."""
    # Test with additional fields
    with symbol_context("BTC-USD", side="buy", quantity=1.5):
        context = get_domain_context()
        assert context["symbol"] == "BTC-USD"
        assert context["side"] == "buy"
        assert context["quantity"] == 1.5


def test_order_context_manager():
    """Test the order context manager."""
    # Test with symbol
    with order_context("order-789", "ETH-USD", side="sell"):
        context = get_domain_context()
        assert context["order_id"] == "order-789"
        assert context["symbol"] == "ETH-USD"
        assert context["side"] == "sell"

    # Test without symbol
    with order_context("order-456"):
        context = get_domain_context()
        assert context["order_id"] == "order-456"
        assert "symbol" not in context


def test_nested_context_managers():
    """Test nesting context managers."""
    with correlation_context("outer-id", operation="outer"):
        assert get_correlation_id() == "outer-id"
        assert get_domain_context()["operation"] == "outer"

        with symbol_context("BTC-USD", side="buy"):
            assert get_correlation_id() == "outer-id"  # Should preserve
            context = get_domain_context()
            assert context["symbol"] == "BTC-USD"
            assert context["side"] == "buy"
            assert context["operation"] == "outer"  # Should preserve

        # After symbol context, should still have outer context
        assert get_correlation_id() == "outer-id"
        assert get_domain_context()["operation"] == "outer"
        assert "symbol" not in get_domain_context()
