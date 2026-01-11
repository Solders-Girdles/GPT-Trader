"""Edge-case tests for correlation context management."""

from __future__ import annotations

from gpt_trader.logging.correlation import (
    correlation_context,
    get_correlation_id,
    get_domain_context,
    get_log_context,
    set_correlation_id,
    set_domain_context,
)


def test_correlation_context_nested_restores_prior_values() -> None:
    set_correlation_id("root-id")
    set_domain_context({"symbol": "BTC-USD", "stage": "root"})

    with correlation_context("outer-id", symbol="ETH-USD"):
        assert get_correlation_id() == "outer-id"
        assert get_domain_context()["symbol"] == "ETH-USD"
        assert get_domain_context()["stage"] == "root"

        with correlation_context("inner-id", order_id="order-123"):
            assert get_correlation_id() == "inner-id"
            assert get_domain_context()["symbol"] == "ETH-USD"
            assert get_domain_context()["order_id"] == "order-123"

        assert get_correlation_id() == "outer-id"
        assert "order_id" not in get_domain_context()
        assert get_domain_context()["symbol"] == "ETH-USD"

    assert get_correlation_id() == "root-id"
    assert get_domain_context() == {"symbol": "BTC-USD", "stage": "root"}


def test_set_correlation_id_and_domain_context_handle_none() -> None:
    set_correlation_id(None)
    set_domain_context(None)

    assert get_correlation_id() == ""
    assert get_domain_context() == {}


def test_get_log_context_returns_copy() -> None:
    set_correlation_id("ctx-id")
    set_domain_context({"symbol": "BTC-USD"})

    context = get_log_context()
    context["symbol"] = "ETH-USD"
    context["correlation_id"] = "mutated"

    assert get_domain_context()["symbol"] == "BTC-USD"
    assert get_log_context()["correlation_id"] == "ctx-id"


def test_reset_correlation_context_autouse_starts_clean() -> None:
    assert get_log_context() == {}

    set_correlation_id("dirty")
    set_domain_context({"symbol": "BTC-USD"})
    assert get_log_context()["symbol"] == "BTC-USD"


def test_reset_correlation_context_autouse_clears_between_tests() -> None:
    assert get_log_context() == {}
