"""Tests for derivatives_discovery US and INTX market behavior."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.derivatives_discovery import (
    discover_derivatives_eligibility,
)


def _make_us_broker(
    balance_summary=None,
    *,
    has_method: bool = True,
    side_effect: Exception | None = None,
) -> Mock:
    if not has_method:
        return Mock(spec=[])
    broker = Mock()
    if side_effect is not None:
        broker.get_cfm_balance_summary.side_effect = side_effect
    else:
        broker.get_cfm_balance_summary.return_value = balance_summary
    return broker


def _make_intx_broker(
    portfolios,
    *,
    get_portfolio=None,
    list_side_effect: Exception | None = None,
    spec=None,
) -> Mock:
    broker = Mock(spec=spec) if spec is not None else Mock()
    if list_side_effect is not None:
        broker.list_portfolios.side_effect = list_side_effect
    else:
        broker.list_portfolios.return_value = portfolios
    if hasattr(broker, "get_intx_portfolio"):
        broker.get_intx_portfolio.return_value = get_portfolio
    return broker


@pytest.mark.parametrize(
    (
        "balance_summary",
        "side_effect",
        "has_method",
        "expected_enabled",
        "expected_accessible",
        "expected_reduce_only",
        "error_substring",
    ),
    [
        ({"balance": "1000"}, None, True, True, True, False, None),
        (None, None, True, True, False, True, "CFM balance summary unavailable"),
        ("invalid", None, True, True, False, None, None),
        ({}, None, True, True, False, None, "CFM balance summary unavailable"),
        (None, RuntimeError("API error"), True, True, False, True, "CFM discovery error"),
        (
            None,
            AttributeError("Missing attr"),
            True,
            False,
            False,
            None,
            "CFM methods not available",
        ),
        (None, None, False, False, False, True, "Broker does not support CFM endpoints"),
    ],
)
def test_us_market_variants(
    balance_summary,
    side_effect,
    has_method: bool,
    expected_enabled: bool,
    expected_accessible: bool,
    expected_reduce_only: bool | None,
    error_substring: str | None,
) -> None:
    broker = _make_us_broker(
        balance_summary,
        has_method=has_method,
        side_effect=side_effect,
    )

    result = discover_derivatives_eligibility(broker, requested_market="US")

    assert result.us_derivatives_enabled is expected_enabled
    assert result.cfm_portfolio_accessible is expected_accessible
    if expected_reduce_only is not None:
        assert result.reduce_only_required is expected_reduce_only
    if error_substring is not None:
        assert error_substring in (result.error_message or "")


@pytest.mark.parametrize(
    "portfolios",
    [
        [{"uuid": "portfolio-123", "type": "perpetuals", "name": "My INTX"}],
        [{"uuid": "perp-123", "type": "trading", "name": "My INTX Perpetuals"}],
        [{"uuid": "intx-123", "type": "INTX", "name": "Trading"}],
        ["invalid", {"uuid": "perp-123", "type": "perpetuals", "name": "INTX"}],
    ],
)
def test_intx_market_accessible_variants(portfolios) -> None:
    broker = _make_intx_broker(portfolios, get_portfolio={"balance": "1000"})

    result = discover_derivatives_eligibility(broker, requested_market="INTX")

    assert result.intx_derivatives_enabled is True
    assert result.intx_portfolio_accessible is True
    assert result.reduce_only_required is False


@pytest.mark.parametrize(
    ("portfolios", "error_substring", "expected_reduce_only"),
    [
        ([], "No portfolios accessible", True),
        (
            [{"uuid": "spot-123", "type": "spot", "name": "Spot Trading"}],
            "No INTX portfolio found",
            None,
        ),
        ([{"type": "perpetuals", "name": "INTX"}], "INTX portfolio missing UUID", None),
        ({"error": "unexpected format"}, None, None),
    ],
)
def test_intx_market_portfolio_errors(
    portfolios,
    error_substring: str | None,
    expected_reduce_only: bool | None,
) -> None:
    broker = _make_intx_broker(portfolios, get_portfolio=None)

    result = discover_derivatives_eligibility(broker, requested_market="INTX")

    assert result.intx_derivatives_enabled is True
    assert result.intx_portfolio_accessible is False
    if error_substring is not None:
        assert error_substring in (result.error_message or "")
    if expected_reduce_only is not None:
        assert result.reduce_only_required is expected_reduce_only


def test_intx_market_missing_list_portfolios_method() -> None:
    broker = Mock(spec=[])

    result = discover_derivatives_eligibility(broker, requested_market="INTX")

    assert result.intx_derivatives_enabled is False
    assert result.intx_portfolio_accessible is False
    assert "Broker does not support portfolio listing" in (result.error_message or "")
    assert result.reduce_only_required is True


def test_intx_market_missing_get_intx_portfolio() -> None:
    broker = _make_intx_broker(
        [{"uuid": "perp-123", "type": "perpetuals", "name": "INTX"}],
        spec=["list_portfolios"],
    )

    result = discover_derivatives_eligibility(broker, requested_market="INTX")

    assert result.intx_portfolio_accessible is False
    assert "INTX portfolio endpoints not available" in (result.error_message or "")


@pytest.mark.parametrize(
    ("side_effect", "expected_enabled", "error_substring"),
    [
        (RuntimeError("API error"), True, "INTX discovery error"),
        (AttributeError("Missing attr"), False, "INTX methods not available"),
    ],
)
def test_intx_market_list_portfolios_errors(
    side_effect: Exception,
    expected_enabled: bool,
    error_substring: str,
) -> None:
    broker = _make_intx_broker([], list_side_effect=side_effect)

    result = discover_derivatives_eligibility(broker, requested_market="INTX")

    assert result.intx_derivatives_enabled is expected_enabled
    assert result.intx_portfolio_accessible is False
    assert error_substring in (result.error_message or "")
