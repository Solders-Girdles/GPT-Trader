from __future__ import annotations

from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.portfolio import PortfolioClientMixin


class _PortfolioClient(CoinbaseClientBase, PortfolioClientMixin):
    pass


def _make_client(api_mode: str = "advanced") -> _PortfolioClient:
    return _PortfolioClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


@pytest.mark.parametrize(
    ("method_name", "args", "expected_method", "expected_path", "expected_payload"),
    [
        ("list_portfolios", (), "GET", "/api/v3/brokerage/portfolios", None),
        (
            "get_portfolio",
            ("pf-1",),
            "GET",
            "/api/v3/brokerage/portfolios/pf-1",
            None,
        ),
        (
            "get_portfolio_breakdown",
            ("pf-1",),
            "GET",
            "/api/v3/brokerage/portfolios/pf-1/breakdown",
            None,
        ),
        (
            "move_funds",
            ({"amount": "1"},),
            "POST",
            "/api/v3/brokerage/portfolios/move_funds",
            {"amount": "1"},
        ),
        (
            "convert_quote",
            ({"side": "buy"},),
            "POST",
            "/api/v3/brokerage/convert/quote",
            {"side": "buy"},
        ),
        (
            "get_convert_trade",
            ("trade-1",),
            "GET",
            "/api/v3/brokerage/convert/trade/trade-1",
            None,
        ),
        (
            "list_payment_methods",
            (),
            "GET",
            "/api/v3/brokerage/payment_methods",
            None,
        ),
        (
            "get_payment_method",
            ("pm-1",),
            "GET",
            "/api/v3/brokerage/payment_methods/pm-1",
            None,
        ),
        (
            "list_cfm_positions",
            (),
            "GET",
            "/api/v3/brokerage/cfm/positions",
            None,
        ),
        (
            "get_cfm_position",
            ("BTC-USD",),
            "GET",
            "/api/v3/brokerage/cfm/positions/BTC-USD",
            None,
        ),
        (
            "intx_allocate",
            ({"amount": "1"},),
            "POST",
            "/api/v3/brokerage/intx/allocate",
            {"amount": "1"},
        ),
        (
            "intx_balances",
            ("pf-1",),
            "GET",
            "/api/v3/brokerage/intx/balances/pf-1",
            None,
        ),
        (
            "intx_portfolio",
            ("pf-1",),
            "GET",
            "/api/v3/brokerage/intx/portfolio/pf-1",
            None,
        ),
        (
            "intx_positions",
            ("pf-1",),
            "GET",
            "/api/v3/brokerage/intx/positions/pf-1",
            None,
        ),
        (
            "intx_position",
            ("pf-1", "BTC-USD"),
            "GET",
            "/api/v3/brokerage/intx/positions/pf-1/BTC-USD",
            None,
        ),
        (
            "intx_multi_asset_collateral",
            (),
            "GET",
            "/api/v3/brokerage/intx/multi_asset_collateral",
            None,
        ),
        (
            "cfm_balance_summary",
            (),
            "GET",
            "/api/v3/brokerage/cfm/balance_summary",
            None,
        ),
        (
            "cfm_positions",
            (),
            "GET",
            "/api/v3/brokerage/cfm/positions",
            None,
        ),
        (
            "cfm_position",
            ("BTC-USD",),
            "GET",
            "/api/v3/brokerage/cfm/positions/BTC-USD",
            None,
        ),
        (
            "cfm_sweeps",
            (),
            "GET",
            "/api/v3/brokerage/cfm/sweeps",
            None,
        ),
        (
            "cfm_sweeps_schedule",
            (),
            "GET",
            "/api/v3/brokerage/cfm/sweeps/schedule",
            None,
        ),
        (
            "cfm_intraday_current_margin_window",
            (),
            "GET",
            "/api/v3/brokerage/cfm/intraday/current_margin_window",
            None,
        ),
        (
            "cfm_intraday_margin_setting",
            ({"risk": "high"},),
            "POST",
            "/api/v3/brokerage/cfm/intraday/margin_setting",
            {"risk": "high"},
        ),
    ],
)
def test_portfolio_requests(
    method_name: str,
    args: tuple,
    expected_method: str,
    expected_path: str,
    expected_payload: dict[str, str] | None,
) -> None:
    client = _make_client()
    client._request = Mock(return_value={})  # type: ignore[attr-defined]

    getattr(client, method_name)(*args)

    if expected_payload is None:
        client._request.assert_called_once_with(  # type: ignore[attr-defined]
            expected_method,
            expected_path,
        )
    else:
        client._request.assert_called_once_with(  # type: ignore[attr-defined]
            expected_method,
            expected_path,
            expected_payload,
        )
