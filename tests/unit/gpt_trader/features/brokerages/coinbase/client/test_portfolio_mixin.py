from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.portfolio import PortfolioClientMixin
from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError


def _make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


class _PortfolioClient(CoinbaseClientBase, PortfolioClientMixin):
    pass


def _make_portfolio_client(api_mode: str = "advanced") -> _PortfolioClient:
    return _PortfolioClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("list_portfolios", ()),
        ("get_portfolio", ("pf-1",)),
        ("get_portfolio_breakdown", ("pf-1",)),
        ("move_funds", ({"amount": "1"},)),
        ("list_payment_methods", ()),
        ("get_payment_method", ("pm-1",)),
        ("commit_convert_trade", ("trade-1",)),
        ("list_cfm_positions", ()),
        ("get_cfm_position", ("BTC-USD",)),
        ("list_positions_raw", ()),
    ],
)
def test_exchange_mode_blocks_portfolio_methods(method_name: str, args: tuple) -> None:
    client = _make_client(api_mode="exchange")

    with pytest.raises(InvalidRequestError):
        getattr(client, method_name)(*args)


def test_commit_convert_trade_defaults_to_empty_payload() -> None:
    client = _make_client()
    captured: dict[str, object] = {}

    def transport(method, url, headers, body, timeout):
        captured["method"] = method
        captured["url"] = url
        captured["body"] = json.loads(body or b"{}")
        return 200, {}, json.dumps({"trade": {"id": "trade-1"}})

    client.set_transport_for_testing(transport)
    _ = client.commit_convert_trade("trade-1")

    assert captured["method"] == "POST"
    assert str(captured["url"]).endswith("/api/v3/brokerage/convert/trade/trade-1")
    assert captured["body"] == {}


def test_list_positions_raw_aliases_cfm_positions() -> None:
    client = _make_portfolio_client()
    client.list_cfm_positions = Mock(return_value={"positions": []})  # type: ignore[attr-defined]

    result = client.list_positions_raw()

    assert result == {"positions": []}
    client.list_cfm_positions.assert_called_once_with()  # type: ignore[attr-defined]


def test_get_intx_portfolio_aliases_intx_portfolio() -> None:
    client = _make_portfolio_client()
    client.intx_portfolio = Mock(return_value={"id": "pf-1"})  # type: ignore[attr-defined]

    result = client.get_intx_portfolio("pf-1")

    assert result == {"id": "pf-1"}
    client.intx_portfolio.assert_called_once_with("pf-1")  # type: ignore[attr-defined]


def test_list_intx_positions_aliases_intx_positions() -> None:
    client = _make_portfolio_client()
    client.intx_positions = Mock(return_value={"positions": []})  # type: ignore[attr-defined]

    result = client.list_intx_positions("pf-1")

    assert result == {"positions": []}
    client.intx_positions.assert_called_once_with("pf-1")  # type: ignore[attr-defined]


def test_get_intx_position_aliases_intx_position() -> None:
    client = _make_portfolio_client()
    client.intx_position = Mock(return_value={"symbol": "BTC-USD"})  # type: ignore[attr-defined]

    result = client.get_intx_position("pf-1", "BTC-USD")

    assert result == {"symbol": "BTC-USD"}
    client.intx_position.assert_called_once_with("pf-1", "BTC-USD")  # type: ignore[attr-defined]


def test_get_intx_multi_asset_collateral_aliases_method() -> None:
    client = _make_portfolio_client()
    client.intx_multi_asset_collateral = Mock(return_value={"ok": True})  # type: ignore[attr-defined]

    result = client.get_intx_multi_asset_collateral()

    assert result == {"ok": True}
    client.intx_multi_asset_collateral.assert_called_once_with()  # type: ignore[attr-defined]
