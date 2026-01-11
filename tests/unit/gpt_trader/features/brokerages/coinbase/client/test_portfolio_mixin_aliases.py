from __future__ import annotations

from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.portfolio import PortfolioClientMixin


class _PortfolioClient(CoinbaseClientBase, PortfolioClientMixin):
    pass


def _make_client(api_mode: str = "advanced") -> _PortfolioClient:
    return _PortfolioClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def test_list_positions_raw_aliases_cfm_positions() -> None:
    client = _make_client()
    client.list_cfm_positions = Mock(return_value={"positions": []})  # type: ignore[attr-defined]

    result = client.list_positions_raw()

    assert result == {"positions": []}
    client.list_cfm_positions.assert_called_once_with()  # type: ignore[attr-defined]


def test_get_intx_portfolio_aliases_intx_portfolio() -> None:
    client = _make_client()
    client.intx_portfolio = Mock(return_value={"id": "pf-1"})  # type: ignore[attr-defined]

    result = client.get_intx_portfolio("pf-1")

    assert result == {"id": "pf-1"}
    client.intx_portfolio.assert_called_once_with("pf-1")  # type: ignore[attr-defined]


def test_list_intx_positions_aliases_intx_positions() -> None:
    client = _make_client()
    client.intx_positions = Mock(return_value={"positions": []})  # type: ignore[attr-defined]

    result = client.list_intx_positions("pf-1")

    assert result == {"positions": []}
    client.intx_positions.assert_called_once_with("pf-1")  # type: ignore[attr-defined]


def test_get_intx_position_aliases_intx_position() -> None:
    client = _make_client()
    client.intx_position = Mock(return_value={"symbol": "BTC-USD"})  # type: ignore[attr-defined]

    result = client.get_intx_position("pf-1", "BTC-USD")

    assert result == {"symbol": "BTC-USD"}
    client.intx_position.assert_called_once_with("pf-1", "BTC-USD")  # type: ignore[attr-defined]


def test_get_intx_multi_asset_collateral_aliases_method() -> None:
    client = _make_client()
    client.intx_multi_asset_collateral = Mock(return_value={"ok": True})  # type: ignore[attr-defined]

    result = client.get_intx_multi_asset_collateral()

    assert result == {"ok": True}
    client.intx_multi_asset_collateral.assert_called_once_with()  # type: ignore[attr-defined]
