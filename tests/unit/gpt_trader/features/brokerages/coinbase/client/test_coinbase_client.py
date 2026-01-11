from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.client.market import MarketDataClientMixin
from gpt_trader.features.brokerages.coinbase.errors import AuthError, PermissionDeniedError


def _make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def test_list_positions_filters_zero_and_maps_side() -> None:
    client = _make_client()

    client.list_cfm_positions = lambda: {  # type: ignore[method-assign]
        "positions": [
            {
                "product_id": "BTC-USD",
                "net_size": "1.5",
                "entry_price": "100",
                "mark_price": "105",
                "unrealized_pnl": "7.5",
                "leverage": 3,
            },
            {
                "product_id": "ETH-USD",
                "net_size": "-2",
                "entry_price": "2000",
                "mark_price": "1950",
                "unrealized_pnl": "-100",
                "leverage": 5,
            },
            {
                "product_id": "SOL-USD",
                "net_size": "0",
                "entry_price": "20",
                "mark_price": "20",
                "unrealized_pnl": "0",
                "leverage": 1,
            },
        ]
    }

    positions = client.list_positions()

    assert len(positions) == 2
    assert positions[0].symbol == "BTC-USD"
    assert positions[0].quantity == Decimal("1.5")
    assert positions[0].side == "long"
    assert positions[1].symbol == "ETH-USD"
    assert positions[1].quantity == Decimal("2")
    assert positions[1].side == "short"


def test_list_balances_warns_when_accounts_missing(caplog: pytest.LogCaptureFixture) -> None:
    client = _make_client()
    client.get_accounts = lambda: {"accounts": []}  # type: ignore[method-assign]

    caplog.set_level("WARNING")
    balances = client.list_balances()

    assert balances == []
    assert "API returned response but no accounts" in caplog.text


def test_get_ticker_auth_error_falls_back_to_public(caplog: pytest.LogCaptureFixture) -> None:
    client = _make_client()

    caplog.set_level("WARNING")
    with patch.object(
        MarketDataClientMixin,
        "get_market_product_ticker",
        side_effect=[
            AuthError("unauthorized"),
            {"price": "101", "bid": "100", "ask": "102"},
        ],
    ) as mock_public:
        result = client.get_ticker("BTC-USD")

    assert result["price"] == "101"
    assert mock_public.call_count == 2
    assert "Falling back to public market endpoints" in caplog.text


def test_get_ticker_auth_error_raises_in_non_advanced() -> None:
    client = _make_client(api_mode="exchange")

    with patch.object(MarketDataClientMixin, "get_ticker", side_effect=AuthError("unauthorized")):
        with pytest.raises(AuthError):
            client.get_ticker("BTC-USD")


def test_get_candles_permission_denied_falls_back_to_public(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _make_client()

    caplog.set_level("WARNING")
    with patch.object(
        MarketDataClientMixin,
        "get_market_product_candles",
        side_effect=[
            PermissionDeniedError("denied"),
            {"candles": [{"open": "1"}]},
        ],
    ) as mock_public:
        result = client.get_candles("BTC-USD", "1H", limit=1)

    assert result["candles"] == [{"open": "1"}]
    assert mock_public.call_count == 2
    assert "Falling back to public market endpoints" in caplog.text


def test_get_candles_permission_denied_raises_in_non_advanced() -> None:
    client = _make_client(api_mode="exchange")

    with patch.object(
        MarketDataClientMixin,
        "get_candles",
        side_effect=PermissionDeniedError("denied"),
    ):
        with pytest.raises(PermissionDeniedError):
            client.get_candles("BTC-USD", "1H", limit=1)
