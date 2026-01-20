from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, Mock

import pytest

import gpt_trader.features.brokerages.coinbase.client.client as client_module
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.client.market import MarketDataClientMixin
from gpt_trader.features.brokerages.coinbase.errors import (
    AuthError,
    InvalidRequestError,
    PermissionDeniedError,
)


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


def test_list_balances_maps_totals() -> None:
    client = _make_client()
    client.get_accounts = lambda: {  # type: ignore[method-assign]
        "accounts": [
            {
                "currency": "USD",
                "available_balance": {"value": "100.25"},
                "hold": {"value": "10.75"},
            },
            {
                "currency": "USDC",
                "available_balance": {"value": "50"},
                "hold": {"value": "0"},
            },
        ]
    }

    balances = client.list_balances()
    data = {balance.asset: balance for balance in balances}

    assert data["USD"].available == Decimal("100.25")
    assert data["USD"].hold == Decimal("10.75")
    assert data["USD"].total == Decimal("111.00")
    assert data["USDC"].total == Decimal("50")


def test_warn_public_market_fallback_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_client()
    mock_logger = MagicMock()
    monkeypatch.setattr(client_module, "logger", mock_logger)

    client._warn_public_market_fallback("ticker", Exception("unauthorized"))
    client._warn_public_market_fallback("ticker", Exception("unauthorized"))

    assert mock_logger.warning.call_count == 1


def test_get_ticker_auth_error_falls_back_to_public(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _make_client()
    mock_public = Mock(
        side_effect=[
            AuthError("unauthorized"),
            {"price": "101", "bid": "100", "ask": "102"},
        ]
    )
    monkeypatch.setattr(MarketDataClientMixin, "get_market_product_ticker", mock_public)

    caplog.set_level("WARNING")
    result = client.get_ticker("BTC-USD")

    assert result["price"] == "101"
    assert mock_public.call_count == 2
    assert "Falling back to public market endpoints" in caplog.text


def test_get_ticker_invalid_request_falls_back_to_authenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    mock_public = Mock(side_effect=InvalidRequestError("unsupported"))
    mock_private = Mock(return_value={"price": "99", "bid": "98", "ask": "100"})
    monkeypatch.setattr(MarketDataClientMixin, "get_market_product_ticker", mock_public)
    monkeypatch.setattr(MarketDataClientMixin, "get_ticker", mock_private)

    result = client.get_ticker("BTC-USD")

    assert result["price"] == "99"
    mock_public.assert_called_once()
    mock_private.assert_called_once()


def test_get_ticker_normalizes_from_trades_and_best_bid_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()

    response = {
        "trades": [{"price": "101", "time": "2024-01-01T00:00:00Z"}],
        "best_bid": "100",
        "best_ask": "102",
    }
    monkeypatch.setattr(
        MarketDataClientMixin, "get_market_product_ticker", Mock(return_value=response)
    )

    result = client.get_ticker("BTC-USD")

    assert result["price"] == "101"
    assert result["bid"] == "100"
    assert result["ask"] == "102"
    assert result["time"] == "2024-01-01T00:00:00Z"


def test_get_ticker_auth_error_raises_in_non_advanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(api_mode="exchange")
    monkeypatch.setattr(
        MarketDataClientMixin, "get_ticker", Mock(side_effect=AuthError("unauthorized"))
    )

    with pytest.raises(AuthError):
        client.get_ticker("BTC-USD")


def test_get_candles_permission_denied_falls_back_to_public(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client()
    mock_public = Mock(
        side_effect=[
            PermissionDeniedError("denied"),
            {"candles": [{"open": "1"}]},
        ]
    )
    monkeypatch.setattr(MarketDataClientMixin, "get_market_product_candles", mock_public)

    caplog.set_level("WARNING")
    result = client.get_candles("BTC-USD", "1H", limit=1)

    assert result["candles"] == [{"open": "1"}]
    assert mock_public.call_count == 2
    assert "Falling back to public market endpoints" in caplog.text


def test_get_candles_permission_denied_raises_in_non_advanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _make_client(api_mode="exchange")
    monkeypatch.setattr(
        MarketDataClientMixin,
        "get_candles",
        Mock(side_effect=PermissionDeniedError("denied")),
    )

    with pytest.raises(PermissionDeniedError):
        client.get_candles("BTC-USD", "1H", limit=1)
