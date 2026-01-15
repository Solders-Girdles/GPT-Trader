"""Edge case coverage for Coinbase model mappers."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as real_datetime
from decimal import Decimal
from unittest.mock import patch

import pytest

import gpt_trader.features.brokerages.coinbase.models as models
from gpt_trader.core import MarketType, OrderSide, OrderStatus, OrderType, TimeInForce
from gpt_trader.features.brokerages.coinbase.models import (
    to_candle,
    to_order,
    to_position,
    to_product,
    to_quote,
)


def test_to_product_perpetual_logs_invalid_funding_time() -> None:
    payload = {
        "product_id": "btc-perp",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "contract_type": "perpetual",
        "contract_size": "1",
        "funding_rate": "0.001",
        "next_funding_time": "not-a-date",
        "max_leverage": 5,
        "base_min_size": "0.01",
        "base_increment": "0.001",
        "quote_increment": "0.1",
    }

    with patch("gpt_trader.features.brokerages.coinbase.models.logger") as mock_logger:
        product = to_product(payload)

    assert product.symbol == "BTC-PERP"
    assert product.market_type == MarketType.PERPETUAL
    assert product.contract_size == Decimal("1")
    assert product.funding_rate == Decimal("0.001")
    assert product.leverage_max == 5
    assert product.next_funding_time is None
    mock_logger.error.assert_called_once()


def test_to_quote_uses_trade_fallback() -> None:
    payload = {
        "product_id": "btc-usd",
        "best_bid": "1",
        "best_ask": "2",
        "price": "",
        "trades": [{"price": "3", "time": "2024-01-01T00:00:00"}],
    }

    quote = to_quote(payload)

    assert quote.symbol == "BTC-USD"
    assert quote.last == Decimal("3")
    assert quote.ts == real_datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


def test_to_quote_logs_bad_trade_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_time = real_datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(models, "utc_now", lambda: fixed_time)

    payload = {"product_id": "BTC-USD", "trades": ["bad"]}

    with patch("gpt_trader.features.brokerages.coinbase.models.logger") as mock_logger:
        quote = to_quote(payload)

    assert quote.last == Decimal("0")
    assert quote.ts == fixed_time
    assert mock_logger.error.call_count >= 1


def test_to_order_stop_limit_with_ioc() -> None:
    payload = {
        "order_id": "o1",
        "product_id": "eth-usd",
        "side": "SELL",
        "type": "stop_limit",
        "time_in_force": "IOC",
        "size": "2",
        "price": "2000",
        "stop_price": "1900",
        "status": "CANCELED",
        "filled_size": "1",
        "average_filled_price": "1950",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:10",
    }

    order = to_order(payload)

    assert order.id == "o1"
    assert order.side == OrderSide.SELL
    assert order.type == OrderType.STOP_LIMIT
    assert order.tif == TimeInForce.IOC
    assert order.status == OrderStatus.CANCELLED
    assert order.filled_quantity == Decimal("1")
    assert order.avg_fill_price == Decimal("1950")
    assert order.stop_price == Decimal("1900")


def test_to_order_contracts_quantity_fallback() -> None:
    payload = {
        "id": "o2",
        "product_id": "btc-perp",
        "side": "buy",
        "type": "market",
        "contracts": "3",
        "status": "open",
    }

    order = to_order(payload)

    assert order.quantity == Decimal("3")
    assert order.type == OrderType.MARKET


def test_to_position_side_from_quantity() -> None:
    payload = {
        "product_id": "btc-perp",
        "size": "-2",
        "entry_price": "100",
        "mark_price": "110",
        "unrealized_pnl": "5",
        "realized_pnl": "1",
    }

    position = to_position(payload)

    assert position.side == "short"
    assert position.quantity == Decimal("2")
    assert position.entry_price == Decimal("100")
    assert position.mark_price == Decimal("110")


def test_to_position_respects_side_override() -> None:
    payload = {
        "symbol": "eth-perp",
        "size": "1",
        "side": "short",
        "avg_entry_price": "1500",
        "index_price": "1490",
        "unrealizedPnl": "-3",
        "realizedPnl": "2",
        "leverage": 3,
    }

    position = to_position(payload)

    assert position.symbol == "ETH-PERP"
    assert position.side == "short"
    assert position.leverage == 3


def test_to_candle_uses_ts_field() -> None:
    candle = to_candle(
        {
            "ts": "2024-01-02T00:00:00",
            "open": "1",
            "high": "2",
            "low": "0.5",
            "close": "1.5",
            "volume": "10",
        }
    )

    assert candle.ts == real_datetime(2024, 1, 2, 0, 0, 0, tzinfo=UTC)
