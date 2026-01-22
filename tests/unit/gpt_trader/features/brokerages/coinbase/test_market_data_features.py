from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.market_data_features import (
    DepthSnapshot,
    RollingWindow,
    TradeTapeAgg,
    get_expected_perps,
)


@dataclass
class _OrderbookEvent:
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


def test_rolling_window_empty_stats() -> None:
    window = RollingWindow(duration_seconds=10)

    stats = window.get_stats()

    assert stats == {"sum": 0.0, "count": 0, "avg": 0.0}


def test_rolling_window_default_timestamp() -> None:
    window = RollingWindow(duration_seconds=5)

    window.add(3.0)

    stats = window.get_stats()
    assert stats["count"] == 1
    assert stats["sum"] == 3.0


def test_depth_snapshot_from_orderbook_update_sorts_levels() -> None:
    event = _OrderbookEvent(
        bids=[(100.0, 1.0), (101.0, 2.0)],
        asks=[(105.0, 1.0), (103.0, 2.0)],
    )

    snapshot = DepthSnapshot.from_orderbook_update(event)

    assert snapshot.bids[0] == (Decimal("101.0"), Decimal("2.0"))
    assert snapshot.asks[0] == (Decimal("103.0"), Decimal("2.0"))


def test_depth_snapshot_mid_none_when_missing_side() -> None:
    snapshot = DepthSnapshot([(Decimal("10"), Decimal("1"), "buy")])

    assert snapshot.mid is None


def test_depth_snapshot_spread_bps_zero_bid() -> None:
    snapshot = DepthSnapshot(
        [(Decimal("0"), Decimal("1"), "bid"), (Decimal("10"), Decimal("1"), "ask")]
    )

    assert snapshot.spread_bps == 0.0


def test_trade_tape_defaults_no_trades() -> None:
    agg = TradeTapeAgg(duration_seconds=60)

    assert agg.get_vwap() == Decimal("0")
    assert agg.get_avg_size() == Decimal("0")
    assert agg.get_aggressor_ratio() == 0.0
    assert agg.get_stats() == {
        "count": 0,
        "volume": Decimal("0"),
        "vwap": Decimal("0"),
        "avg_size": Decimal("0"),
        "aggressor_ratio": 0.0,
    }


def test_trade_tape_cleanup_removes_old_trades() -> None:
    agg = TradeTapeAgg(duration_seconds=10)
    base_time = datetime(2024, 1, 1, 12, 0, 0)

    agg.add_trade(Decimal("100"), Decimal("1"), "buy", base_time)
    agg.add_trade(Decimal("101"), Decimal("1"), "sell", base_time + timedelta(seconds=11))

    assert len(agg.trades) == 1
    assert agg.trades[0]["price"] == Decimal("101")


def test_trade_tape_zero_volume_vwap() -> None:
    agg = TradeTapeAgg(duration_seconds=30)

    agg.add_trade(Decimal("100"), Decimal("0"), "buy", datetime(2024, 1, 1, 12, 0, 0))

    assert agg.get_vwap() == Decimal("0")


def test_trade_tape_default_timestamp() -> None:
    agg = TradeTapeAgg(duration_seconds=30)

    agg.add_trade(Decimal("100"), Decimal("1"), "buy")

    assert len(agg.trades) == 1


def test_get_expected_perps() -> None:
    assert get_expected_perps() == {"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"}


class _DummyAuth:
    def __init__(self) -> None:
        self.called = False

    def get_headers(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.called = True
        return {"Authorization": "Bearer test"}


def _call_ticker(client: CoinbaseClient) -> dict:
    return client.get_ticker("BTC-USD")


def _call_candles(client: CoinbaseClient) -> dict:
    return client.get_candles("BTC-USD", "1H", limit=2)


def _assert_payload(result: dict, payload: dict) -> None:
    if "price" in payload:
        assert result.get("price") == payload["price"]
    else:
        assert result.get("candles") == payload["candles"]


_REQUESTS = [
    (
        _call_ticker,
        "/api/v3/brokerage/market/products/BTC-USD/ticker",
        "/api/v3/brokerage/products/BTC-USD/ticker",
        {"price": "100", "bid": "99", "ask": "101"},
    ),
    (
        _call_candles,
        "/api/v3/brokerage/market/products/BTC-USD/candles?granularity=1H&limit=2",
        "/api/v3/brokerage/products/BTC-USD/candles?granularity=1H&limit=2",
        {"candles": []},
    ),
]


@pytest.mark.parametrize("call_fn, public_suffix, _auth_suffix, payload", _REQUESTS)
def test_public_endpoint_used_when_unauthenticated(
    call_fn, public_suffix: str, _auth_suffix: str, payload: dict
) -> None:
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        return 200, {}, json.dumps(payload)

    client.set_transport_for_testing(transport)

    result = call_fn(client)

    _assert_payload(result, payload)
    assert len(calls) == 1
    assert calls[0].endswith(public_suffix)


@pytest.mark.parametrize("call_fn, public_suffix, _auth_suffix, payload", _REQUESTS)
def test_public_endpoint_used_when_authenticated(
    call_fn, public_suffix: str, _auth_suffix: str, payload: dict
) -> None:
    auth = _DummyAuth()
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        return 200, {}, json.dumps(payload)

    client.set_transport_for_testing(transport)

    result = call_fn(client)

    _assert_payload(result, payload)
    assert len(calls) == 1
    assert calls[0].endswith(public_suffix)
    assert auth.called is False


@pytest.mark.parametrize("call_fn, public_suffix, auth_suffix, payload", _REQUESTS)
def test_public_not_found_falls_back_to_authenticated(
    call_fn, public_suffix: str, auth_suffix: str, payload: dict
) -> None:
    auth = _DummyAuth()
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        if url.endswith(public_suffix):
            return 404, {}, json.dumps({"message": "not found"})
        return 200, {}, json.dumps(payload)

    client.set_transport_for_testing(transport)

    result = call_fn(client)

    _assert_payload(result, payload)
    assert len(calls) == 2
    assert calls[0].endswith(public_suffix)
    assert calls[1].endswith(auth_suffix)
    assert auth.called is True
