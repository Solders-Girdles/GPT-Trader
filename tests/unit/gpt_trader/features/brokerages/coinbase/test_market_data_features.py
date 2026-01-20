from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

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
