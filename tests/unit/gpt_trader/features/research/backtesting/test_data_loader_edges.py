from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from gpt_trader.features.brokerages.coinbase.market_data_features import DepthSnapshot
from gpt_trader.features.research.backtesting.data_loader import (
    HistoricalDataLoader,
    HistoricalDataPoint,
)


class StubEventStore:
    def __init__(self, events: list[dict]) -> None:
        self._events = events

    def list_events(self) -> list[dict]:
        return list(self._events)


def test_load_symbol_filters_invalid_and_truncates() -> None:
    ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts2 = ts1 + timedelta(minutes=1)
    ts3 = ts1 + timedelta(minutes=2)
    ts4 = ts1 + timedelta(minutes=3)

    events = [
        {"type": "price_tick", "data": {"symbol": "BTC-USD", "timestamp": ts1}},
        {
            "type": "price_tick",
            "data": {"symbol": "BTC-USD", "timestamp": ts2, "price": "not-a-number"},
        },
        {
            "type": "price_tick",
            "data": {"symbol": "BTC-USD", "timestamp": ts3, "price": "101"},
        },
        {
            "type": "price_tick",
            "data": {"symbol": "BTC-USD", "timestamp": ts4, "price": "102"},
        },
        {
            "type": "price_tick",
            "data": {"symbol": "ETH-USD", "timestamp": ts3, "price": "999"},
        },
    ]

    loader = HistoricalDataLoader(StubEventStore(events))
    result = loader.load_symbol(
        "BTC-USD", max_points=1, include_orderbook=False, include_trade_flow=False
    )

    assert result.count == 1
    assert result.data_points[0].mark_price == Decimal("101")
    assert result.data_points[0].orderbook_snapshot is None
    assert result.data_points[0].trade_flow_stats is None


def test_load_symbol_include_gates_and_market_data() -> None:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    events = [
        {
            "type": "orderbook_snapshot",
            "data": {
                "symbol": "BTC-USD",
                "timestamp": ts,
                "mid_price": "100",
                "spread_bps": "10",
            },
        },
        {
            "type": "trade_flow_summary",
            "data": {
                "symbol": "BTC-USD",
                "timestamp": ts,
                "trade_count": 5,
                "volume": "10",
                "vwap": "100",
                "avg_size": "2",
                "aggressor_ratio": "0.6",
            },
        },
        {
            "type": "price_tick",
            "data": {"symbol": "BTC-USD", "timestamp": ts, "price": "100"},
        },
    ]

    loader = HistoricalDataLoader(StubEventStore(events))

    result_no = loader.load_symbol("BTC-USD", include_orderbook=False, include_trade_flow=False)
    point_no = result_no.data_points[0]
    assert point_no.orderbook_snapshot is None
    assert point_no.trade_flow_stats is None
    assert point_no.has_market_data() is False

    result_yes = loader.load_symbol("BTC-USD", include_orderbook=True, include_trade_flow=True)
    point_yes = result_yes.data_points[0]
    assert point_yes.orderbook_snapshot is not None
    assert point_yes.trade_flow_stats is not None
    assert point_yes.trade_flow_stats["count"] == 5
    assert point_yes.has_market_data() is True


def test_build_depth_snapshot_valid_and_invalid() -> None:
    loader = HistoricalDataLoader(StubEventStore([]))

    snapshot = loader._build_depth_snapshot({"mid_price": "100", "spread_bps": "10"})
    assert isinstance(snapshot, DepthSnapshot)
    assert snapshot.mid == Decimal("100")

    assert loader._build_depth_snapshot({"spread_bps": "10"}) is None
    assert loader._build_depth_snapshot({"mid_price": "bad", "spread_bps": "10"}) is None


def test_historical_data_point_has_market_data() -> None:
    base = HistoricalDataPoint(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        symbol="BTC-USD",
        mark_price=Decimal("100"),
        orderbook_snapshot=None,
        trade_flow_stats=None,
        spread_bps=None,
    )
    assert base.has_market_data() is False

    depth = DepthSnapshot([(Decimal("99"), Decimal("1"), "bid")])
    with_orderbook = HistoricalDataPoint(
        timestamp=base.timestamp,
        symbol=base.symbol,
        mark_price=base.mark_price,
        orderbook_snapshot=depth,
        trade_flow_stats=None,
        spread_bps=None,
    )
    assert with_orderbook.has_market_data() is True

    with_trade_flow = HistoricalDataPoint(
        timestamp=base.timestamp,
        symbol=base.symbol,
        mark_price=base.mark_price,
        orderbook_snapshot=None,
        trade_flow_stats={"count": 1},
        spread_bps=None,
    )
    assert with_trade_flow.has_market_data() is True


def test_load_all_symbols_returns_all_symbols() -> None:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    events = [
        {
            "type": "price_tick",
            "data": {"symbol": "BTC-USD", "timestamp": ts, "price": "100"},
        },
        {
            "type": "price_tick",
            "data": {"symbol": "ETH-USD", "timestamp": ts, "price": "50"},
        },
        {
            "type": "price_tick",
            "data": {
                "symbol": "BTC-USD",
                "timestamp": ts + timedelta(minutes=1),
                "price": "101",
            },
        },
    ]

    loader = HistoricalDataLoader(StubEventStore(events))
    results = loader.load_all_symbols()

    assert set(results.keys()) == {"BTC-USD", "ETH-USD"}
    assert results["BTC-USD"].count == 2
    assert results["ETH-USD"].count == 1
    assert results["BTC-USD"].data_points[0].mark_price == Decimal("100")
    assert results["ETH-USD"].data_points[0].mark_price == Decimal("50")


def test_load_all_symbols_single_list_events_call() -> None:
    """Regression: load_all_symbols must call list_events exactly once (no N+1)."""
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    symbols = [f"SYM{i}-USD" for i in range(10)]
    events = [
        {"type": "price_tick", "data": {"symbol": s, "timestamp": ts, "price": "1"}}
        for s in symbols
    ]

    call_count = 0

    class CountingEventStore:
        def __init__(self, store_events: list[dict]) -> None:
            self._events = store_events

        def list_events(self) -> list[dict]:
            nonlocal call_count
            call_count += 1
            return list(self._events)

    loader = HistoricalDataLoader(CountingEventStore(events))  # type: ignore[arg-type]
    results = loader.load_all_symbols()

    assert len(results) == 10
    assert call_count == 1
