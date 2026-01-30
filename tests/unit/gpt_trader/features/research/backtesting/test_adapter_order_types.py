from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from gpt_trader.core import Action, Decision
from gpt_trader.core.trading import OrderStatus
from gpt_trader.features.research.backtesting.adapter import BacktestSimulator
from gpt_trader.features.research.backtesting.data_loader import HistoricalDataPoint
from gpt_trader.features.research.backtesting.simulator import BacktestConfig


class LimitBuyStrategy:
    def __init__(self) -> None:
        self._called = False

    def decide(self, *args, **kwargs) -> Decision:
        if not self._called:
            self._called = True
            return Decision(
                action=Action.BUY,
                reason="limit_entry",
                indicators={
                    "order_type": "limit",
                    "price": Decimal("100"),
                },
            )
        return Decision(action=Action.HOLD, reason="hold")


class IocLimitStrategy:
    def __init__(self) -> None:
        self._called = False

    def decide(self, *args, **kwargs) -> Decision:
        if not self._called:
            self._called = True
            return Decision(
                action=Action.BUY,
                reason="ioc_entry",
                indicators={
                    "order_type": "limit",
                    "price": Decimal("90"),
                    "tif": "IOC",
                },
            )
        return Decision(action=Action.HOLD, reason="hold")


class StopBuyStrategy:
    def __init__(self) -> None:
        self._called = False

    def decide(self, *args, **kwargs) -> Decision:
        if not self._called:
            self._called = True
            return Decision(
                action=Action.BUY,
                reason="stop_entry",
                indicators={
                    "order_type": "stop",
                    "stop_price": Decimal("101"),
                },
            )
        return Decision(action=Action.HOLD, reason="hold")


class StopLimitBuyStrategy:
    def __init__(self) -> None:
        self._called = False

    def decide(self, *args, **kwargs) -> Decision:
        if not self._called:
            self._called = True
            return Decision(
                action=Action.BUY,
                reason="stop_limit_entry",
                indicators={
                    "order_type": "stop_limit",
                    "price": Decimal("100"),
                    "stop_price": Decimal("101"),
                },
            )
        return Decision(action=Action.HOLD, reason="hold")


class StopBuyNeverTriggersStrategy:
    def __init__(self) -> None:
        self._called = False

    def decide(self, *args, **kwargs) -> Decision:
        if not self._called:
            self._called = True
            return Decision(
                action=Action.BUY,
                reason="stop_no_trigger",
                indicators={
                    "order_type": "stop",
                    "stop_price": Decimal("150"),
                },
            )
        return Decision(action=Action.HOLD, reason="hold")


def _points(start: datetime, prices: list[str]) -> list[HistoricalDataPoint]:
    points = []
    for idx, price in enumerate(prices):
        points.append(
            HistoricalDataPoint(
                timestamp=start + timedelta(minutes=idx),
                symbol="BTC-USD",
                mark_price=Decimal(price),
                orderbook_snapshot=None,
                trade_flow_stats=None,
                spread_bps=None,
            )
        )
    return points


def test_limit_order_fills_on_next_bar() -> None:
    config = BacktestConfig(
        initial_equity=Decimal("10000"),
        fee_rate_bps=0,
        use_tiered_fees=False,
        slippage_bps=0,
        use_spread_slippage=False,
    )
    simulator = BacktestSimulator(config=config)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data_points = _points(start, ["100", "100"])

    result = simulator.run(LimitBuyStrategy(), data_points)

    assert len(result.orders) == 1
    order = result.orders[0]
    assert order.status == OrderStatus.FILLED
    assert order.filled_at == data_points[1].timestamp
    assert any(trade.reason == "limit_entry" for trade in result.trades)


def test_ioc_limit_order_cancels_when_not_touched() -> None:
    config = BacktestConfig(
        initial_equity=Decimal("10000"),
        fee_rate_bps=0,
        use_tiered_fees=False,
        slippage_bps=0,
        use_spread_slippage=False,
    )
    simulator = BacktestSimulator(config=config)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data_points = _points(start, ["100"])

    result = simulator.run(IocLimitStrategy(), data_points)

    assert len(result.orders) == 1
    order = result.orders[0]
    assert order.status == OrderStatus.CANCELLED
    assert len(result.trades) == 0


def test_stop_order_triggers_on_next_bar() -> None:
    config = BacktestConfig(
        initial_equity=Decimal("10000"),
        fee_rate_bps=0,
        use_tiered_fees=False,
        slippage_bps=0,
        use_spread_slippage=False,
    )
    simulator = BacktestSimulator(config=config)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data_points = _points(start, ["100", "101"])

    result = simulator.run(StopBuyStrategy(), data_points)

    assert len(result.orders) == 1
    order = result.orders[0]
    assert order.status == OrderStatus.FILLED
    assert order.filled_at == data_points[1].timestamp
    assert any(trade.reason == "stop_entry" for trade in result.trades)


def test_stop_limit_order_triggers_and_fills() -> None:
    config = BacktestConfig(
        initial_equity=Decimal("10000"),
        fee_rate_bps=0,
        use_tiered_fees=False,
        slippage_bps=0,
        use_spread_slippage=False,
    )
    simulator = BacktestSimulator(config=config)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data_points = _points(start, ["100", "101"])

    result = simulator.run(StopLimitBuyStrategy(), data_points)

    assert len(result.orders) == 1
    order = result.orders[0]
    assert order.order_type == "stop_limit"
    assert order.status == OrderStatus.FILLED
    assert order.filled_at == data_points[1].timestamp


def test_stop_order_does_not_trigger_stays_open() -> None:
    config = BacktestConfig(
        initial_equity=Decimal("10000"),
        fee_rate_bps=0,
        use_tiered_fees=False,
        slippage_bps=0,
        use_spread_slippage=False,
    )
    simulator = BacktestSimulator(config=config)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data_points = _points(start, ["100", "101"])

    result = simulator.run(StopBuyNeverTriggersStrategy(), data_points)

    assert len(result.orders) == 1
    order = result.orders[0]
    assert order.status == OrderStatus.SUBMITTED
    assert len(result.trades) == 0
