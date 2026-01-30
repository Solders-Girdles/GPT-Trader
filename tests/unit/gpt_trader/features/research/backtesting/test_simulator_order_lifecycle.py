from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from gpt_trader.core.trading import OrderStatus
from gpt_trader.core import Action, Decision
from gpt_trader.features.research.backtesting.adapter import BacktestSimulator
from gpt_trader.features.research.backtesting.data_loader import HistoricalDataPoint
from gpt_trader.features.research.backtesting.simulator import BacktestConfig


class HoldAfterBuyStrategy:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, *args, **kwargs) -> Decision:
        self.calls += 1
        if self.calls == 1:
            return Decision(action=Action.BUY, reason="enter")
        return Decision(action=Action.HOLD, reason="hold")


class BuyThenCloseStrategy:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, *args, **kwargs) -> Decision:
        self.calls += 1
        if self.calls == 1:
            return Decision(action=Action.BUY, reason="enter")
        return Decision(action=Action.CLOSE, reason="exit")


def test_pending_order_fills_after_delay() -> None:
    config = BacktestConfig(
        initial_equity=Decimal("10000"),
        fee_rate_bps=0,
        use_tiered_fees=False,
        slippage_bps=0,
        use_spread_slippage=False,
        order_fill_delay_bars=1,
    )
    simulator = BacktestSimulator(config=config)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data_points = [
        HistoricalDataPoint(
            timestamp=start,
            symbol="BTC-USD",
            mark_price=Decimal("100"),
            orderbook_snapshot=None,
            trade_flow_stats=None,
            spread_bps=None,
        ),
        HistoricalDataPoint(
            timestamp=start + timedelta(minutes=1),
            symbol="BTC-USD",
            mark_price=Decimal("100"),
            orderbook_snapshot=None,
            trade_flow_stats=None,
            spread_bps=None,
        ),
        HistoricalDataPoint(
            timestamp=start + timedelta(minutes=2),
            symbol="BTC-USD",
            mark_price=Decimal("100"),
            orderbook_snapshot=None,
            trade_flow_stats=None,
            spread_bps=None,
        ),
    ]

    result = simulator.run(HoldAfterBuyStrategy(), data_points)

    assert len(result.orders) == 1
    order = result.orders[0]
    assert order.status == OrderStatus.FILLED
    assert order.submitted_at == data_points[0].timestamp
    assert order.filled_at == data_points[1].timestamp


def test_cancel_pending_order_on_close() -> None:
    config = BacktestConfig(
        initial_equity=Decimal("10000"),
        fee_rate_bps=0,
        use_tiered_fees=False,
        slippage_bps=0,
        use_spread_slippage=False,
        order_fill_delay_bars=2,
    )
    simulator = BacktestSimulator(config=config)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data_points = [
        HistoricalDataPoint(
            timestamp=start,
            symbol="BTC-USD",
            mark_price=Decimal("100"),
            orderbook_snapshot=None,
            trade_flow_stats=None,
            spread_bps=None,
        ),
        HistoricalDataPoint(
            timestamp=start + timedelta(minutes=1),
            symbol="BTC-USD",
            mark_price=Decimal("100"),
            orderbook_snapshot=None,
            trade_flow_stats=None,
            spread_bps=None,
        ),
    ]

    result = simulator.run(BuyThenCloseStrategy(), data_points)

    assert len(result.orders) == 1
    order = result.orders[0]
    assert order.status == OrderStatus.CANCELLED
    assert order.cancel_reason
    assert order.cancelled_at == data_points[1].timestamp
