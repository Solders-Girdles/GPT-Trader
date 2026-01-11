from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
    Action,
    Decision,
)
from gpt_trader.features.research.backtesting.data_loader import HistoricalDataPoint
from gpt_trader.features.research.backtesting.simulator import (
    BacktestConfig,
    BacktestSimulator,
)


class HoldAfterBuyStrategy:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, *args, **kwargs) -> Decision:
        self.calls += 1
        if self.calls == 1:
            return Decision(action=Action.BUY, reason="enter")
        return Decision(action=Action.HOLD, reason="hold")


class NeverCalledStrategy:
    def decide(self, *args, **kwargs) -> Decision:
        raise AssertionError("Strategy should not be called for empty data")


def test_run_empty_data_points_returns_defaults() -> None:
    simulator = BacktestSimulator()
    result = simulator.run(NeverCalledStrategy(), [], symbol=None)

    assert result.final_equity == simulator.config.initial_equity
    assert result.trade_count == 0
    assert result.final_position.symbol == "UNKNOWN"


def test_calculate_fill_price_uses_spread_slippage() -> None:
    simulator = BacktestSimulator(config=BacktestConfig())
    point = HistoricalDataPoint(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        symbol="BTC-USD",
        mark_price=Decimal("100"),
        orderbook_snapshot=None,
        trade_flow_stats=None,
        spread_bps=Decimal("20"),
    )

    fill_price = simulator._calculate_fill_price(point, "buy")

    expected_slippage = Decimal("10") / Decimal("10000")
    expected = Decimal("100") * (Decimal("1") + expected_slippage)
    assert fill_price == expected


def test_run_buy_then_hold_closes_at_end() -> None:
    simulator = BacktestSimulator(config=BacktestConfig())
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
            mark_price=Decimal("101"),
            orderbook_snapshot=None,
            trade_flow_stats=None,
            spread_bps=None,
        ),
    ]

    result = simulator.run(HoldAfterBuyStrategy(), data_points)

    assert result.trade_count == 2
    assert len(result.equity_curve) == len(data_points)
    assert result.final_position.side == "flat"
