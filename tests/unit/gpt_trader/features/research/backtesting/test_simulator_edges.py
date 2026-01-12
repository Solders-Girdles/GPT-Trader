from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.types import FeeTier
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


class BuyThenCloseStrategy:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, *args, **kwargs) -> Decision:
        self.calls += 1
        if self.calls == 1:
            return Decision(action=Action.BUY, reason="enter")
        return Decision(action=Action.CLOSE, reason="exit")


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
        spread_bps=20.0,
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


def test_tiered_fees_default() -> None:
    config = BacktestConfig(
        initial_equity=Decimal("10000"),
        fee_rate_bps=1,
        use_tiered_fees=True,
        fee_tier=FeeTier.TIER_2,
        slippage_bps=0,
        use_spread_slippage=False,
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

    position_value = config.initial_equity * Decimal(str(config.position_size_pct))
    fee_model = FeeCalculator(tier=config.fee_tier)
    expected_fee = fee_model.calculate(notional_usd=position_value, is_maker=False)

    assert result.trades[0].fee == expected_fee
    assert result.trades[1].fee == expected_fee
    assert result.final_equity == config.initial_equity - (expected_fee * 2)


def test_funding_accrual_and_settlement() -> None:
    config = BacktestConfig(
        initial_equity=Decimal("10000"),
        fee_rate_bps=0,
        use_tiered_fees=False,
        slippage_bps=0,
        use_spread_slippage=False,
        funding_rates_8h={"BTC-USD": Decimal("0.0008")},
        funding_accrual_hours=1,
        funding_settlement_hours=1,
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
            timestamp=start + timedelta(hours=1),
            symbol="BTC-USD",
            mark_price=Decimal("100"),
            orderbook_snapshot=None,
            trade_flow_stats=None,
            spread_bps=None,
        ),
    ]

    result = simulator.run(HoldAfterBuyStrategy(), data_points)

    position_value = config.initial_equity * Decimal(str(config.position_size_pct))
    assert config.funding_rates_8h is not None
    funding_rate = config.funding_rates_8h["BTC-USD"]
    expected_funding = position_value * funding_rate / Decimal("8")

    assert result.final_equity == config.initial_equity - expected_funding
