from __future__ import annotations

import math
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from gpt_trader.features.research.backtesting.metrics import PerformanceMetrics
from gpt_trader.features.research.backtesting.simulator import (
    BacktestResult,
    Position,
    SimulatedTrade,
)


def _trade(ts: datetime, side: str, price: str) -> SimulatedTrade:
    return SimulatedTrade(
        timestamp=ts,
        symbol="BTC-USD",
        side=side,
        quantity=Decimal("1"),
        price=Decimal(price),
        fee=Decimal("0"),
        reason="test",
    )


def test_from_result_with_empty_equity_curve_returns_zeroed_metrics() -> None:
    result = BacktestResult(
        trades=[],
        final_equity=Decimal("1000"),
        final_position=Position(symbol="BTC-USD"),
        equity_curve=[],
    )

    metrics = PerformanceMetrics.from_result(result)

    assert metrics.total_return == 0.0
    assert metrics.trade_count == 0
    assert metrics.max_drawdown == 0.0
    assert metrics.final_equity == 0.0


def test_calculate_drawdown_reports_max_and_duration() -> None:
    t0 = datetime(2024, 1, 1, 0, 0, 0)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)
    equity_curve = [(t0, 100.0), (t1, 80.0), (t2, 90.0)]

    stats = PerformanceMetrics._calculate_drawdown(equity_curve)

    assert stats["max_drawdown"] == pytest.approx(0.2)
    assert stats["max_duration"] == timedelta(hours=2)


def test_calculate_trade_pnls_pairs_and_skips_mismatches() -> None:
    base = datetime(2024, 1, 1, 0, 0, 0)
    trades = [
        _trade(base, "buy", "100"),
        _trade(base + timedelta(minutes=1), "buy", "101"),
        _trade(base + timedelta(minutes=2), "sell", "110"),
        _trade(base + timedelta(minutes=3), "sell", "120"),
        _trade(base + timedelta(minutes=4), "buy", "100"),
        _trade(base + timedelta(minutes=5), "sell", "90"),
    ]
    result = BacktestResult(
        trades=trades,
        final_equity=Decimal("1000"),
        final_position=Position(symbol="BTC-USD"),
    )

    pnls = PerformanceMetrics._calculate_trade_pnls(result)

    assert pnls == pytest.approx([9.0, 20.0])


def test_calculate_profit_factor_edges() -> None:
    assert math.isinf(PerformanceMetrics._calculate_profit_factor([1.0, 2.0]))
    assert PerformanceMetrics._calculate_profit_factor([-1.0, -2.0]) == 0.0
