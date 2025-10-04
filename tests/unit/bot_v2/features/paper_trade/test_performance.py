"""Unit tests for paper trading performance tracking."""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from bot_v2.features.paper_trade.performance import (
    PerformanceCalculator,
    PerformanceTracker,
    ResultBuilder,
)
from bot_v2.features.paper_trade.types import AccountStatus, PerformanceMetrics


@pytest.fixture
def tracker() -> PerformanceTracker:
    return PerformanceTracker(initial_capital=100_000)


@pytest.fixture
def calculator(tracker: PerformanceTracker) -> PerformanceCalculator:
    return PerformanceCalculator(tracker)


@pytest.fixture
def mock_account() -> SimpleNamespace:
    return SimpleNamespace(total_equity=105_000)


@pytest.fixture
def result_builder(tracker: PerformanceTracker, calculator: PerformanceCalculator) -> ResultBuilder:
    return ResultBuilder(tracker=tracker, calculator=calculator)


def test_tracker_records_snapshots(tracker: PerformanceTracker) -> None:
    timestamp = datetime.utcnow()
    tracker.record(timestamp, 101_000)

    assert len(tracker.snapshots()) == 1
    assert tracker.legacy_history == [{"timestamp": timestamp, "equity": 101_000.0}]


def test_tracker_equity_series_empty(tracker: PerformanceTracker) -> None:
    series = tracker.equity_series()
    assert isinstance(series, pd.Series)
    assert series.iloc[0] == 100_000


def test_tracker_equity_series_with_history(tracker: PerformanceTracker) -> None:
    now = datetime.utcnow()
    tracker.record(now, 101_000)
    tracker.record(now + timedelta(minutes=1), 102_500)

    series = tracker.equity_series()
    assert list(series.values) == [101_000.0, 102_500.0]


def test_calculator_total_return_with_history(
    tracker: PerformanceTracker, calculator: PerformanceCalculator, mock_account: SimpleNamespace
) -> None:
    tracker.record(datetime.utcnow(), 110_000)

    metrics = calculator.calculate(trade_log=[], account_status=mock_account)
    assert isinstance(metrics, PerformanceMetrics)
    assert pytest.approx(metrics.total_return, rel=1e-6) == 0.10


def test_calculator_total_return_without_history(
    calculator: PerformanceCalculator, mock_account: SimpleNamespace
) -> None:
    metrics = calculator.calculate(trade_log=[], account_status=mock_account)
    assert pytest.approx(metrics.total_return, rel=1e-6) == 0.05


def test_calculator_drawdown(
    tracker: PerformanceTracker, calculator: PerformanceCalculator
) -> None:
    now = datetime.utcnow()
    tracker.record(now, 100_000)
    tracker.record(now + timedelta(minutes=1), 120_000)
    tracker.record(now + timedelta(minutes=2), 90_000)

    account = SimpleNamespace(total_equity=90_000)
    metrics = calculator.calculate(trade_log=[], account_status=account)
    assert pytest.approx(metrics.max_drawdown, rel=1e-6) == 0.25


def test_calculator_win_rate(
    tracker: PerformanceTracker,
    calculator: PerformanceCalculator,
) -> None:
    tracker.record(datetime.utcnow(), 100_000)
    tracker.record(datetime.utcnow() + timedelta(minutes=1), 102_000)

    trades = [
        SimpleNamespace(id=1, side="buy", price=100, quantity=1),
        SimpleNamespace(id=2, side="sell", price=110, quantity=1),
        SimpleNamespace(id=3, side="buy", price=90, quantity=1),
        SimpleNamespace(id=4, side="sell", price=80, quantity=1),
    ]
    account = SimpleNamespace(total_equity=102_000)

    metrics = calculator.calculate(trade_log=trades, account_status=account)
    assert metrics.win_rate == 0.5
    assert metrics.profit_factor == 1


def test_result_builder_creates_paper_result(
    tracker: PerformanceTracker,
    calculator: PerformanceCalculator,
    result_builder: ResultBuilder,
    mock_account: SimpleNamespace,
) -> None:
    start = datetime.utcnow()
    end = start + timedelta(hours=1)
    tracker.record(start, 100_000)
    tracker.record(end, 105_000)

    trades: list[SimpleNamespace] = []
    paper_result = result_builder.build_paper_result(
        start_time=start,
        end_time=end,
        account_status=mock_account,
        positions=[],
        trade_log=trades,
    )

    assert paper_result.start_time == start
    assert paper_result.end_time == end
    assert isinstance(paper_result.performance, PerformanceMetrics)
    assert len(paper_result.equity_curve) == 2


def test_result_builder_creates_trading_session(
    tracker: PerformanceTracker,
    calculator: PerformanceCalculator,
    result_builder: ResultBuilder,
) -> None:
    start = datetime.utcnow()
    tracker.record(start, 100_000)
    account = SimpleNamespace(total_equity=100_000)

    account_status = AccountStatus(
        cash=100_000,
        positions_value=0,
        total_equity=account.total_equity,
        buying_power=200_000,
        margin_used=0,
        day_trades_remaining=3,
        positions=None,
        realized_pnl=0.0,
    )

    session_result = result_builder.build_trading_session(
        start_time=start,
        end_time=None,
        account_status=account_status,
        positions=[],
        trade_log=[],
    )

    assert session_result.start_time == start
    assert session_result.positions == []


def test_calculator_handles_no_trades(
    tracker: PerformanceTracker, calculator: PerformanceCalculator
) -> None:
    tracker.record(datetime.utcnow(), 100_000)
    account = SimpleNamespace(total_equity=100_000)

    metrics = calculator.calculate(trade_log=[], account_status=account)
    assert metrics.win_rate == 0
    assert metrics.profit_factor == 0


def test_calculator_handles_all_wins(
    tracker: PerformanceTracker, calculator: PerformanceCalculator
) -> None:
    tracker.record(datetime.utcnow(), 100_000)
    trades = [
        SimpleNamespace(id=1, side="buy", price=100, quantity=1),
        SimpleNamespace(id=2, side="sell", price=120, quantity=1),
    ]
    account = SimpleNamespace(total_equity=120_000)

    metrics = calculator.calculate(trade_log=trades, account_status=account)
    assert metrics.profit_factor == float("inf")
    assert metrics.win_rate == 1
