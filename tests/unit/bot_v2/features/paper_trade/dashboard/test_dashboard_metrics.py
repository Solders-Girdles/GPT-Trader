"""Tests for dashboard metrics assembler."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from bot_v2.features.paper_trade.dashboard.metrics import DashboardMetricsAssembler


@pytest.fixture
def mock_engine() -> SimpleNamespace:
    return SimpleNamespace(
        calculate_equity=lambda: 110_000.0,
        initial_capital=100_000.0,
        cash=50_000.0,
        positions={
            "AAPL": SimpleNamespace(quantity=1.5, entry_price=100.0, current_price=120.0),
            "MSFT": SimpleNamespace(quantity=2.0, entry_price=200.0, current_price=0.0),
        },
        trades=[
            SimpleNamespace(pnl=500.0),
            SimpleNamespace(pnl=-200.0),
            SimpleNamespace(pnl=300.0),
        ],
    )


def test_metrics_returns(mock_engine: SimpleNamespace) -> None:
    assembler = DashboardMetricsAssembler(initial_equity=100_000.0)
    metrics = assembler.calculate(mock_engine)
    assert metrics["equity"] == 110_000.0
    assert pytest.approx(metrics["returns_pct"], rel=1e-6) == 10.0


def test_metrics_win_rate(mock_engine: SimpleNamespace) -> None:
    assembler = DashboardMetricsAssembler(initial_equity=100_000.0)
    metrics = assembler.calculate(mock_engine)
    assert metrics["winning_trades"] == 2
    assert metrics["losing_trades"] == 1
    assert pytest.approx(metrics["win_rate"], rel=1e-6) == (2 / 3) * 100


def test_metrics_exposure(mock_engine: SimpleNamespace) -> None:
    assembler = DashboardMetricsAssembler(initial_equity=100_000.0)
    metrics = assembler.calculate(mock_engine)
    # Positions value: 1.5*120 + 2*200 = 180 + 400 = 580
    assert pytest.approx(metrics["positions_value"], rel=1e-6) == 580.0
    assert pytest.approx(metrics["exposure_pct"], rel=1e-6) == (580.0 / 110_000.0) * 100


def test_metrics_zero_initial_equity(mock_engine: SimpleNamespace) -> None:
    assembler = DashboardMetricsAssembler(initial_equity=0.0)
    metrics = assembler.calculate(mock_engine)
    assert metrics["returns_pct"] == 0.0


def test_metrics_no_trades(mock_engine: SimpleNamespace) -> None:
    mock_engine.trades = []
    assembler = DashboardMetricsAssembler(initial_equity=100_000.0)
    metrics = assembler.calculate(mock_engine)
    assert metrics["winning_trades"] == 0
    assert metrics["losing_trades"] == 0
    assert metrics["win_rate"] == 0.0
