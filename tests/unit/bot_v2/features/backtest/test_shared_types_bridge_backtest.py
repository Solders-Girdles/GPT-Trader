from datetime import datetime
from decimal import Decimal

import pandas as pd

from bot_v2.features.backtest.types import BacktestMetrics, BacktestResult


def make_result() -> BacktestResult:
    metrics = BacktestMetrics(
        total_return=12.5,
        sharpe_ratio=1.1,
        max_drawdown=5.0,
        win_rate=55.0,
        total_trades=20,
        profit_factor=1.4,
    )
    trades = [
        {
            "id": 1,
            "date": pd.Timestamp("2024-01-01T10:00:00Z"),
            "side": "buy",
            "price": 100.0,
            "quantity": 10,
            "commission": 1.0,
        },
        {
            "id": 2,
            "date": pd.Timestamp("2024-01-05T10:00:00Z"),
            "side": "sell",
            "price": 110.0,
            "quantity": 10,
            "commission": 1.1,
        },
    ]
    index = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])
    equity_curve = pd.Series([1000.0, 1010.0, 1110.0], index=index)
    returns = equity_curve.pct_change().fillna(0)

    return BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        returns=returns,
        metrics=metrics,
        initial_capital=1000.0,
    )


def test_metrics_to_performance_summary() -> None:
    metrics = BacktestMetrics(
        total_return=10.0,
        sharpe_ratio=1.2,
        max_drawdown=3.0,
        win_rate=60.0,
        total_trades=15,
        profit_factor=1.5,
    )

    summary = metrics.to_performance_summary()
    assert summary.total_return == 10.0
    assert summary.trades_count == 15
    assert summary.win_rate == 60.0


def test_backtest_result_final_equity() -> None:
    result = make_result()
    assert result.final_equity() == 1110.0


def test_backtest_result_to_trading_session() -> None:
    result = make_result()
    session = result.to_trading_session(symbol="BTC-USD", account_id="acct-test")

    assert session.account.account_id == "acct-test"
    assert session.account.equity == Decimal("1110.0")
    assert session.performance is not None
    assert session.fills[0].symbol == "BTC-USD"
    assert session.fills[0].quantity == Decimal("10")
    assert session.fills[0].side.value == "buy"

    # ensure timestamps converted
    assert isinstance(session.start_time, datetime)
    assert isinstance(session.end_time, datetime)
