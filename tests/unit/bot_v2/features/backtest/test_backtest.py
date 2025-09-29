"""
Modern backtest tests that exercise the current slice API.

These tests avoid network by relying on the MockProvider via TESTING=true
and validate both happy-path and validation/error behaviors.
"""

from datetime import datetime, timedelta
import os

import pandas as pd

import pytest
from bot_v2.features.backtest import run_backtest, BacktestResult
from bot_v2.errors import StrategyError, ValidationError as RichValidationError
from bot_v2.errors import ValidationError as LegacyValidationError

ValidationErrors = (RichValidationError, LegacyValidationError)
ValidationOrStrategyErrors = ValidationErrors + (StrategyError,)


def _date_range(days: int = 90):
    end = datetime.now()
    start = end - timedelta(days=days)
    return start, end


def test_run_backtest_success_with_mock_provider(monkeypatch):
    # Force MockProvider to avoid network and ensure determinism
    monkeypatch.setenv("TESTING", "true")

    start, end = _date_range(90)

    # No monkeypatching required: use MockProvider and built-in validators

    result = run_backtest(
        strategy="SimpleMAStrategy",
        symbol="AAPL",
        start=start,
        end=end,
        initial_capital=10_000.0,
        commission=0.001,
        slippage=0.0005,
        fast_period=10,
        slow_period=30,
    )

    # Structure checks
    assert isinstance(result, BacktestResult)
    assert isinstance(result.trades, list)
    assert isinstance(result.equity_curve, pd.Series)
    assert isinstance(result.returns, pd.Series)
    assert result.metrics is not None

    # Basic sanity on outputs
    assert len(result.equity_curve) > 0
    assert len(result.returns) == len(result.equity_curve)
    assert result.equity_curve.index.is_monotonic_increasing

    # Metrics presence and types
    m = result.metrics
    assert hasattr(m, "total_return")
    assert hasattr(m, "sharpe_ratio")
    assert hasattr(m, "max_drawdown")
    assert hasattr(m, "total_trades")


def test_run_backtest_invalid_date_range_raises(monkeypatch):
    monkeypatch.setenv("TESTING", "true")

    end = datetime.now()
    start = end - timedelta(days=1)  # Intentionally too short for min_data_points

    with pytest.raises(ValidationErrors):
        run_backtest(
            strategy="SimpleMAStrategy",
            symbol="AAPL",
            start=start,
            end=end,
        )


def test_run_backtest_rejects_unknown_strategy(monkeypatch):
    monkeypatch.setenv("TESTING", "true")
    start, end = _date_range(90)

    # Unknown strategy name should cause an error; current validation may surface as ValidationError
    with pytest.raises(ValidationOrStrategyErrors):
        run_backtest(
            strategy="NotARealStrategy",
            symbol="AAPL",
            start=start,
            end=end,
        )
