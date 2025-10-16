from __future__ import annotations

from collections import defaultdict

import pandas as pd
import pytest

from bot_v2.features.analyze import analyze


def test_fetch_data_normalizes_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    source = pd.DataFrame({"Close": [101, 102], "High": [103, 104], "Low": [99, 100]})

    class Provider:
        def get_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
            assert symbol == "AAPL"
            assert period == "30d"
            return source.copy()

    monkeypatch.setattr(analyze, "get_data_provider", lambda: Provider())

    result = analyze.fetch_data("AAPL", 30)

    assert list(result.columns) == ["close", "high", "low"]
    pd.testing.assert_series_equal(result["close"], source["Close"], check_names=False)


def test_calculate_correlations_builds_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    price_series = {
        "AAPL": pd.DataFrame({"close": [100, 101, 102, 103]}),
        "MSFT": pd.DataFrame({"close": [50, 49, 51, 52]}),
    }

    class Provider:
        def get_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
            return price_series[symbol]

    monkeypatch.setattr(analyze, "get_data_provider", lambda: Provider())

    matrix = analyze.calculate_correlations(["AAPL", "MSFT"], lookback_days=5)

    expect = pd.DataFrame(
        [
            [
                1.0,
                price_series["AAPL"]["close"]
                .pct_change()
                .dropna()
                .corr(price_series["MSFT"]["close"].pct_change().dropna()),
            ],
            [
                price_series["MSFT"]["close"]
                .pct_change()
                .dropna()
                .corr(price_series["AAPL"]["close"].pct_change().dropna()),
                1.0,
            ],
        ],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    pd.testing.assert_frame_equal(matrix, expect)


def test_calculate_correlations_handles_failed_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = defaultdict(int)

        def get_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
            self.calls[symbol] += 1
            if symbol == "BAD":
                raise RuntimeError("boom")
            return pd.DataFrame({"close": [1.0, 1.1, 1.2]})

    provider = Provider()
    monkeypatch.setattr(analyze, "get_data_provider", lambda: provider)

    matrix = analyze.calculate_correlations(["GOOD", "BAD"], lookback_days=3)

    assert provider.calls["BAD"] == 1
    assert list(matrix.columns) == ["GOOD"]
    assert matrix.loc["GOOD", "GOOD"] == pytest.approx(1.0)


def test_backtest_strategy_returns_random_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    sequence = iter([0.1, 1.5, 0.05, 0.6])
    monkeypatch.setattr(analyze.np.random, "uniform", lambda *args, **kwargs: next(sequence))

    metrics = analyze.backtest_strategy("Demo", pd.DataFrame())

    assert metrics == {
        "return": 0.1,
        "sharpe": 1.5,
        "max_drawdown": 0.05,
        "win_rate": 0.6,
    }
