from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import bot_v2.features.optimize.optimize as optimize_module
from bot_v2.features.optimize.optimize import (
    fetch_data,
    grid_search,
    optimize_strategy,
    walk_forward_analysis,
)
from bot_v2.features.optimize.types import OptimizationResult


def _stub_provider(data):
    return SimpleNamespace(get_historical_data=lambda symbol, period: data)


def _result_for(strategy, symbol, start, end, metrics):
    return OptimizationResult(
        strategy=strategy,
        symbol=symbol,
        period=(start, end),
        best_params={"lookback": 5},
        best_metrics=metrics,
        all_results=[
            {"params": {"lookback": 5}, "metrics": metrics, "score": metrics.sharpe_ratio}
        ],
        optimization_time=0.01,
    )


def test_optimize_strategy_selects_best_params(
    monkeypatch,
    seeded_ohlc_sets,
    backtest_metrics_factory,
):
    data = seeded_ohlc_sets["uptrend"]
    provider = _stub_provider(data)

    monkeypatch.setattr(optimize_module, "get_data_provider", lambda: provider)
    monkeypatch.setattr(
        optimize_module,
        "get_strategy_params",
        lambda strategy: {"fast": [5, 10], "slow": [20]},
    )

    metrics_low = backtest_metrics_factory.build(sharpe_ratio=0.5, total_return=0.05)
    metrics_high = backtest_metrics_factory.build(sharpe_ratio=1.2, total_return=0.15)

    def fake_runner(strategy, dataset, params, commission, slippage):
        return metrics_high if params["fast"] == 10 else metrics_low

    monkeypatch.setattr(optimize_module, "run_backtest_local", fake_runner)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    start = data.index.min().to_pydatetime()
    end = data.index.max().to_pydatetime()

    result = optimize_strategy("momentum", "BTC-USD", start, end)

    assert result.best_params == {"fast": 10, "slow": 20}
    assert result.best_metrics is metrics_high
    assert len(result.all_results) == 2
    assert result.best_metrics.sharpe_ratio == pytest.approx(1.2)


def test_optimize_strategy_raises_when_no_combinations(monkeypatch, seeded_ohlc_sets):
    data = seeded_ohlc_sets["flat"]
    monkeypatch.setattr(optimize_module, "get_data_provider", lambda: _stub_provider(data))
    monkeypatch.setattr(
        optimize_module,
        "get_strategy_params",
        lambda strategy: {"length": []},
    )
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    with pytest.raises(ValueError):
        optimize_strategy(
            "mean_reversion",
            "ETH-USD",
            datetime(2024, 1, 1),
            datetime(2024, 1, 10),
        )


def test_fetch_data_filters_and_standardizes(monkeypatch, seeded_ohlc_sets):
    data = seeded_ohlc_sets["downtrend"].copy()
    data.index = data.index.tz_localize(None)
    data.columns = [col.upper() for col in data.columns]

    provider = _stub_provider(data)
    monkeypatch.setattr(optimize_module, "get_data_provider", lambda: provider)

    start = data.index.min() + (data.index.freq * 2)
    end = data.index.max() - (data.index.freq * 2)

    filtered = fetch_data("SOL-USD", start.to_pydatetime(), end.to_pydatetime())

    assert list(filtered.columns) == ["open", "high", "low", "close", "volume"]
    assert filtered.index.min() >= start
    assert filtered.index.max() <= end


def test_grid_search_selects_best_strategy(monkeypatch, backtest_metrics_factory):
    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 20)

    metrics_a = backtest_metrics_factory.build(sharpe_ratio=0.8)
    metrics_b = backtest_metrics_factory.build(sharpe_ratio=1.3)

    def fake_optimize(strategy, **kwargs):
        metrics = metrics_a if strategy == "alpha" else metrics_b
        return _result_for(
            strategy, kwargs["symbol"], kwargs["start_date"], kwargs["end_date"], metrics
        )

    monkeypatch.setattr(optimize_module, "optimize_strategy", fake_optimize)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    results = grid_search(["alpha", "beta"], "BTC-USD", start, end, metric="sharpe_ratio")

    assert results["beta"].best_metrics is metrics_b
    assert set(results.keys()) == {"alpha", "beta"}


def test_walk_forward_analysis_generates_windows(
    monkeypatch,
    seeded_ohlc_sets,
    backtest_metrics_factory,
):
    data = seeded_ohlc_sets["uptrend"]
    provider = _stub_provider(data)

    monkeypatch.setattr(optimize_module, "fetch_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(optimize_module, "get_data_provider", lambda: provider)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    train_metrics = backtest_metrics_factory.build(total_return=0.20, sharpe_ratio=1.0)

    def fake_optimize(strategy, symbol, start_date, end_date, **kwargs):
        return _result_for(strategy, symbol, start_date, end_date, train_metrics)

    counter = 0

    def fake_backtest(strategy, dataset, params, commission, slippage):
        nonlocal counter
        counter += 1
        return backtest_metrics_factory.build(
            total_return=0.05 + counter * 0.005,
            sharpe_ratio=0.7,
        )

    monkeypatch.setattr(optimize_module, "optimize_strategy", fake_optimize)
    monkeypatch.setattr(optimize_module, "run_backtest_local", fake_backtest)

    result = walk_forward_analysis(
        strategy="trend",
        symbol="BTC-USD",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 4, 1),
        window_size=30,
        step_size=20,
        test_size=15,
    )

    assert len(result.windows) > 0
    first_window = result.windows[0]
    efficiency = first_window.test_metrics.total_return / first_window.train_metrics.total_return
    avg_expected = sum(w.get_efficiency() for w in result.windows) / len(result.windows)
    assert result.avg_efficiency == pytest.approx(avg_expected)
    assert 0.0 <= result.robustness_score <= 1.0
