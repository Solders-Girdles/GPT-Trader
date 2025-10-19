from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.analyze import analyze
from bot_v2.features.analyze.types import (
    AnalysisResult,
    MarketRegime,
    StrategyComparison,
    SupportResistance,
    TechnicalIndicators,
)


def _indicators() -> TechnicalIndicators:
    return TechnicalIndicators(
        sma_20=10.0,
        sma_50=10.0,
        sma_200=10.0,
        ema_12=10.0,
        ema_26=10.0,
        rsi=50.0,
        macd=0.1,
        macd_signal=0.05,
        macd_histogram=0.05,
        bollinger_upper=11.0,
        bollinger_middle=10.0,
        bollinger_lower=9.0,
        atr=0.2,
        volume_sma=100.0,
        obv=1.0,
        stochastic_k=50.0,
        stochastic_d=50.0,
    )


def _regime(trend: str = "neutral") -> MarketRegime:
    return MarketRegime(
        trend=trend,
        volatility="medium",
        momentum="neutral",
        volume_profile="neutral",
        strength=50.0,
    )


def _levels() -> SupportResistance:
    return SupportResistance(
        immediate_support=9.0,
        strong_support=8.5,
        immediate_resistance=11.0,
        strong_resistance=11.5,
        pivot_point=10.0,
    )


def _analysis(symbol: str, recommendation: str) -> AnalysisResult:
    return AnalysisResult(
        symbol=symbol,
        timestamp=datetime.utcnow(),
        current_price=100.0,
        indicators=_indicators(),
        regime=_regime("bullish" if recommendation == "strong_buy" else "bearish"),
        patterns=[],
        levels=_levels(),
        strategy_signals=[],
        recommendation=recommendation,  # type: ignore[arg-type]
        confidence=0.8,
    )


def test_analyze_portfolio_requires_symbols():
    with pytest.raises(ValueError):
        analyze.analyze_portfolio([])


def test_analyze_portfolio_aggregates_results(monkeypatch: pytest.MonkeyPatch) -> None:
    analyses = {
        "AAPL": _analysis("AAPL", "strong_buy"),
        "MSFT": _analysis("MSFT", "strong_sell"),
    }
    correlation_matrix = pd.DataFrame(
        np.eye(len(analyses)),
        index=analyses.keys(),
        columns=analyses.keys(),
    )

    monkeypatch.setattr(
        analyze, "analyze_symbol", lambda symbol, lookback_days=90: analyses[symbol]
    )
    monkeypatch.setattr(
        analyze,
        "calculate_correlations",
        lambda symbols, lookback_days: correlation_matrix,
    )
    monkeypatch.setattr(
        analyze,
        "calculate_portfolio_risk",
        lambda symbols, weights, matrix: {"beta": 1.5, "concentration_risk": max(weights.values())},
    )
    monkeypatch.setattr(
        analyze,
        "generate_rebalance_suggestions",
        lambda symbol_analyses, current_weights: [{"symbol": "AAPL", "action": "increase"}],
    )

    result = analyze.analyze_portfolio(["AAPL", "MSFT"], weights=None, lookback_days=30)

    assert set(result.symbol_analyses.keys()) == {"AAPL", "MSFT"}
    assert result.risk_metrics["beta"] == 1.5
    assert result.rebalance_suggestions == [{"symbol": "AAPL", "action": "increase"}]
    # Default weights should split evenly
    assert result.risk_metrics["concentration_risk"] == pytest.approx(0.5)


def test_analyze_portfolio_handles_analysis_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    symbols = ["GOOD", "BAD"]
    analyses = {"GOOD": _analysis("GOOD", "buy")}

    def fake_analyze_symbol(symbol: str, lookback_days: int = 90) -> AnalysisResult:
        if symbol == "BAD":
            raise RuntimeError("boom")
        return analyses[symbol]

    monkeypatch.setattr(analyze, "analyze_symbol", fake_analyze_symbol)
    monkeypatch.setattr(
        analyze,
        "calculate_correlations",
        lambda symbols, lookback_days: pd.DataFrame(),
    )
    monkeypatch.setattr(
        analyze,
        "calculate_portfolio_risk",
        lambda symbols, weights, matrix: {"beta": 1.0, "concentration_risk": max(weights.values())},
    )
    monkeypatch.setattr(
        analyze,
        "generate_rebalance_suggestions",
        lambda symbol_analyses, current_weights: [],
    )

    result = analyze.analyze_portfolio(symbols, weights={"GOOD": 1.0, "BAD": 0.0}, lookback_days=5)

    assert "GOOD" in result.symbol_analyses
    assert "BAD" not in result.symbol_analyses
    assert result.risk_metrics["beta"] == 1.0


def test_compare_strategies_ranks_best(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_prices = pd.DataFrame({"close": [100, 102, 101, 104]})
    monkeypatch.setattr(analyze, "fetch_data", lambda symbol, lookback: fake_prices)

    strategy_metrics = {
        "S1": {"return": 0.1, "sharpe": 1.5, "max_drawdown": 0.05},
        "S2": {"return": 0.05, "sharpe": 1.0, "max_drawdown": 0.02},
    }

    monkeypatch.setattr(
        analyze,
        "backtest_strategy",
        lambda strategy, data: strategy_metrics[strategy],
    )

    comparison = analyze.compare_strategies("AAPL", strategies=["S1", "S2"], lookback_days=10)

    assert isinstance(comparison, StrategyComparison)
    assert comparison.best_strategy == "S1"
    assert comparison.rankings["S1"] == 1
    assert comparison.metrics["S2"]["max_drawdown"] == 0.02
