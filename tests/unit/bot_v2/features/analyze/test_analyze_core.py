from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.analyze import analyze
from bot_v2.features.analyze.types import (
    MarketRegime,
    PricePattern,
    StrategySignals,
    SupportResistance,
    TechnicalIndicators,
)


def _basic_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14],
            "high": [11, 12, 13, 14, 15],
            "low": [9, 10, 11, 12, 13],
            "close": [10, 11, 12, 13, 14],
            "volume": [100, 120, 110, 150, 180],
        }
    )


def test_calculate_indicators_handles_short_series():
    data = _basic_data()

    indicators = analyze.calculate_indicators(data)

    assert isinstance(indicators, TechnicalIndicators)
    assert np.isnan(indicators.sma_20)
    assert np.isnan(indicators.sma_50)
    assert np.isnan(indicators.sma_200)
    assert indicators.rsi >= 0
    assert indicators.macd_histogram is not None
    assert indicators.stochastic_k is not None


def _regime_indicators(rsi: float, volume_sma: float) -> TechnicalIndicators:
    base = analyze.calculate_indicators(_basic_data())
    return TechnicalIndicators(
        sma_20=base.sma_20,
        sma_50=base.sma_50,
        sma_200=base.sma_200,
        ema_12=base.ema_12,
        ema_26=base.ema_26,
        rsi=rsi,
        macd=base.macd,
        macd_signal=base.macd_signal,
        macd_histogram=base.macd_histogram,
        bollinger_upper=base.bollinger_upper,
        bollinger_middle=base.bollinger_middle,
        bollinger_lower=base.bollinger_lower,
        atr=base.atr,
        volume_sma=volume_sma,
        obv=base.obv,
        stochastic_k=base.stochastic_k,
        stochastic_d=base.stochastic_d,
    )


def test_determine_regime_classifies_volume_profiles():
    data = _basic_data()
    high_volume_up = data.copy()
    high_volume_up["volume"] = [200, 220, 210, 260, 500]
    high_volume_up["close"] = [10, 11, 12, 13, 14]

    bullish_indicators = _regime_indicators(rsi=80, volume_sma=150)
    regime = analyze.determine_regime(high_volume_up, bullish_indicators)

    assert regime.momentum == "strong_up"
    assert regime.volume_profile == "accumulation"
    assert regime.strength >= 25

    high_volume_down = high_volume_up.copy()
    high_volume_down["close"] = [10, 11, 12, 13, 12]  # price falls on last bar

    bearish_indicators = _regime_indicators(rsi=20, volume_sma=150)
    regime_bear = analyze.determine_regime(high_volume_down, bearish_indicators)
    assert regime_bear.momentum == "strong_down"
    assert regime_bear.volume_profile == "distribution"


def test_determine_regime_defaults_on_unknown_values():
    data = _basic_data()
    indicators = _regime_indicators(rsi=50, volume_sma=200)

    with patch.object(analyze, "calculate_volatility", return_value="extreme"):
        regime = analyze.determine_regime(data, indicators)

    assert regime.volatility == "medium"


@pytest.mark.parametrize(
    "rsi,trend,signals,patterns,expected",
    [
        (20, "bullish", [StrategySignals("s1", 2, 0.9, "test")], [], "buy"),
        (75, "bearish", [], [], "sell"),
        (55, "neutral", [], [], "buy"),
    ],
)
def test_generate_recommendation_scores_paths(
    rsi: float,
    trend: str,
    signals: list[StrategySignals],
    patterns: list[PricePattern],
    expected: str,
) -> None:
    indicators = _regime_indicators(rsi, 100)
    regime = MarketRegime(
        trend=trend, volatility="medium", momentum="neutral", volume_profile="neutral", strength=50
    )

    recommendation, confidence = analyze.generate_recommendation(
        indicators, regime, patterns, signals
    )

    assert recommendation == expected
    assert 0 <= confidence <= 0.9


def test_generate_recommendation_pattern_bias():
    base = analyze.calculate_indicators(_basic_data())
    indicators = _regime_indicators(base.rsi, 100)
    regime = MarketRegime(
        trend="neutral",
        volatility="medium",
        momentum="neutral",
        volume_profile="neutral",
        strength=50,
    )
    bull = PricePattern("Bull Flag", 0.7, 120, 90, "bullish")
    bear = PricePattern("Bear Flag", 0.7, 80, 110, "bearish")

    recommendation, _ = analyze.generate_recommendation(indicators, regime, [bull, bear, bull], [])

    assert recommendation in {"buy", "strong_buy", "hold"}


def test_calculate_portfolio_risk_handles_empty_matrix():
    result = analyze.calculate_portfolio_risk(["AAPL"], {"AAPL": 1.0}, pd.DataFrame())

    assert result["correlation_risk"] == 0.0
    assert result["concentration_risk"] == 1.0


def test_calculate_portfolio_risk_averages_upper_triangle():
    matrix = pd.DataFrame(
        [[1, 0.5, 0.2], [0.5, 1, -0.1], [0.2, -0.1, 1]], columns=list("ABC"), index=list("ABC")
    )
    weights = {"A": 0.4, "B": 0.3, "C": 0.3}

    result = analyze.calculate_portfolio_risk(list(weights.keys()), weights, matrix)

    expected_avg = np.mean([0.5, 0.2, -0.1])
    assert result["correlation_risk"] == pytest.approx(abs(expected_avg))


def test_generate_rebalance_suggestions_reacts_to_recommendations():
    analysis = analyze.AnalysisResult(
        symbol="AAPL",
        timestamp=datetime.now(),
        current_price=150.0,
        indicators=analyze.calculate_indicators(_basic_data()),
        regime=MarketRegime("bullish", "low", "weak_up", "neutral", 70),
        patterns=[],
        levels=SupportResistance(100, 90, 160, 170, 120),
        strategy_signals=[],
        recommendation="strong_buy",
        confidence=0.8,
    )

    suggestions = analyze.generate_rebalance_suggestions({"AAPL": analysis}, {"AAPL": 0.1})

    assert suggestions
    assert suggestions[0]["action"] == "increase"

    analysis.recommendation = "strong_sell"
    suggestions = analyze.generate_rebalance_suggestions({"AAPL": analysis}, {"AAPL": 0.2})

    assert suggestions
    assert suggestions[0]["action"] == "decrease"


def test_analyze_symbol_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _basic_data()

    provider_mock = MagicMock()
    provider_mock.get_historical_data.return_value = data
    monkeypatch.setattr(analyze, "get_data_provider", lambda: provider_mock)
    monkeypatch.setattr(
        analyze, "detect_patterns", lambda _: [PricePattern("Bull Flag", 0.6, 120, 90, "")]
    )
    monkeypatch.setattr(
        analyze,
        "analyze_with_strategies",
        lambda _: [StrategySignals("demo", 1, 0.7, "Uptrend")],
    )
    result = analyze.analyze_symbol("AAPL", lookback_days=5)

    assert result.symbol == "AAPL"
    assert result.current_price == data["close"].iloc[-1]
    assert result.patterns
    assert result.strategy_signals


def test_analyze_symbol_raises_on_empty_data(monkeypatch: pytest.MonkeyPatch) -> None:
    provider_mock = MagicMock()
    provider_mock.get_historical_data.return_value = pd.DataFrame(columns=["close"])
    monkeypatch.setattr(analyze, "get_data_provider", lambda: provider_mock)

    with pytest.raises(ValueError):
        analyze.analyze_symbol("AAPL")
