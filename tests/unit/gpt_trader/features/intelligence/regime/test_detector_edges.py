"""Edge tests for MarketRegimeDetector internals."""

from decimal import Decimal

import pytest

from gpt_trader.features.intelligence.regime import MarketRegimeDetector, RegimeConfig, RegimeType
from gpt_trader.features.intelligence.regime.detector import _SymbolRegimeState


def test_apply_persistence_same_regime_clears_pending() -> None:
    config = RegimeConfig(min_regime_ticks=2)
    detector = MarketRegimeDetector(config)
    state = _SymbolRegimeState(config)
    state.current_regime = RegimeType.BULL_QUIET
    state.regime_ticks = 1
    state.pending_regime = RegimeType.BEAR_QUIET
    state.pending_ticks = 1

    final_regime, age = detector._apply_persistence(state, RegimeType.BULL_QUIET)

    assert final_regime == RegimeType.BULL_QUIET
    assert age == 2
    assert state.regime_ticks == 2
    assert state.pending_regime is None
    assert state.pending_ticks == 0


def test_apply_persistence_transitions_after_min_ticks() -> None:
    config = RegimeConfig(min_regime_ticks=2)
    detector = MarketRegimeDetector(config)
    state = _SymbolRegimeState(config)
    state.current_regime = RegimeType.BULL_QUIET
    state.regime_ticks = 5
    state.pending_regime = RegimeType.BEAR_QUIET
    state.pending_ticks = 1

    final_regime, age = detector._apply_persistence(state, RegimeType.BEAR_QUIET)

    assert final_regime == RegimeType.BEAR_QUIET
    assert age == 2
    assert state.current_regime == RegimeType.BEAR_QUIET
    assert state.regime_ticks == 2
    assert state.pending_regime is None
    assert state.pending_ticks == 0


def test_estimate_transition_probability_stable_uses_history() -> None:
    config = RegimeConfig(min_regime_ticks=2)
    detector = MarketRegimeDetector(config)
    state = _SymbolRegimeState(config)
    state.current_regime = RegimeType.BULL_QUIET
    state.regime_ticks = 5
    symbol = "BTC-USD"

    matrix = detector._get_transition_matrix(symbol)
    matrix.record_transition("BULL_QUIET", "BULL_QUIET")
    matrix.record_transition("BULL_QUIET", "BEAR_QUIET")

    result = detector._estimate_transition_probability(state, RegimeType.BULL_QUIET, symbol)

    base_prob = max(0.05, 0.2 - state.regime_ticks * 0.01)
    expected = (base_prob + 0.5) / 2
    assert result == pytest.approx(expected)


def test_estimate_transition_probability_pending_uses_history_weighted() -> None:
    config = RegimeConfig(min_regime_ticks=2)
    detector = MarketRegimeDetector(config)
    state = _SymbolRegimeState(config)
    state.current_regime = RegimeType.BULL_QUIET
    state.pending_regime = RegimeType.BEAR_QUIET
    state.pending_ticks = 1
    symbol = "BTC-USD"

    matrix = detector._get_transition_matrix(symbol)
    matrix.record_transition("BULL_QUIET", "BEAR_QUIET")

    result = detector._estimate_transition_probability(state, RegimeType.BEAR_QUIET, symbol)

    progress = state.pending_ticks / config.min_regime_ticks
    expected = min(0.95, progress * 0.7 + 1.0 * 0.3)
    assert result == pytest.approx(expected)


def test_estimate_transition_probability_fallback_when_detected_diff() -> None:
    config = RegimeConfig(min_regime_ticks=2)
    detector = MarketRegimeDetector(config)
    state = _SymbolRegimeState(config)
    state.current_regime = RegimeType.BULL_QUIET

    result = detector._estimate_transition_probability(state, RegimeType.BEAR_QUIET)

    assert result == 0.3


def test_check_crisis_high_volatility_path() -> None:
    config = RegimeConfig(crisis_volatility_multiplier=3.0)
    detector = MarketRegimeDetector(config)
    state = _SymbolRegimeState(config)
    state.returns_welford.count = 21
    state.returns_welford.m2 = Decimal("21")
    state.volatility_history.add(Decimal("0.1"))

    assert detector._check_crisis(state, volatility_percentile=0.96, drawdown=0.0)


def test_check_crisis_drawdown_path() -> None:
    config = RegimeConfig(crisis_drawdown_threshold=0.1)
    detector = MarketRegimeDetector(config)
    state = _SymbolRegimeState(config)

    assert detector._check_crisis(state, volatility_percentile=0.2, drawdown=0.2)


def test_classify_normal_regime_adx_weak_raises_threshold() -> None:
    detector = MarketRegimeDetector(RegimeConfig())

    regime = detector._classify_normal_regime(
        trend_score=0.03,
        volatility_percentile=0.2,
        adx=10.0,
        squeeze_score=0.5,
    )

    assert regime == RegimeType.SIDEWAYS_QUIET


def test_classify_normal_regime_adx_strong_lowers_threshold() -> None:
    detector = MarketRegimeDetector(RegimeConfig())

    regime = detector._classify_normal_regime(
        trend_score=0.02,
        volatility_percentile=0.2,
        adx=50.0,
        squeeze_score=0.5,
    )

    assert regime == RegimeType.BULL_QUIET


def test_classify_normal_regime_high_squeeze_biases_quiet() -> None:
    detector = MarketRegimeDetector(RegimeConfig())

    regime = detector._classify_normal_regime(
        trend_score=0.0,
        volatility_percentile=0.75,
        adx=None,
        squeeze_score=0.8,
    )

    assert regime == RegimeType.SIDEWAYS_QUIET


def test_getters_missing_symbol_return_defaults() -> None:
    detector = MarketRegimeDetector(RegimeConfig())

    assert detector.get_transition_forecast("BTC-USD") == {}
    assert detector.get_indicator_values("BTC-USD") == {}
    assert detector.get_atr_percentile("BTC-USD") is None
    assert detector.get_adx("BTC-USD") is None
    assert detector.get_squeeze_score("BTC-USD") is None


def test_getters_new_state_defaults() -> None:
    detector = MarketRegimeDetector(RegimeConfig())
    detector._get_or_create_state("BTC-USD")

    assert detector.get_atr_percentile("BTC-USD") is None
    assert detector.get_adx("BTC-USD") is None
    assert detector.get_squeeze_score("BTC-USD") == 0.5
