"""Unit tests for the position sizing orchestrator."""

from __future__ import annotations

import pytest

from bot_v2.errors import ValidationError
from bot_v2.features.position_sizing.position_sizing import calculate_position_size
from bot_v2.features.position_sizing.regime import (
    RegimeMultipliers,
    dynamic_regime_multipliers,
    portfolio_regime_allocation,
    regime_adjusted_size,
    regime_correlation_adjustment,
    regime_momentum_factor,
    regime_transition_adjustment,
    regime_volatility_scaling,
    safe_regime_calculation,
    validate_regime_inputs,
)
from bot_v2.features.position_sizing.types import PositionSizeRequest, RiskParameters, SizingMethod


def _base_request(**overrides) -> PositionSizeRequest:
    params = RiskParameters(
        max_position_size=0.3,
        min_position_size=0.0,
        kelly_fraction=0.5,
        confidence_threshold=0.6,
    )
    base = PositionSizeRequest(
        symbol="AAPL",
        current_price=150.0,
        portfolio_value=100_000.0,
        strategy_name="momentum",
        method=SizingMethod.INTELLIGENT,
        win_rate=0.58,
        avg_win=0.07,
        avg_loss=-0.035,
        confidence=0.72,
        market_regime="bull_quiet",
        volatility=0.02,
        risk_params=params,
        strategy_multiplier=1.0,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_calculate_position_size_intelligent_happy_path() -> None:
    request = _base_request()

    response = calculate_position_size(request)

    assert response.method_used is SizingMethod.INTELLIGENT
    assert response.recommended_value > 0
    assert response.position_size_pct <= request.risk_params.max_position_size + 1e-9
    assert response.confidence_adjustment is not None
    assert response.regime_adjustment is not None
    assert response.kelly_fraction is not None


def test_calculate_position_size_intelligent_low_confidence_zero_position() -> None:
    request = _base_request(confidence=0.4)

    response = calculate_position_size(request)

    assert response.recommended_shares == 0
    assert response.position_size_pct == 0
    assert response.confidence_adjustment == 0.0
    assert any("below threshold" in warning for warning in response.warnings)


def test_calculate_position_size_kelly_requires_statistics() -> None:
    request = _base_request(method=SizingMethod.KELLY, win_rate=None, avg_win=None, avg_loss=None)

    response = calculate_position_size(request)

    assert response.method_used is SizingMethod.KELLY
    assert response.recommended_value == 0
    assert response.warnings == ["Kelly sizing requires win_rate, avg_win, and avg_loss"]


def test_calculate_position_size_kelly_happy_path() -> None:
    request = _base_request(method=SizingMethod.KELLY)

    response = calculate_position_size(request)

    assert response.method_used is SizingMethod.KELLY
    assert response.kelly_fraction is not None and response.kelly_fraction > 0
    assert response.recommended_value > 0


def test_calculate_position_size_requires_symbol() -> None:
    bad_request = _base_request(symbol="")

    with pytest.raises(ValidationError):
        calculate_position_size(bad_request)


# Additional tests for regime sizing utilities


@pytest.mark.parametrize(
    "base_size,market_regime,expected_size,expected_explanation_contains",
    [
        (0.1, "bull_quiet", 0.15, "bull_quiet"),  # Bull quiet multiplier 1.5
        (0.1, "bear_quiet", 0.06, "bear_quiet"),  # Bear quiet multiplier 0.6
        (0.1, "crisis", 0.02, "crisis"),  # Crisis multiplier 0.2
        (0.1, "", 0.1, "No regime data"),  # Empty regime
        (0.5, "bull_quiet", 0.5, "bull_quiet"),  # Size clamping to 1.0 max
        (0.0, "bull_quiet", 0.0, "bull_quiet"),  # Zero base size
    ],
)
def test_regime_adjusted_size_various_scenarios(
    base_size, market_regime, expected_size, expected_explanation_contains
):
    multipliers = RegimeMultipliers()

    adjusted_size, explanation = regime_adjusted_size(base_size, market_regime, multipliers)

    assert abs(adjusted_size - expected_size) < 1e-9
    assert expected_explanation_contains in explanation


def test_regime_adjusted_size_validation_errors():
    multipliers = RegimeMultipliers()

    # Invalid base_size
    with pytest.raises(ValidationError):
        regime_adjusted_size(-0.1, "bull_quiet", multipliers)

    # Invalid regime
    with pytest.raises(ValidationError):
        regime_adjusted_size(0.1, "invalid_regime", multipliers)

    # No multipliers
    with pytest.raises(ValidationError):
        regime_adjusted_size(0.1, "bull_quiet", None)


def test_dynamic_regime_multipliers():
    regime_history = [
        ("bull_quiet", 0.05),
        ("bull_quiet", 0.03),
        ("bull_quiet", 0.07),
        ("bear_quiet", -0.02),
        ("bear_quiet", -0.01),
    ]

    multipliers = dynamic_regime_multipliers(regime_history)

    # Bull quiet should have positive adjustment due to good performance
    assert multipliers.bull_quiet > 1.5  # Base is 1.5

    # Bear quiet should have negative adjustment due to poor performance
    assert multipliers.bear_quiet < 0.6  # Base is 0.6


def test_dynamic_regime_multipliers_empty_history():
    multipliers = dynamic_regime_multipliers([])

    # Should return default multipliers
    assert multipliers.bull_quiet == 1.5
    assert multipliers.bear_quiet == 0.6


@pytest.mark.parametrize(
    "current_regime,previous_regime,transition_confidence,base_multiplier,expected",
    [
        ("bull_quiet", "bull_quiet", 0.8, 1.5, 1.5),  # No transition
        ("bear_quiet", "bull_quiet", 0.9, 0.6, 0.6),  # High confidence transition
        ("bear_quiet", "bull_quiet", 0.6, 0.6, 0.45),  # Low confidence transition
    ],
)
def test_regime_transition_adjustment(
    current_regime, previous_regime, transition_confidence, base_multiplier, expected
):
    result = regime_transition_adjustment(
        current_regime, previous_regime, transition_confidence, base_multiplier
    )
    assert abs(result - expected) < 1e-9


@pytest.mark.parametrize(
    "regime_duration_days,regime,expected_range",
    [
        (10, "bull_quiet", (1.0, 1.1)),  # Bull regimes get momentum bonus
        (25, "bull_quiet", (1.1, 1.2)),  # Longer duration increases bonus
        (20, "bear_quiet", (0.95, 1.0)),  # Bear regimes get diminishing returns
        (30, "bear_quiet", (0.8, 0.9)),  # Very long bear reduces further
        (5, "crisis", (0.8, 0.8)),  # Crisis always 0.8
        (40, "sideways_quiet", (1.05, 1.1)),  # Sideways gets slight bonus
    ],
)
def test_regime_momentum_factor(regime_duration_days, regime, expected_range):
    result = regime_momentum_factor(regime_duration_days, regime)
    assert expected_range[0] <= result <= expected_range[1]


def test_portfolio_regime_allocation():
    current_regimes = {
        "BTC-USD": "bull_quiet",
        "ETH-USD": "bear_quiet",
        "ADA-USD": "sideways_quiet",
    }
    regime_confidences = {
        "BTC-USD": 0.9,
        "ETH-USD": 0.8,
        "ADA-USD": 0.7,
    }
    total_risk_budget = 0.5

    allocations = portfolio_regime_allocation(
        current_regimes, regime_confidences, total_risk_budget
    )

    assert len(allocations) == 3
    assert sum(allocations.values()) <= total_risk_budget + 1e-9
    assert all(v >= 0 for v in allocations.values())


def test_portfolio_regime_allocation_empty():
    allocations = portfolio_regime_allocation({}, {}, 0.5)
    assert allocations == {}


def test_regime_correlation_adjustment():
    symbol_regimes = {
        "BTC-USD": "bull_quiet",
        "ETH-USD": "bull_quiet",
        "ADA-USD": "bear_quiet",
    }

    adjustments = regime_correlation_adjustment(symbol_regimes)

    # BTC and ETH in same regime should have penalty
    assert adjustments["BTC-USD"] < 1.0
    assert adjustments["ETH-USD"] < 1.0
    # ADA in different regime should have no penalty
    assert adjustments["ADA-USD"] == 1.0


@pytest.mark.parametrize(
    "regime,realized_volatility,expected_volatility,expected_range",
    [
        ("crisis", 0.05, 0.03, (0.0, 0.5)),  # Crisis: very sensitive
        ("bull_volatile", 0.04, 0.03, (0.9, 1.1)),  # Volatile: less sensitive
        ("bull_quiet", 0.04, 0.03, (0.8, 1.0)),  # Quiet: more sensitive
        ("default", 0.04, 0.03, (0.85, 0.95)),  # Default case
    ],
)
def test_regime_volatility_scaling(
    regime, realized_volatility, expected_volatility, expected_range
):
    result = regime_volatility_scaling(regime, realized_volatility, expected_volatility)
    assert expected_range[0] <= result <= expected_range[1]


def test_regime_volatility_scaling_zero_expected():
    result = regime_volatility_scaling("bull_quiet", 0.05, 0.0)
    assert result == 1.0


def test_validate_regime_inputs():
    multipliers = RegimeMultipliers()

    # Valid inputs
    errors = validate_regime_inputs("bull_quiet", multipliers)
    assert errors == []

    # Invalid regime
    errors = validate_regime_inputs("invalid_regime", multipliers)
    assert len(errors) > 0

    # Invalid multiplier
    multipliers.bull_quiet = -1.0
    errors = validate_regime_inputs("bull_quiet", multipliers)
    assert len(errors) > 0


@pytest.mark.parametrize(
    "regime,base_multiplier,confidence,expected_range",
    [
        ("bull_quiet", 1.5, 1.0, (1.4, 1.6)),  # Full confidence
        ("bull_quiet", 1.5, 0.5, (1.25, 1.35)),  # Half confidence
        ("invalid_regime", 1.5, 1.0, (0.95, 1.05)),  # Invalid regime defaults to 1.0
        ("bull_quiet", 10.0, 1.0, (2.9, 3.1)),  # Clamped high multiplier
        ("bull_quiet", 0.0, 1.0, (0.1, 0.15)),  # Clamped low multiplier
    ],
)
def test_safe_regime_calculation(regime, base_multiplier, confidence, expected_range):
    result = safe_regime_calculation(regime, base_multiplier, confidence)
    assert expected_range[0] <= result <= expected_range[1]
