"""Unit tests for the position sizing orchestrator."""

from __future__ import annotations

import pytest

from bot_v2.errors import ValidationError
from bot_v2.features.position_sizing.position_sizing import calculate_position_size
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
