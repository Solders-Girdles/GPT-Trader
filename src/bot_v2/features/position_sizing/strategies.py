"""Alternative sizing strategies built on top of core utilities."""

from __future__ import annotations

from bot_v2.errors import ValidationError, log_error
from bot_v2.features.position_sizing.confidence import confidence_adjusted_size
from bot_v2.features.position_sizing.kelly import (
    fractional_kelly,
    kelly_criterion,
    kelly_position_value,
)
from bot_v2.features.position_sizing.regime import regime_adjusted_size
from bot_v2.features.position_sizing.types import (
    ConfidenceAdjustment,
    PositionSizeRequest,
    PositionSizeResponse,
    RegimeMultipliers,
    SizingMethod,
)

from .utils import create_error_response, estimate_position_risk
from .validation import extract_kelly_params


def calculate_kelly_size(request: PositionSizeRequest) -> PositionSizeResponse:
    kelly_inputs = extract_kelly_params(request)
    if kelly_inputs is None:
        return create_error_response(
            request, ["Kelly sizing requires win_rate, avg_win, and avg_loss"]
        )

    win_rate, avg_win, avg_loss = kelly_inputs
    kelly_size = kelly_criterion(win_rate, avg_win, avg_loss)
    position_value, share_count = kelly_position_value(
        request.portfolio_value,
        kelly_size,
        request.current_price,
        request.risk_params,
    )

    position_size_pct = position_value / request.portfolio_value
    risk_pct = estimate_position_risk(request, position_size_pct)

    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.KELLY,
        kelly_fraction=kelly_size,
        max_loss_estimate=position_value * abs(avg_loss),
        expected_return=position_value * avg_win * win_rate,
        calculation_notes=[f"Full Kelly Criterion: {kelly_size:.4f}"],
    )


def calculate_fractional_kelly_size(request: PositionSizeRequest) -> PositionSizeResponse:
    kelly_inputs = extract_kelly_params(request)
    if kelly_inputs is None:
        return create_error_response(
            request, ["Fractional Kelly sizing requires win_rate, avg_win, and avg_loss"]
        )

    win_rate, avg_win, avg_loss = kelly_inputs
    kelly_size = fractional_kelly(
        win_rate,
        avg_win,
        avg_loss,
        request.risk_params.kelly_fraction,
    )

    position_value, share_count = kelly_position_value(
        request.portfolio_value,
        kelly_size,
        request.current_price,
        request.risk_params,
    )

    position_size_pct = position_value / request.portfolio_value
    risk_pct = estimate_position_risk(request, position_size_pct)

    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.FRACTIONAL_KELLY,
        kelly_fraction=kelly_size,
        max_loss_estimate=position_value * abs(avg_loss),
        expected_return=position_value * avg_win * win_rate,
        calculation_notes=[
            f"Fractional Kelly ({request.risk_params.kelly_fraction:.2f}): {kelly_size:.4f}"
        ],
    )


def calculate_confidence_size(request: PositionSizeRequest) -> PositionSizeResponse:
    if request.confidence is None:
        return create_error_response(
            request, ["Confidence sizing requires confidence score"]
        )

    base_size = request.risk_params.max_position_size * 0.5
    adj_params = ConfidenceAdjustment(confidence=request.confidence)
    adjusted_size, explanation = confidence_adjusted_size(
        base_size,
        request.confidence,
        adj_params,
    )

    position_value, share_count = kelly_position_value(
        request.portfolio_value,
        adjusted_size,
        request.current_price,
        request.risk_params,
    )

    position_size_pct = position_value / request.portfolio_value
    risk_pct = estimate_position_risk(request, position_size_pct)

    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.CONFIDENCE_ADJUSTED,
        confidence_adjustment=adjusted_size / base_size,
        max_loss_estimate=position_value * 0.05,
        expected_return=position_value * 0.03 * request.confidence,
        calculation_notes=[explanation],
    )


def calculate_regime_size(request: PositionSizeRequest) -> PositionSizeResponse:
    if not request.market_regime:
        return create_error_response(request, ["Regime sizing requires market_regime"])

    base_size = request.risk_params.max_position_size * 0.5
    multipliers = RegimeMultipliers()
    adjusted_size, explanation = regime_adjusted_size(
        base_size,
        request.market_regime,
        multipliers,
    )

    position_value, share_count = kelly_position_value(
        request.portfolio_value,
        adjusted_size,
        request.current_price,
        request.risk_params,
    )

    position_size_pct = position_value / request.portfolio_value
    risk_pct = estimate_position_risk(request, position_size_pct)

    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.REGIME_ADJUSTED,
        regime_adjustment=adjusted_size / base_size,
        max_loss_estimate=position_value * 0.05,
        expected_return=position_value * 0.03,
        calculation_notes=[explanation],
    )


def calculate_fixed_size(request: PositionSizeRequest) -> PositionSizeResponse:
    fixed_size = request.risk_params.max_position_size * 0.3
    position_value, share_count = kelly_position_value(
        request.portfolio_value,
        fixed_size,
        request.current_price,
        request.risk_params,
    )

    position_size_pct = position_value / request.portfolio_value
    risk_pct = estimate_position_risk(request, position_size_pct)

    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.FIXED,
        max_loss_estimate=position_value * 0.05,
        expected_return=position_value * 0.03,
        calculation_notes=[f"Fixed sizing: {fixed_size:.4f}"],
    )


__all__ = [
    "calculate_kelly_size",
    "calculate_fractional_kelly_size",
    "calculate_confidence_size",
    "calculate_regime_size",
    "calculate_fixed_size",
]
