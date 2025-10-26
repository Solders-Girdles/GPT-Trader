"""Primary entry points for position sizing calculations."""

from __future__ import annotations

from typing import List

from bot_v2.errors import ValidationError, log_error
from bot_v2.features.position_sizing.types import (
    PositionSizeRequest,
    PositionSizeResponse,
    PositionSizingResult,
    SizingMethod,
)
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.validation import validate_inputs

from .intelligent import calculate_intelligent_size
from .strategies import (
    calculate_confidence_size,
    calculate_fixed_size,
    calculate_fractional_kelly_size,
    calculate_kelly_size,
    calculate_regime_size,
)
from .utils import create_error_response
from .validation import validate_position_request

logger = get_logger(__name__, component="position_sizing")


@validate_inputs(request=validate_position_request)
def calculate_position_size(request: PositionSizeRequest) -> PositionSizeResponse:
    try:
        default_config = {
            "max_position_size": 0.25,
            "min_position_size": 0.01,
            "default_leverage": 2.0,
            "volatility_adjustment": True,
        }

        if hasattr(request.risk_params, "__dict__"):
            for key, value in default_config.items():
                if hasattr(request.risk_params, key) and getattr(request.risk_params, key) is None:
                    setattr(request.risk_params, key, value)

        logger.info(
            "Calculating position size for %s using %s",
            request.symbol,
            request.method.value,
        )

    except Exception as exc:
        error = ValidationError(f"Configuration error: {exc}", field="config")
        log_error(error)
        return create_error_response(request, [str(error)])

    if request.method == SizingMethod.INTELLIGENT:
        return calculate_intelligent_size(request)
    if request.method == SizingMethod.KELLY:
        return calculate_kelly_size(request)
    if request.method == SizingMethod.FRACTIONAL_KELLY:
        return calculate_fractional_kelly_size(request)
    if request.method == SizingMethod.CONFIDENCE_ADJUSTED:
        return calculate_confidence_size(request)
    if request.method == SizingMethod.REGIME_ADJUSTED:
        return calculate_regime_size(request)
    if request.method == SizingMethod.FIXED:
        return calculate_fixed_size(request)
    return create_error_response(request, [f"Unknown sizing method: {request.method}"])


def calculate_portfolio_allocation(
    requests: List[PositionSizeRequest],
) -> PositionSizingResult:
    if not requests:
        return PositionSizingResult(
            primary=create_error_response(
                PositionSizeRequest("", 0, 0, ""),
                ["No requests provided"],
            )
        )

    individual_responses = [calculate_position_size(request) for request in requests]

    total_position_pct = sum(resp.position_size_pct for resp in individual_responses)
    max_portfolio_allocation = 0.8
    if total_position_pct > max_portfolio_allocation:
        scale_factor = max_portfolio_allocation / total_position_pct
        scaled_responses: List[PositionSizeResponse] = []
        for resp in individual_responses:
            if resp.recommended_value == 0:
                scaled_responses.append(resp)
                continue

            scaled_value = resp.recommended_value * scale_factor
            per_share_value = resp.recommended_value / max(1, resp.recommended_shares)
            scaled_shares = int(scaled_value / per_share_value) if per_share_value else 0

            scaled_responses.append(
                PositionSizeResponse(
                    symbol=resp.symbol,
                    recommended_shares=scaled_shares,
                    recommended_value=scaled_value,
                    position_size_pct=resp.position_size_pct * scale_factor,
                    risk_pct=resp.risk_pct * scale_factor,
                    method_used=resp.method_used,
                    kelly_fraction=resp.kelly_fraction,
                    confidence_adjustment=resp.confidence_adjustment,
                    regime_adjustment=resp.regime_adjustment,
                    max_loss_estimate=resp.max_loss_estimate * scale_factor,
                    expected_return=resp.expected_return * scale_factor,
                    calculation_notes=resp.calculation_notes
                    + [f"Portfolio scaled by {scale_factor:.3f}"],
                    warnings=resp.warnings,
                )
            )

        individual_responses = scaled_responses

    portfolio_impact = {
        "total_allocation_pct": sum(resp.position_size_pct for resp in individual_responses),
        "total_risk_pct": sum(resp.risk_pct for resp in individual_responses),
        "expected_portfolio_return": sum(resp.expected_return for resp in individual_responses),
        "max_portfolio_loss": sum(resp.max_loss_estimate for resp in individual_responses),
        "num_positions": len(
            [resp for resp in individual_responses if resp.recommended_shares > 0]
        ),
    }

    primary = max(individual_responses, key=lambda x: x.recommended_value)

    return PositionSizingResult(
        primary=primary,
        alternatives=individual_responses,
        portfolio_impact=portfolio_impact,
    )


__all__ = ["calculate_position_size", "calculate_portfolio_allocation"]
