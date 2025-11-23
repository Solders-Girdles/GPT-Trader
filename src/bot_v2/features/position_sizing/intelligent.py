"""Intelligent position sizing calculations."""

from __future__ import annotations

from typing import List

from bot_v2.errors import ValidationError, log_error
from bot_v2.features.position_sizing.confidence import confidence_adjusted_size
from bot_v2.features.position_sizing.kelly import (
    fractional_kelly,
    kelly_position_value,
    kelly_with_volatility_scaling,
)
from bot_v2.features.position_sizing.regime import regime_adjusted_size
from bot_v2.features.position_sizing.types import (
    ConfidenceAdjustment,
    PositionSizeRequest,
    PositionSizeResponse,
    RegimeMultipliers,
    SizingMethod,
)
from bot_v2.utilities.logging_patterns import get_logger

from .utils import estimate_position_risk, optional_float
from .validation import extract_kelly_params, validate_kelly_safety

logger = get_logger(__name__, component="position_sizing")


def calculate_intelligent_size(request: PositionSizeRequest) -> PositionSizeResponse:
    notes: List[str] = []
    warnings: List[str] = []
    kelly_inputs = extract_kelly_params(request)

    try:
        if kelly_inputs is not None:
            win_rate, avg_win, avg_loss = kelly_inputs
            validate_kelly_safety(win_rate, avg_win, avg_loss)

            if request.recent_prices and len(request.recent_prices) >= 20:
                base_size, vol_metrics = kelly_with_volatility_scaling(
                    win_rate,
                    avg_win,
                    avg_loss,
                    request.recent_prices,
                    fraction=request.risk_params.kelly_fraction,
                )
                notes.append(
                    f"Volatility-aware Kelly sizing: {base_size:.4f} "
                    f"(regime={vol_metrics.get('regime', 'unknown')}, "
                    f"scaling={vol_metrics.get('scaling_factor', 1.0):.2f}x)"
                )
            else:
                base_size = fractional_kelly(
                    win_rate,
                    avg_win,
                    avg_loss,
                    request.risk_params.kelly_fraction,
                )
                notes.append(f"Base Kelly sizing: {base_size:.4f}")

            if base_size > request.risk_params.max_position_size:
                warnings.append(
                    f"Kelly fraction {base_size:.4f} exceeds max position size, "
                    f"capping at {request.risk_params.max_position_size:.4f}"
                )
                base_size = request.risk_params.max_position_size
        else:
            base_size = request.risk_params.max_position_size * 0.5
            notes.append(f"No trade statistics, using conservative fixed size: {base_size:.4f}")
            warnings.append("No historical trade data available for Kelly calculation")

    except Exception as exc:
        error = ValidationError(f"Kelly calculation failed: {exc}", field="kelly_inputs")
        log_error(error)
        warnings.append(str(error))
        base_size = request.risk_params.max_position_size * 0.1

    confidence_adjustment = 1.0
    if request.confidence is not None:
        try:
            if request.confidence >= request.risk_params.confidence_threshold:
                adj_params = ConfidenceAdjustment(confidence=request.confidence)
                adjusted_size, conf_explanation = confidence_adjusted_size(
                    base_size,
                    request.confidence,
                    adj_params,
                )
                confidence_adjustment = adjusted_size / base_size if base_size > 0 else 1.0
                base_size = adjusted_size
                notes.append(f"Confidence adjustment: {conf_explanation}")
            else:
                confidence_adjustment = 0.0
                base_size = 0.0
                warnings.append(
                    f"Confidence {request.confidence:.2f} below threshold "
                    f"{request.risk_params.confidence_threshold:.2f}"
                )
        except ValidationError as exc:
            log_error(exc)
            warnings.append(f"Invalid confidence score: {exc.message}")

    regime_adjustment = 1.0
    if request.market_regime:
        try:
            multipliers = RegimeMultipliers()
            adjusted_size, regime_explanation = regime_adjusted_size(
                base_size,
                request.market_regime,
                multipliers,
            )
            regime_adjustment = adjusted_size / base_size if base_size > 0 else 1.0
            base_size = adjusted_size
            notes.append(f"Regime adjustment: {regime_explanation}")
        except Exception as exc:
            error = ValidationError(f"Regime adjustment failed: {exc}", field="market_regime")
            log_error(error)
            warnings.append(f"Regime adjustment failed: {exc}")

    base_size *= request.strategy_multiplier
    if request.strategy_multiplier != 1.0:
        notes.append(f"Strategy multiplier: {request.strategy_multiplier:.2f}x")

    if base_size < 0:
        raise ValidationError("Position size cannot be negative", field="base_size")

    if base_size > request.risk_params.max_position_size:
        warnings.append("Position size capped at max allowed")
        base_size = request.risk_params.max_position_size

    try:
        position_value, share_count = kelly_position_value(
            request.portfolio_value,
            base_size,
            request.current_price,
            request.risk_params,
        )
    except Exception as exc:
        error = ValidationError(
            f"Position value calculation failed: {exc}", field="position_calculation"
        )
        log_error(error)
        raise error

    position_size_pct = position_value / request.portfolio_value
    risk_pct = estimate_position_risk(request, position_size_pct)

    return PositionSizeResponse(
        symbol=request.symbol,
        recommended_shares=share_count,
        recommended_value=position_value,
        position_size_pct=position_size_pct,
        risk_pct=risk_pct,
        method_used=SizingMethod.INTELLIGENT,
        kelly_fraction=base_size if kelly_inputs is not None else None,
        confidence_adjustment=confidence_adjustment if request.confidence else None,
        regime_adjustment=regime_adjustment if request.market_regime else None,
        max_loss_estimate=position_value * abs(optional_float(request.avg_loss, 0.05)),
        expected_return=position_value
        * optional_float(request.avg_win, 0.03)
        * optional_float(request.win_rate, 0.5),
        calculation_notes=notes,
        warnings=warnings,
    )


__all__ = ["calculate_intelligent_size"]
