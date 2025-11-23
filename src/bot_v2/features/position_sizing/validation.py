"""Request validation helpers for position sizing calculations."""

from __future__ import annotations

from typing import Tuple

from bot_v2.errors import ValidationError
from bot_v2.features.position_sizing.kelly import validate_kelly_inputs
from bot_v2.features.position_sizing.types import PositionSizeRequest
from bot_v2.validation import PercentageValidator, PositiveNumberValidator


def validate_position_request(request: PositionSizeRequest, field_name: str) -> PositionSizeRequest:
    """Validate position sizing request with comprehensive checks."""

    if not isinstance(request, PositionSizeRequest):
        raise ValidationError(f"{field_name} must be PositionSizeRequest", field=field_name)

    if not request.symbol:
        raise ValidationError("Symbol is required", field="symbol")

    PositiveNumberValidator()(request.current_price, "current_price")
    PositiveNumberValidator()(request.portfolio_value, "portfolio_value")
    PositiveNumberValidator()(request.strategy_multiplier, "strategy_multiplier")

    kelly_inputs = extract_kelly_params(request)
    if kelly_inputs is not None:
        win_rate, avg_win, avg_loss = kelly_inputs
        try:
            validate_kelly_inputs(win_rate, avg_win, avg_loss)
        except Exception as exc:  # pragma: no cover - delegated validation
            raise ValidationError(f"Kelly validation failed: {exc}", field="kelly_inputs")

    if request.confidence is not None:
        PercentageValidator(as_decimal=True)(request.confidence, "confidence")

    return request


def validate_kelly_safety(win_rate: float, avg_win: float, avg_loss: float) -> None:
    """Additional guardrails for Kelly based sizing."""

    if abs(avg_loss) < 1e-10:
        raise ValidationError(
            "Average loss too close to zero - division by zero risk", field="avg_loss"
        )

    if win_rate < 0.01 or win_rate > 0.99:
        raise ValidationError(
            f"Win rate {win_rate} outside reasonable bounds [0.01, 0.99]",
            field="win_rate",
        )

    expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
    if expected_value <= 0:
        raise ValidationError(
            f"Strategy has negative expected value: {expected_value:.6f}",
            field="expected_value",
        )


def extract_kelly_params(request: PositionSizeRequest) -> Tuple[float, float, float] | None:
    if request.win_rate is None or request.avg_win is None or request.avg_loss is None:
        return None
    return (
        float(request.win_rate),
        float(request.avg_win),
        float(request.avg_loss),
    )


__all__ = ["validate_position_request", "validate_kelly_safety", "extract_kelly_params"]
