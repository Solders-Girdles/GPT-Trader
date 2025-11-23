"""Portfolio conversions and risk metrics for Kelly sizing."""

from __future__ import annotations

from bot_v2.errors import RiskLimitExceeded, ValidationError, log_error
from bot_v2.features.position_sizing.types import RiskParameters
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.validation import PositiveNumberValidator, RangeValidator

logger = get_logger(__name__, component="position_sizing")


def kelly_position_value(
    portfolio_value: float,
    kelly_fraction: float,
    price_per_share: float,
    risk_params: RiskParameters,
) -> tuple[float, int]:
    """Convert a Kelly fraction into position value and share count."""
    try:
        PositiveNumberValidator()(portfolio_value, "portfolio_value")
        PositiveNumberValidator()(price_per_share, "price_per_share")
        RangeValidator(min_value=0.0, max_value=1.0)(kelly_fraction, "kelly_fraction")

        if not risk_params:
            raise ValidationError("Risk parameters required", field="risk_params")

        raw_position_value = portfolio_value * kelly_fraction
        max_position_value = portfolio_value * risk_params.max_position_size
        min_position_value = portfolio_value * risk_params.min_position_size

        if max_position_value <= 0:
            raise ValidationError("Max position value must be positive", field="max_position_size")
        if min_position_value < 0:
            raise ValidationError(
                "Min position value cannot be negative", field="min_position_size"
            )
        if min_position_value > max_position_value:
            raise ValidationError(
                "Min position size exceeds max position size", field="position_limits"
            )

        if raw_position_value > max_position_value:
            logger.warning(
                "Kelly position $%.2f exceeds max $%.2f", raw_position_value, max_position_value
            )

        position_value = max(min_position_value, min(max_position_value, raw_position_value))

        if position_value < price_per_share:
            logger.warning(
                "Position value $%.2f less than share price $%.2f",
                position_value,
                price_per_share,
            )
            return 0.0, 0

        share_count = int(position_value / price_per_share)
        actual_position_value = share_count * price_per_share

        if actual_position_value > portfolio_value:
            raise RiskLimitExceeded(
                "Position value exceeds portfolio value",
                limit_type="portfolio_value",
                limit_value=portfolio_value,
                current_value=actual_position_value,
            )

        return actual_position_value, share_count

    except (ValidationError, RiskLimitExceeded):
        raise
    except Exception as exc:  # pragma: no cover - defensive
        error = ValidationError(
            f"Position value calculation failed: {exc}", field="position_calculation"
        )
        log_error(error)
        raise error


def kelly_risk_metrics(kelly_fraction: float, avg_loss: float, portfolio_value: float) -> dict:
    """Calculate risk metrics for a Kelly-sized position."""
    position_value = portfolio_value * kelly_fraction
    max_expected_loss = position_value * abs(avg_loss)
    max_loss_pct = max_expected_loss / portfolio_value if portfolio_value else 0
    risk_adjusted_return = kelly_fraction * 0.1

    return {
        "position_value": position_value,
        "max_expected_loss": max_expected_loss,
        "max_loss_pct": max_loss_pct,
        "risk_adjusted_return": risk_adjusted_return,
        "kelly_fraction": kelly_fraction,
    }


__all__ = ["kelly_position_value", "kelly_risk_metrics"]
