"""Core Kelly criterion calculations."""

from __future__ import annotations

from bot_v2.errors import ValidationError
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.validation import PositiveNumberValidator, RangeValidator

logger = get_logger(__name__, component="position_sizing")


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate the optimal Kelly fraction for a trade setup."""
    try:
        RangeValidator(min_value=0.01, max_value=0.99)(win_rate, "win_rate")
        PositiveNumberValidator()(avg_win, "avg_win")

        if avg_loss >= 0:
            raise ValidationError("Average loss must be negative", field="avg_loss", value=avg_loss)

        if abs(avg_loss) < 1e-10:
            raise ValidationError("Average loss too close to zero", field="avg_loss", value=avg_loss)

        odds_ratio = abs(avg_win / avg_loss)
        if odds_ratio > 100:
            logger.warning("Extreme odds ratio %.2f, capping Kelly calculation", odds_ratio)
            odds_ratio = 100

        kelly_fraction = win_rate - (1 - win_rate) / odds_ratio
        kelly_fraction = max(0.0, kelly_fraction)

        if kelly_fraction > 0.5:
            logger.warning(
                "High Kelly fraction %.4f, consider using fractional Kelly", kelly_fraction
            )

        return kelly_fraction

    except ValidationError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ValidationError(
            f"Kelly calculation failed: {exc}", field="kelly_calculation"
        ) from exc


def fractional_kelly(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25) -> float:
    """Apply fractional Kelly sizing for a more conservative position."""
    full_kelly = kelly_criterion(win_rate, avg_win, avg_loss)
    return full_kelly * fraction


def validate_kelly_inputs(win_rate: float, avg_win: float, avg_loss: float) -> list[str]:
    """Validate inputs for Kelly criterion calculations."""
    errors: list[str] = []

    if not (0 < win_rate < 1):
        errors.append(f"Win rate must be between 0 and 1, got {win_rate}")

    if avg_win <= 0:
        errors.append(f"Average win must be positive, got {avg_win}")

    if avg_loss >= 0:
        errors.append(f"Average loss must be negative, got {avg_loss}")

    if win_rate > 0 and avg_win > 0 and avg_loss < 0:
        expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss
        if expected_value <= 0:
            errors.append(f"Strategy has negative expected value: {expected_value:.4f}")

    return errors
