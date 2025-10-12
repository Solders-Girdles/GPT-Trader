"""Common utility patterns and helpers used throughout the codebase."""

from __future__ import annotations

import decimal
from decimal import Decimal
from typing import Any, TypeVar

from ..errors import ValidationError

T = TypeVar("T")


def validate_decimal_positive(
    value: Any,
    field_name: str = "value",
    allow_zero: bool = False,
) -> Decimal:
    """Validate that a value is a positive decimal.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        allow_zero: Whether zero values are allowed

    Returns:
        Validated Decimal value

    Raises:
        ValidationError: If validation fails
    """
    try:
        decimal_value = Decimal(str(value))
    except (TypeError, ValueError, decimal.InvalidOperation) as exc:
        raise ValidationError(
            f"{field_name} must be a valid number", field=field_name, value=value
        ) from exc

    if allow_zero:
        if decimal_value < 0:
            raise ValidationError(
                f"{field_name} must be >= 0", field=field_name, value=decimal_value
            )
    else:
        if decimal_value <= 0:
            raise ValidationError(
                f"{field_name} must be > 0", field=field_name, value=decimal_value
            )

    return decimal_value


def validate_decimal_range(
    value: Any,
    field_name: str = "value",
    min_value: Decimal | None = None,
    max_value: Decimal | None = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> Decimal:
    """Validate that a decimal value is within a specified range.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        min_value: Minimum allowed value (None for no minimum)
        max_value: Maximum allowed value (None for no maximum)
        inclusive_min: Whether minimum is inclusive
        inclusive_max: Whether maximum is inclusive

    Returns:
        Validated Decimal value

    Raises:
        ValidationError: If validation fails
    """
    try:
        decimal_value = Decimal(str(value))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{field_name} must be a valid number", field=field_name, value=value
        ) from exc

    if min_value is not None:
        if inclusive_min:
            if decimal_value < min_value:
                raise ValidationError(
                    f"{field_name} must be >= {min_value}", field=field_name, value=decimal_value
                )
        else:
            if decimal_value <= min_value:
                raise ValidationError(
                    f"{field_name} must be > {min_value}", field=field_name, value=decimal_value
                )

    if max_value is not None:
        if inclusive_max:
            if decimal_value > max_value:
                raise ValidationError(
                    f"{field_name} must be <= {max_value}", field=field_name, value=decimal_value
                )
        else:
            if decimal_value >= max_value:
                raise ValidationError(
                    f"{field_name} must be < {max_value}", field=field_name, value=decimal_value
                )

    return decimal_value


def safe_decimal_division(
    numerator: Decimal | int | float | str,
    denominator: Decimal | int | float | str,
    default: Decimal | None = None,
    field_name: str = "result",
) -> Decimal:
    """Safely perform decimal division with zero-division protection.

    Args:
        numerator: The dividend
        denominator: The divisor
        default: Default value if division fails (None to raise)
        field_name: Name of the field for error messages

    Returns:
        Result of division or default value

    Raises:
        ValidationError: If division fails and no default is provided
    """
    try:
        num_decimal = Decimal(str(numerator))
        den_decimal = Decimal(str(denominator))

        if den_decimal == 0:
            if default is not None:
                return default
            raise ValidationError(f"{field_name}: Division by zero", field=field_name)

        return num_decimal / den_decimal
    except (TypeError, ValueError, decimal.InvalidOperation) as exc:
        if default is not None:
            return default
        raise ValidationError(
            f"{field_name}: Invalid division operation", field=field_name
        ) from exc


def format_decimal(
    value: Decimal | str | int | float,
    decimal_places: int | None = None,
    strip_trailing_zeros: bool = True,
) -> str:
    """Format a decimal value for display.

    Args:
        value: Decimal value to format (accepts multiple types)
        decimal_places: Number of decimal places (auto-detect if None)
        strip_trailing_zeros: Whether to strip trailing zeros

    Returns:
        Formatted string representation
    """
    # Convert to Decimal first
    if not isinstance(value, Decimal):
        try:
            value = Decimal(str(value))
        except (TypeError, ValueError, decimal.InvalidOperation):
            return str(value)

    result = format(value, "f")  # Use format to avoid scientific notation

    if decimal_places is not None:
        # Use Decimal quantize for proper rounding
        if decimal_places == 0:
            quantize_str = "1"
        else:
            quantize_str = "0." + "0" * (decimal_places - 1) + "1"
        value = value.quantize(Decimal(quantize_str))
        result = format(value, "f")

        # Ensure we have exactly the right number of decimal places
        if decimal_places > 0:
            if "." not in result:
                result += "." + "0" * decimal_places
            else:
                integer_part, decimal_part = result.split(".", 1)
                if len(decimal_part) < decimal_places:
                    decimal_part = decimal_part.ljust(decimal_places, "0")
                    result = f"{integer_part}.{decimal_part}"
                elif len(decimal_part) > decimal_places:
                    decimal_part = decimal_part[:decimal_places]
                    result = f"{integer_part}.{decimal_part}"

        # Don't strip zeros when decimal_places is explicitly specified
        # unless strip_trailing_zeros is True and decimal_places is 0
        if strip_trailing_zeros and decimal_places == 0 and "." in result:
            result = result.split(".")[0]
    elif strip_trailing_zeros and "." in result:
        result = result.rstrip("0").rstrip(".")

    return result


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
) -> float:
    """Calculate exponential backoff delay.

    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        multiplier: Backoff multiplier
        max_delay: Maximum delay in seconds

    Returns:
        Calculated delay in seconds
    """
    delay = base_delay * (multiplier**attempt)
    return min(delay, max_delay)
