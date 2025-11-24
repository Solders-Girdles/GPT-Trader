"""Trading hours validation."""

from datetime import datetime

from .input_sanitizer import ValidationResult


class TradingHoursValidator:
    """Validate trading is allowed during market hours."""

    @classmethod
    def check_trading_hours(
        cls, symbol: str, timestamp: datetime | None = None
    ) -> ValidationResult:
        """Check if trading is allowed at current time"""
        errors = []
        if timestamp is None:
            timestamp = datetime.now()
        elif not isinstance(timestamp, datetime):
            return ValidationResult(
                is_valid=False, errors=["Invalid timestamp provided"], sanitized_value=None
            )

        # Market hours (simplified - NYSE: 9:30 AM - 4:00 PM ET)
        hour = timestamp.hour
        minute = timestamp.minute
        weekday = timestamp.weekday()

        # Check weekend
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            errors.append("Market closed on weekends")

        # Check market hours (simplified, not accounting for timezone)
        elif hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
            errors.append("Outside market hours (9:30 AM - 4:00 PM ET)")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)
