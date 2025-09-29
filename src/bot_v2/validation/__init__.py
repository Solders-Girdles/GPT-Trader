"""
Comprehensive validation framework for GPT-Trader V2

Provides input validation, data validation, and configuration validation
with detailed error messages and type checking.
"""

import re
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

import pandas as pd

from ..errors import ValidationError


# Base validators
class Validator:
    """Base validator class.

    Provides optional predicate support so simple validation rules can be
    expressed without creating a dedicated subclass. Subclasses can override
    :meth:`validate` to provide richer behaviour.
    """

    def __init__(
        self,
        error_message: str | None = None,
        predicate: Callable[[Any], bool | tuple[bool, Any]] | None = None,
    ) -> None:
        self.error_message = error_message
        self._predicate = predicate

    def validate(self, value: Any, field_name: str = "value") -> Any:
        """Validate a value and return it if valid.

        When a predicate is supplied the validator will call it and expect a
        truthy result. Predicates may optionally return a ``(bool, value)``
        tuple to allow simple coercion. Without a predicate the validator acts
        as a no-op pass-through.
        """

        if self._predicate is None:
            return value

        try:
            result = self._predicate(value)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = self.error_message or f"{field_name} failed validation"
            raise ValidationError(message, field=field_name, value=value) from exc

        transformed = value
        passed: bool

        if isinstance(result, tuple):
            if len(result) != 2:
                message = self.error_message or f"{field_name} failed validation"
                raise ValidationError(message, field=field_name, value=value)
            passed = bool(result[0])
            transformed = result[1]
        else:
            passed = bool(result)

        if not passed:
            message = self.error_message or f"{field_name} failed validation"
            raise ValidationError(message, field=field_name, value=value)

        return transformed

    def __call__(self, value: Any, field_name: str = "value") -> Any:
        return self.validate(value, field_name)


class TypeValidator(Validator):
    """Validate that value is of specific type"""

    def __init__(self, expected_type: type, error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.expected_type = expected_type

    def validate(self, value: Any, field_name: str = "value") -> Any:
        if not isinstance(value, self.expected_type):
            msg = (
                self.error_message or f"{field_name} must be of type {self.expected_type.__name__}"
            )
            raise ValidationError(msg, field=field_name, value=value)
        return value


class RangeValidator(Validator):
    """Validate that value is within range"""

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        inclusive: bool = True,
        error_message: str | None = None,
    ) -> None:
        super().__init__(error_message)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def validate(self, value: Any, field_name: str = "value") -> Any:
        if self.min_value is not None:
            if self.inclusive and value < self.min_value:
                msg = self.error_message or f"{field_name} must be >= {self.min_value}"
                raise ValidationError(msg, field=field_name, value=value)
            elif not self.inclusive and value <= self.min_value:
                msg = self.error_message or f"{field_name} must be > {self.min_value}"
                raise ValidationError(msg, field=field_name, value=value)

        if self.max_value is not None:
            if self.inclusive and value > self.max_value:
                msg = self.error_message or f"{field_name} must be <= {self.max_value}"
                raise ValidationError(msg, field=field_name, value=value)
            elif not self.inclusive and value >= self.max_value:
                msg = self.error_message or f"{field_name} must be < {self.max_value}"
                raise ValidationError(msg, field=field_name, value=value)

        return value


class ChoiceValidator(Validator):
    """Validate that value is one of allowed choices"""

    def __init__(self, choices: list[Any], error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.choices = choices

    def validate(self, value: Any, field_name: str = "value") -> Any:
        if value not in self.choices:
            msg = self.error_message or f"{field_name} must be one of {self.choices}"
            raise ValidationError(msg, field=field_name, value=value)
        return value


class RegexValidator(Validator):
    """Validate that string matches regex pattern"""

    def __init__(self, pattern: str, error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.pattern = re.compile(pattern)

    def validate(self, value: Any, field_name: str = "value") -> Any:
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field=field_name, value=value)

        if not self.pattern.match(value):
            msg = self.error_message or f"{field_name} does not match required pattern"
            raise ValidationError(msg, field=field_name, value=value)

        return value


# Specialized validators for trading
class SymbolValidator(Validator):
    """Validate trading symbol (equities or crypto).

    Accepts common formats such as:
    - Equities: "AAPL", "MSFT"
    - Crypto spot: "BTC-USD", "ETH-USD"
    - Perpetuals: "BTC-PERP", "ETH-PERP"
    """

    def validate(self, value: Any, field_name: str = "symbol") -> str:
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field=field_name, value=value)

        # Normalize to uppercase
        value = value.upper()

        # Allow simple tickers or hyphenated pairs/suffixes (e.g., BTC-USD, BTC-PERP)
        if not re.match(r"^[A-Z0-9]{1,10}(-[A-Z0-9]{2,10})?$", value):
            raise ValidationError(
                f"{field_name} must be a valid symbol (e.g., AAPL, BTC-USD, BTC-PERP)",
                field=field_name,
                value=value,
            )

        return value


class StrategyNameValidator(Validator):
    """Validate strategy name identifier.

    Accepts names like 'SimpleMAStrategy', 'Momentum', 'mean_reversion'.
    Must start with a letter; allows letters, numbers, underscores and dashes.
    """

    def validate(self, value: Any, field_name: str = "strategy") -> str:
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field=field_name, value=value)
        if not re.match(r"^[A-Za-z][A-Za-z0-9_\-]*$", value):
            raise ValidationError(
                f"{field_name} must start with a letter and contain only letters, numbers, '_' or '-'",
                field=field_name,
                value=value,
            )
        return value


class DateValidator(Validator):
    """Validate date/datetime"""

    def __init__(
        self,
        min_date: date | datetime | None = None,
        max_date: date | datetime | None = None,
        error_message: str | None = None,
    ) -> None:
        super().__init__(error_message)
        self.min_date = min_date
        self.max_date = max_date

    def validate(self, value: Any, field_name: str = "date") -> date | datetime:
        # Convert string to datetime if needed
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                raise ValidationError(
                    f"{field_name} must be a valid date/datetime", field=field_name, value=value
                )

        if not isinstance(value, (date, datetime)):
            raise ValidationError(
                f"{field_name} must be a date or datetime", field=field_name, value=value
            )

        if self.min_date and value < self.min_date:
            raise ValidationError(
                f"{field_name} must be after {self.min_date}", field=field_name, value=value
            )

        if self.max_date and value > self.max_date:
            raise ValidationError(
                f"{field_name} must be before {self.max_date}", field=field_name, value=value
            )

        return value


class PositiveNumberValidator(Validator):
    """Validate positive number"""

    def __init__(self, allow_zero: bool = False, error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.allow_zero = allow_zero

    def validate(self, value: Any, field_name: str = "value") -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be a number", field=field_name, value=value)

        if self.allow_zero and value < 0:
            raise ValidationError(f"{field_name} must be >= 0", field=field_name, value=value)
        elif not self.allow_zero and value <= 0:
            raise ValidationError(f"{field_name} must be > 0", field=field_name, value=value)

        return value


class PercentageValidator(Validator):
    """Validate percentage (0-100 or 0-1)"""

    def __init__(self, as_decimal: bool = True, error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.as_decimal = as_decimal

    def validate(self, value: Any, field_name: str = "percentage") -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be a number", field=field_name, value=value)

        if self.as_decimal:
            if not 0 <= value <= 1:
                raise ValidationError(
                    f"{field_name} must be between 0 and 1", field=field_name, value=value
                )
        else:
            if not 0 <= value <= 100:
                raise ValidationError(
                    f"{field_name} must be between 0 and 100", field=field_name, value=value
                )
            value = value / 100  # Convert to decimal

        return value


# Data validators
class DataFrameValidator(Validator):
    """Validate pandas DataFrame"""

    def __init__(
        self,
        required_columns: list[str] | None = None,
        min_rows: int | None = None,
        error_message: str | None = None,
    ) -> None:
        super().__init__(error_message)
        self.required_columns = required_columns
        self.min_rows = min_rows

    def validate(self, value: Any, field_name: str = "dataframe") -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise ValidationError(
                f"{field_name} must be a pandas DataFrame",
                field=field_name,
                value=type(value).__name__,
            )

        if self.required_columns:
            missing = set(self.required_columns) - set(value.columns)
            if missing:
                raise ValidationError(
                    f"{field_name} missing required columns: {missing}",
                    field=field_name,
                    value=list(value.columns),
                )

        if self.min_rows is not None and len(value) < self.min_rows:
            raise ValidationError(
                f"{field_name} must have at least {self.min_rows} rows",
                field=field_name,
                value=len(value),
            )

        return value


class OHLCDataValidator(DataFrameValidator):
    """Validate OHLC price data"""

    def __init__(self, error_message: str | None = None) -> None:
        super().__init__(
            required_columns=["Open", "High", "Low", "Close", "Volume"],
            min_rows=1,
            error_message=error_message,
        )

    def validate(self, value: Any, field_name: str = "ohlc_data") -> pd.DataFrame:
        df = super().validate(value, field_name)

        # Validate OHLC relationships
        invalid_high_low = df["High"] < df["Low"]
        if invalid_high_low.any():
            raise ValidationError(
                f"{field_name} has invalid High < Low relationships",
                field=field_name,
                value=f"{invalid_high_low.sum()} invalid rows",
            )

        invalid_high = (df["High"] < df["Open"]) | (df["High"] < df["Close"])
        if invalid_high.any():
            raise ValidationError(
                f"{field_name} has High values below Open or Close",
                field=field_name,
                value=f"{invalid_high.sum()} invalid rows",
            )

        invalid_low = (df["Low"] > df["Open"]) | (df["Low"] > df["Close"])
        if invalid_low.any():
            raise ValidationError(
                f"{field_name} has Low values above Open or Close",
                field=field_name,
                value=f"{invalid_low.sum()} invalid rows",
            )

        # Check for negative values
        if (df[["Open", "High", "Low", "Close", "Volume"]] < 0).any().any():
            raise ValidationError(f"{field_name} contains negative values", field=field_name)

        return df


class SeriesValidator(Validator):
    """Validate pandas Series"""

    def __init__(self, error_message: str | None = None) -> None:
        super().__init__(error_message)

    def validate(self, value: Any, field_name: str = "series") -> pd.Series:
        if not isinstance(value, pd.Series):
            raise ValidationError(
                f"{field_name} must be a pandas Series",
                field=field_name,
                value=type(value).__name__,
            )
        return value


# Validation decorator
def validate_inputs(**validators: Validator):
    """Decorator to validate function inputs"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    bound.arguments[param_name] = validator(value, param_name)

            # Call function with validated arguments
            return func(**bound.arguments)

        return wrapper

    return decorator


# Composite validator
class CompositeValidator(Validator):
    """Combine multiple validators"""

    def __init__(self, *validators: Validator) -> None:
        self.validators = validators

    def validate(self, value: Any, field_name: str = "value") -> Any:
        for validator in self.validators:
            value = validator(value, field_name)
        return value


# Configuration validators
def validate_config(config: dict[str, Any], schema: dict[str, Validator]) -> dict[str, Any]:
    """Validate configuration dictionary against schema"""
    validated = {}

    for key, validator in schema.items():
        if key not in config:
            raise ValidationError(f"Missing required config key: {key}", field=key)

        validated[key] = validator(config[key], key)

    # Check for extra keys
    extra_keys = set(config.keys()) - set(schema.keys())
    if extra_keys:
        import logging

        logging.warning(f"Unknown config keys will be ignored: {extra_keys}")

    return validated


# Export main components
__all__ = [
    "Validator",
    "TypeValidator",
    "RangeValidator",
    "ChoiceValidator",
    "RegexValidator",
    "SymbolValidator",
    "DateValidator",
    "PositiveNumberValidator",
    "PercentageValidator",
    "DataFrameValidator",
    "SeriesValidator",
    "OHLCDataValidator",
    "CompositeValidator",
    "validate_inputs",
    "validate_config",
]
