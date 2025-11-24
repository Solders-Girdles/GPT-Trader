"""
Data validators for pandas DataFrames and Series.
"""

from typing import Any

from gpt_trader.errors import ValidationError
from gpt_trader.utilities.importing import optional_import

from .base_validators import Validator

# Optional pandas import
pandas = optional_import("pandas")


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

    def validate(self, value: Any, field_name: str = "dataframe") -> Any:
        if pandas is None:
            raise ValidationError(
                f"{field_name} validation requires pandas but it's not available",
                field=field_name,
            )

        if not isinstance(value, pandas.DataFrame):
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

    def validate(self, value: Any, field_name: str = "ohlc_data") -> Any:
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

    def validate(self, value: Any, field_name: str = "series") -> Any:
        if pandas is None:
            raise ValidationError(
                f"{field_name} validation requires pandas but it's not available",
                field=field_name,
            )

        if not isinstance(value, pandas.Series):
            raise ValidationError(
                f"{field_name} must be a pandas Series",
                field=field_name,
                value=type(value).__name__,
            )
        return value
