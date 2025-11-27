"""Tests for data validators (DataFrames and Series)."""

from __future__ import annotations

import pandas as pd
import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.data_validators import (
    DataFrameValidator,
    OHLCDataValidator,
    SeriesValidator,
)


class TestDataFrameValidator:
    """Tests for DataFrameValidator."""

    def test_validates_empty_dataframe(self) -> None:
        validator = DataFrameValidator()
        df = pd.DataFrame()
        assert validator.validate(df, "data").equals(df)

    def test_validates_dataframe_with_data(self) -> None:
        validator = DataFrameValidator()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = validator.validate(df, "data")
        assert result.equals(df)

    def test_validates_required_columns_present(self) -> None:
        validator = DataFrameValidator(required_columns=["name", "value"])
        df = pd.DataFrame({"name": ["a"], "value": [1], "extra": [True]})
        result = validator.validate(df, "data")
        assert result.equals(df)

    def test_rejects_missing_columns(self) -> None:
        validator = DataFrameValidator(required_columns=["name", "value", "missing"])
        df = pd.DataFrame({"name": ["a"], "value": [1]})
        with pytest.raises(ValidationError, match="missing required columns"):
            validator.validate(df, "data")

    def test_validates_min_rows(self) -> None:
        validator = DataFrameValidator(min_rows=2)
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validator.validate(df, "data")
        assert len(result) == 3

    def test_rejects_insufficient_rows(self) -> None:
        validator = DataFrameValidator(min_rows=5)
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValidationError, match="must have at least 5 rows"):
            validator.validate(df, "data")

    def test_rejects_non_dataframe(self) -> None:
        validator = DataFrameValidator()
        with pytest.raises(ValidationError, match="must be a pandas DataFrame"):
            validator.validate([1, 2, 3], "data")

    def test_rejects_dict(self) -> None:
        validator = DataFrameValidator()
        with pytest.raises(ValidationError, match="must be a pandas DataFrame"):
            validator.validate({"a": 1}, "data")

    def test_field_name_in_error(self) -> None:
        validator = DataFrameValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("not a df", "market_data")
        assert exc_info.value.context["field"] == "market_data"


class TestOHLCDataValidator:
    """Tests for OHLCDataValidator."""

    @pytest.fixture
    def valid_ohlc(self) -> pd.DataFrame:
        """Valid OHLC DataFrame."""
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [102.0, 103.0, 104.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [101.0, 102.0, 103.0],
                "Volume": [1000, 1100, 1200],
            }
        )

    def test_validates_valid_ohlc(self, valid_ohlc: pd.DataFrame) -> None:
        validator = OHLCDataValidator()
        result = validator.validate(valid_ohlc, "ohlc")
        assert result.equals(valid_ohlc)

    def test_validates_flat_candle(self) -> None:
        # All values equal (flat candle)
        validator = OHLCDataValidator()
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [100.0],
                "Low": [100.0],
                "Close": [100.0],
                "Volume": [500],
            }
        )
        result = validator.validate(df, "ohlc")
        assert len(result) == 1

    def test_rejects_missing_columns(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame({"Open": [100.0], "High": [102.0], "Low": [99.0]})
        with pytest.raises(ValidationError, match="missing required columns"):
            validator.validate(df, "ohlc")

    def test_rejects_high_less_than_low(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [98.0],  # Invalid: High < Low
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000],
            }
        )
        with pytest.raises(ValidationError, match="invalid High < Low"):
            validator.validate(df, "ohlc")

    def test_rejects_high_below_open(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame(
            {
                "Open": [105.0],  # Open > High
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000],
            }
        )
        with pytest.raises(ValidationError, match="High values below Open or Close"):
            validator.validate(df, "ohlc")

    def test_rejects_high_below_close(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [105.0],  # Close > High
                "Volume": [1000],
            }
        )
        with pytest.raises(ValidationError, match="High values below Open or Close"):
            validator.validate(df, "ohlc")

    def test_rejects_low_above_open(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame(
            {
                "Open": [98.0],  # Open < Low
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000],
            }
        )
        with pytest.raises(ValidationError, match="Low values above Open or Close"):
            validator.validate(df, "ohlc")

    def test_rejects_low_above_close(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [97.0],  # Close < Low
                "Volume": [1000],
            }
        )
        with pytest.raises(ValidationError, match="Low values above Open or Close"):
            validator.validate(df, "ohlc")

    def test_rejects_negative_price(self) -> None:
        validator = OHLCDataValidator()
        # All prices negative but OHLC relationships valid
        df = pd.DataFrame(
            {
                "Open": [-100.0],
                "High": [-98.0],  # Highest (least negative)
                "Low": [-102.0],  # Lowest (most negative)
                "Close": [-99.0],
                "Volume": [1000],
            }
        )
        with pytest.raises(ValidationError, match="contains negative values"):
            validator.validate(df, "ohlc")

    def test_rejects_negative_volume(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [-100],  # Negative volume
            }
        )
        with pytest.raises(ValidationError, match="contains negative values"):
            validator.validate(df, "ohlc")

    def test_rejects_empty_dataframe(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        with pytest.raises(ValidationError, match="must have at least 1 rows"):
            validator.validate(df, "ohlc")

    def test_validates_multiple_valid_rows(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 99.0],
                "High": [102.0, 103.0, 102.0],
                "Low": [99.0, 100.0, 98.0],
                "Close": [101.0, 102.0, 101.0],
                "Volume": [1000, 1100, 900],
            }
        )
        result = validator.validate(df, "ohlc")
        assert len(result) == 3

    def test_detects_single_invalid_row_among_valid(self) -> None:
        validator = OHLCDataValidator()
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 99.0],
                "High": [102.0, 98.0, 102.0],  # Middle row: High < Low
                "Low": [99.0, 100.0, 98.0],
                "Close": [101.0, 102.0, 101.0],
                "Volume": [1000, 1100, 900],
            }
        )
        with pytest.raises(ValidationError, match="invalid High < Low"):
            validator.validate(df, "ohlc")


class TestSeriesValidator:
    """Tests for SeriesValidator."""

    def test_validates_series(self) -> None:
        validator = SeriesValidator()
        series = pd.Series([1, 2, 3], name="values")
        result = validator.validate(series, "data")
        assert result.equals(series)

    def test_validates_empty_series(self) -> None:
        validator = SeriesValidator()
        series = pd.Series([], dtype=float)
        result = validator.validate(series, "data")
        assert len(result) == 0

    def test_validates_named_series(self) -> None:
        validator = SeriesValidator()
        series = pd.Series([1.0, 2.0], name="prices")
        result = validator.validate(series, "data")
        assert result.name == "prices"

    def test_rejects_list(self) -> None:
        validator = SeriesValidator()
        with pytest.raises(ValidationError, match="must be a pandas Series"):
            validator.validate([1, 2, 3], "series")

    def test_rejects_dataframe(self) -> None:
        validator = SeriesValidator()
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValidationError, match="must be a pandas Series"):
            validator.validate(df, "series")

    def test_rejects_scalar(self) -> None:
        validator = SeriesValidator()
        with pytest.raises(ValidationError, match="must be a pandas Series"):
            validator.validate(42, "series")

    def test_field_name_in_error(self) -> None:
        validator = SeriesValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("not a series", "price_series")
        assert exc_info.value.context["field"] == "price_series"
