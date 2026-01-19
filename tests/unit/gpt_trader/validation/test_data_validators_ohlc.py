"""Tests for data validators (OHLC data)."""

from __future__ import annotations

import pandas as pd
import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.data_validators import OHLCDataValidator


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
