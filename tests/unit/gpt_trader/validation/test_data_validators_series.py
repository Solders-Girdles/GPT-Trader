"""Tests for data validators (Series)."""

from __future__ import annotations

import pandas as pd
import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.data_validators import SeriesValidator


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
