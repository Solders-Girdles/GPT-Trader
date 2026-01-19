"""Tests for data validators (DataFrames)."""

from __future__ import annotations

import pandas as pd
import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.data_validators import DataFrameValidator


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
