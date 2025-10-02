"""Comprehensive tests for validation module to improve coverage."""

import pytest
import pandas as pd
from datetime import datetime, date
from bot_v2.errors import ValidationError
from bot_v2.validation import (
    TypeValidator,
    RangeValidator,
    ChoiceValidator,
    RegexValidator,
    SymbolValidator,
    StrategyNameValidator,
    DateValidator,
    PositiveNumberValidator,
    PercentageValidator,
    DataFrameValidator,
    OHLCDataValidator,
    SeriesValidator,
    Validator,
)


class TestTypeValidator:
    """Test TypeValidator"""

    def test_type_validator_success(self):
        """Test TypeValidator with correct type"""
        validator = TypeValidator(int)
        assert validator.validate(42, "number") == 42

    def test_type_validator_failure(self):
        """Test TypeValidator with wrong type"""
        validator = TypeValidator(int)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("not an int", "number")
        assert "must be of type int" in str(exc_info.value)

    def test_type_validator_custom_message(self):
        """Test TypeValidator with custom error message"""
        validator = TypeValidator(str, error_message="Value must be text")
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(123, "text")
        assert "Value must be text" in str(exc_info.value)


class TestRangeValidator:
    """Test RangeValidator"""

    def test_range_validator_non_inclusive_min(self):
        """Test RangeValidator with non-inclusive minimum"""
        validator = RangeValidator(min_value=0, inclusive=False)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(0, "value")
        assert "must be > 0" in str(exc_info.value)

    def test_range_validator_non_inclusive_max(self):
        """Test RangeValidator with non-inclusive maximum"""
        validator = RangeValidator(max_value=100, inclusive=False)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(100, "value")
        assert "must be < 100" in str(exc_info.value)

    def test_range_validator_inclusive_within_range(self):
        """Test RangeValidator within inclusive range"""
        validator = RangeValidator(min_value=0, max_value=100, inclusive=True)
        assert validator.validate(0, "value") == 0
        assert validator.validate(100, "value") == 100
        assert validator.validate(50, "value") == 50


class TestRegexValidator:
    """Test RegexValidator"""

    def test_regex_validator_non_string_input(self):
        """Test RegexValidator with non-string input"""
        validator = RegexValidator(r"^\d+$")
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(123, "code")
        assert "must be a string" in str(exc_info.value)

    def test_regex_validator_pattern_mismatch(self):
        """Test RegexValidator with pattern mismatch"""
        validator = RegexValidator(r"^\d{3}$")
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("abc", "code")
        assert "does not match required pattern" in str(exc_info.value)

    def test_regex_validator_custom_message(self):
        """Test RegexValidator with custom error message"""
        validator = RegexValidator(r"^\d+$", error_message="Must be digits only")
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("abc123", "code")
        assert "Must be digits only" in str(exc_info.value)


class TestSymbolValidator:
    """Test SymbolValidator"""

    def test_symbol_validator_non_string(self):
        """Test SymbolValidator with non-string input"""
        validator = SymbolValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(12345, "symbol")
        assert "must be a string" in str(exc_info.value)

    def test_symbol_validator_invalid_format(self):
        """Test SymbolValidator with invalid format"""
        validator = SymbolValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("invalid symbol!", "symbol")
        assert "must be a valid symbol" in str(exc_info.value)

    def test_symbol_validator_valid_formats(self):
        """Test SymbolValidator with valid formats"""
        validator = SymbolValidator()
        assert validator.validate("AAPL", "symbol") == "AAPL"
        assert validator.validate("btc-usd", "symbol") == "BTC-USD"
        assert validator.validate("eth-perp", "symbol") == "ETH-PERP"


class TestStrategyNameValidator:
    """Test StrategyNameValidator"""

    def test_strategy_name_validator_non_string(self):
        """Test StrategyNameValidator with non-string input"""
        validator = StrategyNameValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(123, "strategy")
        assert "must be a string" in str(exc_info.value)

    def test_strategy_name_validator_invalid_start(self):
        """Test StrategyNameValidator with invalid start character"""
        validator = StrategyNameValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("123strategy", "strategy")
        assert "must start with a letter" in str(exc_info.value)

    def test_strategy_name_validator_invalid_characters(self):
        """Test StrategyNameValidator with invalid characters"""
        validator = StrategyNameValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("strategy!", "strategy")
        assert "must start with a letter" in str(exc_info.value)

    def test_strategy_name_validator_valid_names(self):
        """Test StrategyNameValidator with valid names"""
        validator = StrategyNameValidator()
        assert validator.validate("SimpleMA", "strategy") == "SimpleMA"
        assert validator.validate("mean_reversion", "strategy") == "mean_reversion"
        assert validator.validate("Strategy-V2", "strategy") == "Strategy-V2"


class TestDateValidator:
    """Test DateValidator"""

    def test_date_validator_string_conversion(self):
        """Test DateValidator converts ISO string to datetime"""
        validator = DateValidator()
        result = validator.validate("2024-01-15", "date")
        assert isinstance(result, datetime)

    def test_date_validator_invalid_string(self):
        """Test DateValidator with invalid date string"""
        validator = DateValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("not a date", "date")
        assert "must be a valid date/datetime" in str(exc_info.value)

    def test_date_validator_invalid_type(self):
        """Test DateValidator with invalid type"""
        validator = DateValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(123, "date")
        assert "must be a date or datetime" in str(exc_info.value)

    def test_date_validator_min_date_check(self):
        """Test DateValidator with minimum date constraint"""
        min_date = datetime(2024, 1, 1)
        validator = DateValidator(min_date=min_date)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(datetime(2023, 12, 31), "date")
        assert "must be after" in str(exc_info.value)

    def test_date_validator_max_date_check(self):
        """Test DateValidator with maximum date constraint"""
        max_date = datetime(2024, 12, 31)
        validator = DateValidator(max_date=max_date)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(datetime(2025, 1, 1), "date")
        assert "must be before" in str(exc_info.value)

    def test_date_validator_within_range(self):
        """Test DateValidator within valid range"""
        min_date = datetime(2024, 1, 1)
        max_date = datetime(2024, 12, 31)
        validator = DateValidator(min_date=min_date, max_date=max_date)
        result = validator.validate(datetime(2024, 6, 15), "date")
        assert result == datetime(2024, 6, 15)


class TestPositiveNumberValidator:
    """Test PositiveNumberValidator"""

    def test_positive_number_validator_invalid_type(self):
        """Test PositiveNumberValidator with non-numeric input"""
        validator = PositiveNumberValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("not a number", "value")
        assert "must be a number" in str(exc_info.value)

    def test_positive_number_validator_negative(self):
        """Test PositiveNumberValidator with negative number"""
        validator = PositiveNumberValidator(allow_zero=False)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(-5, "value")
        assert "must be > 0" in str(exc_info.value)

    def test_positive_number_validator_zero_not_allowed(self):
        """Test PositiveNumberValidator with zero when not allowed"""
        validator = PositiveNumberValidator(allow_zero=False)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(0, "value")
        assert "must be > 0" in str(exc_info.value)

    def test_positive_number_validator_zero_allowed(self):
        """Test PositiveNumberValidator with zero when allowed"""
        validator = PositiveNumberValidator(allow_zero=True)
        assert validator.validate(0, "value") == 0.0

    def test_positive_number_validator_negative_with_allow_zero(self):
        """Test PositiveNumberValidator with negative when zero allowed"""
        validator = PositiveNumberValidator(allow_zero=True)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(-1, "value")
        assert "must be >= 0" in str(exc_info.value)


class TestPercentageValidator:
    """Test PercentageValidator"""

    def test_percentage_validator_invalid_type(self):
        """Test PercentageValidator with non-numeric input"""
        validator = PercentageValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("not a number", "percentage")
        assert "must be a number" in str(exc_info.value)

    def test_percentage_validator_decimal_out_of_range(self):
        """Test PercentageValidator with decimal out of range"""
        validator = PercentageValidator(as_decimal=True)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(1.5, "percentage")
        assert "must be between 0 and 1" in str(exc_info.value)

    def test_percentage_validator_percent_out_of_range(self):
        """Test PercentageValidator with percentage out of range"""
        validator = PercentageValidator(as_decimal=False)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(150, "percentage")
        assert "must be between 0 and 100" in str(exc_info.value)

    def test_percentage_validator_converts_percent_to_decimal(self):
        """Test PercentageValidator converts percentage to decimal"""
        validator = PercentageValidator(as_decimal=False)
        result = validator.validate(50, "percentage")
        assert result == 0.5


class TestDataFrameValidator:
    """Test DataFrameValidator"""

    def test_dataframe_validator_non_dataframe(self):
        """Test DataFrameValidator with non-DataFrame input"""
        validator = DataFrameValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate([1, 2, 3], "dataframe")
        assert "must be a pandas DataFrame" in str(exc_info.value)

    def test_dataframe_validator_missing_columns(self):
        """Test DataFrameValidator with missing required columns"""
        validator = DataFrameValidator(required_columns=["a", "b", "c"])
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(df, "dataframe")
        assert "missing required columns" in str(exc_info.value)
        assert "'c'" in str(exc_info.value)

    def test_dataframe_validator_insufficient_rows(self):
        """Test DataFrameValidator with insufficient rows"""
        validator = DataFrameValidator(min_rows=10)
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(df, "dataframe")
        assert "must have at least 10 rows" in str(exc_info.value)

    def test_dataframe_validator_valid(self):
        """Test DataFrameValidator with valid DataFrame"""
        validator = DataFrameValidator(required_columns=["a", "b"], min_rows=2)
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = validator.validate(df, "dataframe")
        assert isinstance(result, pd.DataFrame)


class TestOHLCDataValidator:
    """Test OHLCDataValidator"""

    def test_ohlc_validator_missing_columns(self):
        """Test OHLCDataValidator with missing required columns"""
        validator = OHLCDataValidator()
        df = pd.DataFrame({"Open": [100, 101], "High": [102, 103]})
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(df, "ohlc_data")
        assert "missing required columns" in str(exc_info.value)

    def test_ohlc_validator_invalid_high_low(self):
        """Test OHLCDataValidator with High < Low"""
        validator = OHLCDataValidator()
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 99],  # High < Low on second row
            "Low": [98, 100],
            "Close": [101, 100],
            "Volume": [1000, 1100],
        })
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(df, "ohlc_data")
        assert "invalid High < Low" in str(exc_info.value)

    def test_ohlc_validator_high_below_open(self):
        """Test OHLCDataValidator with High below Open"""
        validator = OHLCDataValidator()
        df = pd.DataFrame({
            "Open": [100, 105],  # Open > High on second row
            "High": [102, 103],
            "Low": [98, 102],
            "Close": [101, 103],
            "Volume": [1000, 1100],
        })
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(df, "ohlc_data")
        assert "High values below Open or Close" in str(exc_info.value)

    def test_ohlc_validator_high_below_close(self):
        """Test OHLCDataValidator with High below Close"""
        validator = OHLCDataValidator()
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [98, 100],
            "Close": [101, 105],  # Close > High on second row
            "Volume": [1000, 1100],
        })
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(df, "ohlc_data")
        assert "High values below Open or Close" in str(exc_info.value)

    def test_ohlc_validator_low_above_open(self):
        """Test OHLCDataValidator with Low above Open"""
        validator = OHLCDataValidator()
        df = pd.DataFrame({
            "Open": [100, 99],  # Open < Low on second row
            "High": [102, 103],
            "Low": [98, 100],
            "Close": [101, 102],
            "Volume": [1000, 1100],
        })
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(df, "ohlc_data")
        assert "Low values above Open or Close" in str(exc_info.value)

    def test_ohlc_validator_low_above_close(self):
        """Test OHLCDataValidator with Low above Close"""
        validator = OHLCDataValidator()
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [98, 100],
            "Close": [101, 99],  # Close < Low on second row
            "Volume": [1000, 1100],
        })
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(df, "ohlc_data")
        assert "Low values above Open or Close" in str(exc_info.value)

    def test_ohlc_validator_negative_values(self):
        """Test OHLCDataValidator with negative values"""
        validator = OHLCDataValidator()
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [98, -5],  # Negative value
            "Close": [101, 100],
            "Volume": [1000, 1100],
        })
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(df, "ohlc_data")
        assert "contains negative values" in str(exc_info.value)

    def test_ohlc_validator_valid_data(self):
        """Test OHLCDataValidator with valid OHLC data"""
        validator = OHLCDataValidator()
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [103, 104, 105],
            "Volume": [1000, 1100, 1200],
        })
        result = validator.validate(df, "ohlc_data")
        assert isinstance(result, pd.DataFrame)


class TestSeriesValidator:
    """Test SeriesValidator"""

    def test_series_validator_non_series(self):
        """Test SeriesValidator with non-Series input"""
        validator = SeriesValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate([1, 2, 3], "series")
        assert "must be a pandas Series" in str(exc_info.value)

    def test_series_validator_valid(self):
        """Test SeriesValidator with valid Series"""
        validator = SeriesValidator()
        series = pd.Series([1, 2, 3, 4, 5])
        result = validator.validate(series, "series")
        assert isinstance(result, pd.Series)


class TestValidatorPredicate:
    """Test Validator with predicate functions"""

    def test_validator_predicate_exception(self):
        """Test Validator handles predicate exceptions"""
        validator = Validator(predicate=lambda x: 1 / 0)  # Will raise ZeroDivisionError
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(10, "value")
        assert "failed validation" in str(exc_info.value)

    def test_validator_predicate_invalid_tuple_length(self):
        """Test Validator handles invalid tuple length from predicate"""
        validator = Validator(predicate=lambda x: (True, x, "extra"))  # 3-tuple instead of 2
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(10, "value")
        assert "failed validation" in str(exc_info.value)

    def test_validator_callable(self):
        """Test Validator is callable"""
        validator = Validator(predicate=lambda x: x > 0)
        assert validator(5, "value") == 5
        with pytest.raises(ValidationError):
            validator(-5, "value")
