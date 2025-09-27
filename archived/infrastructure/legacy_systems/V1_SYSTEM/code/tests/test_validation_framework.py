"""Unit tests for the validation framework."""

import numpy as np
import pandas as pd
from bot.validation import (
    DataValidator,
    FinancialValidator,
    MathValidator,
    ValidationConfig,
    ValidationLevel,
    ValidationResult,
    get_data_validator,
    get_financial_validator,
    get_math_validator,
)


class TestMathValidator:
    """Test suite for MathValidator."""

    def test_safe_divide_scalar(self):
        """Test safe division with scalar values."""
        validator = MathValidator()

        # Normal division
        assert validator.safe_divide(10.0, 2.0) == 5.0
        assert validator.safe_divide(100, 4) == 25.0

        # Division by zero
        assert validator.safe_divide(10.0, 0.0, default=999.0) == 999.0
        assert validator.safe_divide(10.0, 1e-10, default=-1.0) == -1.0

        # NaN handling
        assert validator.safe_divide(np.nan, 2.0, default=0.0) == 0.0
        assert validator.safe_divide(2.0, np.nan, default=0.0) == 0.0

        # Inf handling
        assert validator.safe_divide(np.inf, 2.0, default=0.0) == 0.0
        assert validator.safe_divide(2.0, np.inf, default=0.0) == 0.0

    def test_safe_divide_array(self):
        """Test safe division with numpy arrays."""
        validator = MathValidator()

        # Normal array division
        num = np.array([10.0, 20.0, 30.0])
        denom = np.array([2.0, 4.0, 5.0])
        result = validator.safe_divide(num, denom)
        np.testing.assert_array_equal(result, [5.0, 5.0, 6.0])

        # Division by zero in array
        denom_with_zero = np.array([2.0, 0.0, 5.0])
        result = validator.safe_divide(num, denom_with_zero, default=0.0)
        np.testing.assert_array_equal(result, [5.0, 0.0, 6.0])

        # NaN in array
        num_with_nan = np.array([10.0, np.nan, 30.0])
        result = validator.safe_divide(num_with_nan, denom, default=-1.0)
        np.testing.assert_array_equal(result, [5.0, -1.0, 6.0])

        # Scalar denominator
        result = validator.safe_divide(num, 2.0)
        np.testing.assert_array_equal(result, [5.0, 10.0, 15.0])

        # Zero scalar denominator
        result = validator.safe_divide(num, 0.0, default=99.0)
        np.testing.assert_array_equal(result, [99.0, 99.0, 99.0])

    def test_safe_divide_series(self):
        """Test safe division with pandas Series."""
        validator = MathValidator()

        # Normal series division
        num = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])
        denom = pd.Series([2.0, 4.0, 5.0], index=["a", "b", "c"])
        result = validator.safe_divide(num, denom)
        pd.testing.assert_series_equal(result, pd.Series([5.0, 5.0, 6.0], index=["a", "b", "c"]))

        # Division by zero in series
        denom_with_zero = pd.Series([2.0, 0.0, 5.0], index=["a", "b", "c"])
        result = validator.safe_divide(num, denom_with_zero, default=0.0)
        pd.testing.assert_series_equal(result, pd.Series([5.0, 0.0, 6.0], index=["a", "b", "c"]))

    def test_safe_log(self):
        """Test safe logarithm."""
        validator = MathValidator()

        # Normal log
        assert np.isclose(validator.safe_log(np.e), 1.0)
        assert np.isclose(validator.safe_log(10.0), np.log(10.0))

        # Invalid values
        assert validator.safe_log(0.0, default=-999.0) == -999.0
        assert validator.safe_log(-1.0, default=-999.0) == -999.0
        assert validator.safe_log(np.nan, default=-999.0) == -999.0
        assert validator.safe_log(np.inf, default=-999.0) == -999.0

        # Array log
        arr = np.array([1.0, np.e, 0.0, -1.0])
        result = validator.safe_log(arr, default=0.0)
        expected = np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_safe_sqrt(self):
        """Test safe square root."""
        validator = MathValidator()

        # Normal sqrt
        assert validator.safe_sqrt(4.0) == 2.0
        assert validator.safe_sqrt(9.0) == 3.0

        # Invalid values
        assert validator.safe_sqrt(-1.0, default=0.0) == 0.0
        assert validator.safe_sqrt(np.nan, default=0.0) == 0.0
        assert validator.safe_sqrt(np.inf, default=0.0) == 0.0

        # Array sqrt
        arr = np.array([4.0, 9.0, -1.0, np.nan])
        result = validator.safe_sqrt(arr, default=0.0)
        expected = np.array([2.0, 3.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)


class TestFinancialValidator:
    """Test suite for FinancialValidator."""

    def test_validate_price(self):
        """Test price validation."""
        validator = FinancialValidator()

        # Valid price
        result = validator.validate_price(100.50, "AAPL")
        assert result.is_valid
        assert result.value == 100.50

        # NaN price
        result = validator.validate_price(np.nan, "AAPL")
        assert not result.is_valid
        assert result.value == 0.0
        assert "NaN" in result.issues[0]

        # Negative price
        result = validator.validate_price(-10.0, "AAPL")
        assert not result.is_valid
        assert result.value == 10.0  # Converted to absolute
        assert "Negative" in result.issues[0]

        # Below minimum
        result = validator.validate_price(0.001, "AAPL", min_price=0.01)
        assert not result.is_valid
        assert result.value == 0.0

        # Above maximum
        result = validator.validate_price(2_000_000, "AAPL", max_price=1_000_000)
        assert not result.is_valid
        assert result.value == 1_000_000

    def test_validate_quantity(self):
        """Test quantity validation."""
        validator = FinancialValidator()

        # Valid quantity
        result = validator.validate_quantity(100, "AAPL")
        assert result.is_valid
        assert result.value == 100

        # NaN quantity
        result = validator.validate_quantity(np.nan, "AAPL")
        assert not result.is_valid
        assert result.value == 0.0

        # Below minimum
        result = validator.validate_quantity(-10, "AAPL", min_qty=0)
        assert not result.is_valid
        assert result.value == 0

        # Above maximum
        result = validator.validate_quantity(2_000_000, "AAPL", max_qty=1_000_000)
        assert not result.is_valid
        assert result.value == 1_000_000

        # Rounding
        result = validator.validate_quantity(10.5678, "AAPL", round_to=2)
        assert result.value == 10.57
        assert "Rounded" in result.fixes_applied[0]

    def test_validate_portfolio_weight(self):
        """Test portfolio weight validation."""
        validator = FinancialValidator()

        # Valid weight
        result = validator.validate_portfolio_weight(0.25, "AAPL")
        assert result.is_valid
        assert result.value == 0.25

        # Above 1.0
        result = validator.validate_portfolio_weight(1.5, "AAPL")
        assert not result.is_valid
        assert result.value == 1.0

        # Below 0.0
        result = validator.validate_portfolio_weight(-0.1, "AAPL")
        assert not result.is_valid
        assert result.value == 0.0

        # NaN weight
        result = validator.validate_portfolio_weight(np.nan, "AAPL")
        assert not result.is_valid
        assert result.value == 0.0

    def test_validate_return(self):
        """Test return validation."""
        validator = FinancialValidator()

        # Normal return
        result = validator.validate_return(0.05, "daily")
        assert result.is_valid
        assert result.value == 0.05

        # Extreme positive return
        result = validator.validate_return(15.0, "daily", max_return=10.0)
        assert not result.is_valid
        assert result.value == 10.0

        # Extreme negative return
        result = validator.validate_return(-15.0, "daily", max_return=10.0)
        assert not result.is_valid
        assert result.value == -10.0

        # NaN return
        result = validator.validate_return(np.nan, "daily")
        assert not result.is_valid
        assert result.value == 0.0


class TestDataValidator:
    """Test suite for DataValidator."""

    def test_validate_ohlcv(self):
        """Test OHLCV data validation."""
        validator = DataValidator()

        # Valid OHLCV data
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [99, 100, 101],
                "Close": [104, 105, 106],
                "Volume": [1000, 1100, 1200],
            }
        )

        validated_df, result = validator.validate_ohlcv(df)
        assert result.is_valid
        assert len(result.issues) == 0

        # High < Low (invalid)
        df_invalid = df.copy()
        df_invalid.loc[0, "High"] = 95  # Less than Low
        validated_df, result = validator.validate_ohlcv(df_invalid, repair=True)
        assert not result.is_valid
        assert "High < Low" in result.issues[0]
        assert validated_df.loc[0, "High"] == 99  # Should be swapped

        # Close > High (invalid)
        df_invalid = df.copy()
        df_invalid.loc[0, "Close"] = 110  # Greater than High
        validated_df, result = validator.validate_ohlcv(df_invalid, repair=True)
        assert not result.is_valid
        assert "Close > High" in result.issues[0]
        assert validated_df.loc[0, "High"] == 110  # High adjusted to Close

        # NaN values
        df_nan = df.copy()
        df_nan.loc[1, "Close"] = np.nan
        validated_df, result = validator.validate_ohlcv(df_nan, repair=True)
        assert not result.is_valid
        assert "NaN" in result.issues[0]
        assert not np.isnan(validated_df.loc[1, "Close"])  # Should be filled

        # Negative volume
        df_neg_vol = df.copy()
        df_neg_vol.loc[0, "Volume"] = -100
        validated_df, result = validator.validate_ohlcv(df_neg_vol, repair=True)
        assert not result.is_valid
        assert "Negative volume" in result.issues[0]
        assert validated_df.loc[0, "Volume"] == 0

        # Zero/negative prices
        df_zero = df.copy()
        df_zero.loc[1, "Low"] = 0
        validated_df, result = validator.validate_ohlcv(df_zero, repair=True)
        assert not result.is_valid
        assert "Zero/negative" in result.issues[0]
        assert validated_df.loc[1, "Low"] > 0  # Should be replaced

    def test_validate_returns_series(self):
        """Test returns series validation."""
        validator = DataValidator()

        # Normal returns
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        validated_returns, result = validator.validate_returns_series(returns)
        assert result.is_valid
        pd.testing.assert_series_equal(returns, validated_returns)

        # NaN values
        returns_nan = returns.copy()
        returns_nan.iloc[2] = np.nan
        validated_returns, result = validator.validate_returns_series(returns_nan, repair=True)
        assert not result.is_valid
        assert "NaN" in result.issues[0]
        assert validated_returns.iloc[2] == 0.0

        # Infinite values
        returns_inf = returns.copy()
        returns_inf.iloc[1] = np.inf
        validated_returns, result = validator.validate_returns_series(returns_inf, repair=True)
        assert not result.is_valid
        assert "infinite" in result.issues[0]
        assert validated_returns.iloc[1] == 0.0

        # Extreme returns
        returns_extreme = pd.Series([0.01, 15.0, -12.0, 0.02])
        validated_returns, result = validator.validate_returns_series(
            returns_extreme, max_return=10.0, repair=True
        )
        assert not result.is_valid
        assert "extreme" in result.issues[0]
        assert validated_returns.iloc[1] == 10.0
        assert validated_returns.iloc[2] == -10.0


class TestValidationConfig:
    """Test validation configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = ValidationConfig()
        assert config.level == ValidationLevel.MODERATE
        assert config.log_issues is True
        assert config.raise_on_critical is True
        assert config.epsilon == 1e-9

    def test_strict_config(self):
        """Test strict configuration."""
        config = ValidationConfig(level=ValidationLevel.STRICT)
        validator = FinancialValidator(config)

        # In strict mode, below minimum price should clamp to minimum
        result = validator.validate_price(0.001, "TEST", min_price=0.01)
        assert result.value == 0.01  # Clamped to minimum in strict mode

    def test_lenient_config(self):
        """Test lenient configuration."""
        config = ValidationConfig(level=ValidationLevel.LENIENT, log_issues=False)
        validator = MathValidator(config)

        # Should still handle division by zero gracefully
        result = validator.safe_divide(10.0, 0.0, default=999.0)
        assert result == 999.0


class TestSingletons:
    """Test singleton pattern for validators."""

    def test_math_validator_singleton(self):
        """Test MathValidator singleton."""
        validator1 = get_math_validator()
        validator2 = get_math_validator()
        assert validator1 is validator2

    def test_financial_validator_singleton(self):
        """Test FinancialValidator singleton."""
        validator1 = get_financial_validator()
        validator2 = get_financial_validator()
        assert validator1 is validator2

    def test_data_validator_singleton(self):
        """Test DataValidator singleton."""
        validator1 = get_data_validator()
        validator2 = get_data_validator()
        assert validator1 is validator2


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_properties(self):
        """Test ValidationResult properties."""
        # Valid result
        result = ValidationResult(
            is_valid=True, value=100.0, original_value=100.0, issues=[], fixes_applied=[]
        )
        assert not result.has_issues
        assert not result.was_modified

        # Invalid result with fixes
        result = ValidationResult(
            is_valid=False,
            value=0.0,
            original_value=np.nan,
            issues=["Value is NaN"],
            fixes_applied=["Set to 0.0"],
        )
        assert result.has_issues
        assert result.was_modified
