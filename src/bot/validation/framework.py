"""Comprehensive validation framework for GPT-Trader.

This module provides safe mathematical operations, bounds checking,
and data validation to prevent runtime errors and ensure data integrity.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T", float, Decimal, np.ndarray, pd.Series)


class ValidationLevel(str, Enum):
    """Validation strictness levels."""

    STRICT = "strict"  # Raise exceptions on any validation failure
    MODERATE = "moderate"  # Log warnings and use defaults for recoverable issues
    LENIENT = "lenient"  # Only log debug messages, always use defaults


class ValidationResult(BaseModel):
    """Result of a validation operation."""

    is_valid: bool
    value: Any
    original_value: Any
    issues: list[str] = Field(default_factory=list)
    fixes_applied: list[str] = Field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        """Check if validation found any issues."""
        return len(self.issues) > 0

    @property
    def was_modified(self) -> bool:
        """Check if the value was modified during validation."""
        return self.value != self.original_value


@dataclass
class ValidationConfig:
    """Configuration for validation behavior."""

    level: ValidationLevel = ValidationLevel.MODERATE
    log_issues: bool = True
    raise_on_critical: bool = True
    default_on_nan: bool = True
    default_on_inf: bool = True
    epsilon: float = 1e-9  # Small value for float comparisons
    max_correction_attempts: int = 3


class MathValidator:
    """Safe mathematical operations with validation."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize math validator.

        Args:
            config: Validation configuration. Uses defaults if not provided.
        """
        self.config = config or ValidationConfig()

    def safe_divide(
        self, numerator: T, denominator: T, default: T | None = None, name: str = "division"
    ) -> T:
        """Perform safe division with zero checking.

        Args:
            numerator: The numerator value.
            denominator: The denominator value.
            default: Default value to return on division by zero.
            name: Name of the operation for logging.

        Returns:
            Result of division or default value.
        """
        # Handle scalar values
        if isinstance(numerator, int | float | Decimal):
            return self._safe_divide_scalar(numerator, denominator, default, name)

        # Handle numpy arrays
        if isinstance(numerator, np.ndarray):
            return self._safe_divide_array(numerator, denominator, default, name)

        # Handle pandas Series
        if isinstance(numerator, pd.Series):
            return self._safe_divide_series(numerator, denominator, default, name)

        # Fallback
        if self.config.log_issues:
            logger.warning(f"Unsupported type for safe_divide: {type(numerator)}")
        return default if default is not None else numerator

    def _safe_divide_scalar(
        self,
        num: float | Decimal,
        denom: float | Decimal,
        default: float | Decimal | None,
        name: str,
    ) -> float | Decimal:
        """Safe division for scalar values."""
        if default is None:
            default = 0.0 if isinstance(num, float) else Decimal("0")

        # Check for zero denominator
        if abs(float(denom)) < self.config.epsilon:
            if self.config.log_issues:
                logger.warning(
                    f"{name}: Division by zero avoided ({num}/{denom}), "
                    f"returning default: {default}"
                )
            return default

        # Check for inf/nan
        if isinstance(num, float):
            if np.isnan(num) or np.isnan(denom):
                if self.config.log_issues:
                    logger.warning(f"{name}: NaN detected, returning default: {default}")
                return default
            if np.isinf(num) or np.isinf(denom):
                if self.config.log_issues:
                    logger.warning(f"{name}: Inf detected, returning default: {default}")
                return default

        try:
            result = num / denom

            # Check result for inf/nan
            if isinstance(result, float) and (np.isnan(result) or np.isinf(result)):
                if self.config.log_issues:
                    logger.warning(f"{name}: Result is {result}, returning default: {default}")
                return default

            return result

        except (ZeroDivisionError, InvalidOperation) as e:
            if self.config.log_issues:
                logger.warning(f"{name}: Division error ({e}), returning default: {default}")
            return default

    def _safe_divide_array(
        self,
        num: np.ndarray,
        denom: np.ndarray | float,
        default: np.ndarray | float | None,
        name: str,
    ) -> np.ndarray:
        """Safe division for numpy arrays."""
        if default is None:
            default = 0.0

        # Create output array
        result = np.full_like(num, default, dtype=np.float64)

        # Handle scalar denominator
        if isinstance(denom, int | float):
            if abs(denom) < self.config.epsilon:
                if self.config.log_issues:
                    logger.warning(f"{name}: Scalar denominator near zero, returning defaults")
                return result
            valid_mask = ~(np.isnan(num) | np.isinf(num))
        else:
            # Array denominator
            valid_mask = (np.abs(denom) >= self.config.epsilon) & ~(
                np.isnan(num) | np.isinf(num) | np.isnan(denom) | np.isinf(denom)
            )

        # Perform division only where valid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result[valid_mask] = num[valid_mask] / (
                denom if isinstance(denom, int | float) else denom[valid_mask]
            )

        # Log if we had invalid values
        invalid_count = np.sum(~valid_mask)
        if invalid_count > 0 and self.config.log_issues:
            logger.debug(f"{name}: {invalid_count} invalid divisions, used default: {default}")

        return result

    def _safe_divide_series(
        self, num: pd.Series, denom: pd.Series | float, default: pd.Series | float | None, name: str
    ) -> pd.Series:
        """Safe division for pandas Series."""
        if default is None:
            default = 0.0

        # Convert to numpy, perform safe division, convert back
        num_array = num.values
        denom_array = denom.values if isinstance(denom, pd.Series) else denom

        result_array = self._safe_divide_array(num_array, denom_array, default, name)

        return pd.Series(result_array, index=num.index, name=num.name)

    def safe_log(self, value: T, default: T | None = None, name: str = "log") -> T:
        """Compute safe logarithm with validation.

        Args:
            value: Value to compute log of.
            default: Default value for invalid inputs.
            name: Name of the operation for logging.

        Returns:
            Logarithm or default value.
        """
        if default is None:
            default = 0.0 if isinstance(value, int | float) else np.zeros_like(value)

        # Scalar case
        if isinstance(value, int | float):
            if value <= 0 or np.isnan(value) or np.isinf(value):
                if self.config.log_issues:
                    logger.warning(f"{name}: Invalid value for log({value}), returning default")
                return default
            return np.log(value)

        # Array case
        if isinstance(value, np.ndarray | pd.Series):
            result = np.full_like(value, default, dtype=np.float64)
            valid_mask = (value > 0) & ~np.isnan(value) & ~np.isinf(value)

            if isinstance(value, pd.Series):
                result = pd.Series(result, index=value.index)
                result[valid_mask] = np.log(value[valid_mask])
            else:
                result[valid_mask] = np.log(value[valid_mask])

            invalid_count = np.sum(~valid_mask)
            if invalid_count > 0 and self.config.log_issues:
                logger.debug(f"{name}: {invalid_count} invalid log values, used default")

            return result

        return default

    def safe_sqrt(self, value: T, default: T | None = None, name: str = "sqrt") -> T:
        """Compute safe square root with validation.

        Args:
            value: Value to compute sqrt of.
            default: Default value for invalid inputs.
            name: Name of the operation for logging.

        Returns:
            Square root or default value.
        """
        if default is None:
            default = 0.0 if isinstance(value, int | float) else np.zeros_like(value)

        # Scalar case
        if isinstance(value, int | float):
            if value < 0 or np.isnan(value) or np.isinf(value):
                if self.config.log_issues:
                    logger.warning(f"{name}: Invalid value for sqrt({value}), returning default")
                return default
            return np.sqrt(value)

        # Array case
        if isinstance(value, np.ndarray | pd.Series):
            result = np.full_like(value, default, dtype=np.float64)
            valid_mask = (value >= 0) & ~np.isnan(value) & ~np.isinf(value)

            if isinstance(value, pd.Series):
                result = pd.Series(result, index=value.index)
                result[valid_mask] = np.sqrt(value[valid_mask])
            else:
                result[valid_mask] = np.sqrt(value[valid_mask])

            invalid_count = np.sum(~valid_mask)
            if invalid_count > 0 and self.config.log_issues:
                logger.debug(f"{name}: {invalid_count} invalid sqrt values, used default")

            return result

        return default


class FinancialValidator:
    """Validation for financial values and calculations."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize financial validator.

        Args:
            config: Validation configuration. Uses defaults if not provided.
        """
        self.config = config or ValidationConfig()
        self.math = MathValidator(config)

    def validate_price(
        self,
        price: float,
        symbol: str = "",
        min_price: float = 0.01,
        max_price: float = 1_000_000.0,
    ) -> ValidationResult:
        """Validate a price value.

        Args:
            price: Price to validate.
            symbol: Symbol for logging.
            min_price: Minimum valid price.
            max_price: Maximum valid price.

        Returns:
            ValidationResult with validated price.
        """
        issues = []
        fixes = []
        validated_price = price

        # Check for NaN
        if np.isnan(price):
            issues.append(f"Price is NaN for {symbol}")
            validated_price = 0.0
            fixes.append("Set to 0.0")

        # Check for infinity
        elif np.isinf(price):
            issues.append(f"Price is infinite for {symbol}")
            validated_price = 0.0
            fixes.append("Set to 0.0")

        # Check bounds
        elif price < min_price:
            issues.append(f"Price {price} below minimum {min_price} for {symbol}")
            if self.config.level == ValidationLevel.STRICT:
                validated_price = min_price
                fixes.append(f"Clamped to minimum {min_price}")
            else:
                validated_price = 0.0
                fixes.append("Set to 0.0")

        elif price > max_price:
            issues.append(f"Price {price} above maximum {max_price} for {symbol}")
            validated_price = max_price
            fixes.append(f"Clamped to maximum {max_price}")

        # Check for negative
        elif price < 0:
            issues.append(f"Negative price {price} for {symbol}")
            validated_price = abs(price)
            fixes.append("Converted to absolute value")

        return ValidationResult(
            is_valid=len(issues) == 0,
            value=validated_price,
            original_value=price,
            issues=issues,
            fixes_applied=fixes,
        )

    def validate_quantity(
        self,
        quantity: float,
        symbol: str = "",
        min_qty: float = 0.0,
        max_qty: float = 1_000_000.0,
        round_to: int | None = None,
    ) -> ValidationResult:
        """Validate a quantity/position size.

        Args:
            quantity: Quantity to validate.
            symbol: Symbol for logging.
            min_qty: Minimum valid quantity.
            max_qty: Maximum valid quantity.
            round_to: Decimal places to round to.

        Returns:
            ValidationResult with validated quantity.
        """
        issues = []
        fixes = []
        validated_qty = quantity

        # Check for NaN
        if np.isnan(quantity):
            issues.append(f"Quantity is NaN for {symbol}")
            validated_qty = 0.0
            fixes.append("Set to 0.0")

        # Check for infinity
        elif np.isinf(quantity):
            issues.append(f"Quantity is infinite for {symbol}")
            validated_qty = 0.0
            fixes.append("Set to 0.0")

        # Check bounds
        elif quantity < min_qty:
            issues.append(f"Quantity {quantity} below minimum {min_qty} for {symbol}")
            validated_qty = min_qty
            fixes.append(f"Clamped to minimum {min_qty}")

        elif quantity > max_qty:
            issues.append(f"Quantity {quantity} above maximum {max_qty} for {symbol}")
            validated_qty = max_qty
            fixes.append(f"Clamped to maximum {max_qty}")

        # Round if specified
        if round_to is not None and not np.isnan(validated_qty):
            rounded = round(validated_qty, round_to)
            if rounded != validated_qty:
                fixes.append(f"Rounded from {validated_qty} to {rounded}")
                validated_qty = rounded

        return ValidationResult(
            is_valid=len(issues) == 0,
            value=validated_qty,
            original_value=quantity,
            issues=issues,
            fixes_applied=fixes,
        )

    def validate_portfolio_weight(
        self, weight: float, symbol: str = "", min_weight: float = 0.0, max_weight: float = 1.0
    ) -> ValidationResult:
        """Validate a portfolio weight.

        Args:
            weight: Weight to validate (0-1).
            symbol: Symbol for logging.
            min_weight: Minimum valid weight.
            max_weight: Maximum valid weight.

        Returns:
            ValidationResult with validated weight.
        """
        issues = []
        fixes = []
        validated_weight = weight

        # Check for NaN
        if np.isnan(weight):
            issues.append(f"Weight is NaN for {symbol}")
            validated_weight = 0.0
            fixes.append("Set to 0.0")

        # Check bounds
        elif weight < min_weight:
            issues.append(f"Weight {weight} below minimum {min_weight} for {symbol}")
            validated_weight = min_weight
            fixes.append(f"Clamped to minimum {min_weight}")

        elif weight > max_weight:
            issues.append(f"Weight {weight} above maximum {max_weight} for {symbol}")
            validated_weight = max_weight
            fixes.append(f"Clamped to maximum {max_weight}")

        return ValidationResult(
            is_valid=len(issues) == 0,
            value=validated_weight,
            original_value=weight,
            issues=issues,
            fixes_applied=fixes,
        )

    def validate_return(
        self, return_value: float, period: str = "daily", max_return: float = 10.0  # 1000% return
    ) -> ValidationResult:
        """Validate a return value.

        Args:
            return_value: Return to validate.
            period: Time period for context.
            max_return: Maximum believable return.

        Returns:
            ValidationResult with validated return.
        """
        issues = []
        fixes = []
        validated_return = return_value

        # Check for NaN
        if np.isnan(return_value):
            issues.append(f"Return is NaN for {period}")
            validated_return = 0.0
            fixes.append("Set to 0.0")

        # Check for extreme values
        elif abs(return_value) > max_return:
            issues.append(f"Extreme {period} return: {return_value}")
            if return_value > 0:
                validated_return = max_return
                fixes.append(f"Clamped to maximum {max_return}")
            else:
                validated_return = -max_return
                fixes.append(f"Clamped to minimum {-max_return}")

        return ValidationResult(
            is_valid=len(issues) == 0,
            value=validated_return,
            original_value=return_value,
            issues=issues,
            fixes_applied=fixes,
        )


class DataValidator:
    """Validation for data integrity and quality."""

    def __init__(self, config: ValidationConfig | None = None) -> None:
        """Initialize data validator.

        Args:
            config: Validation configuration. Uses defaults if not provided.
        """
        self.config = config or ValidationConfig()
        self.financial = FinancialValidator(config)

    def validate_ohlcv(
        self, df: pd.DataFrame, symbol: str = "", repair: bool = True
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """Validate OHLCV data.

        Args:
            df: DataFrame with OHLC columns.
            symbol: Symbol for logging.
            repair: Whether to attempt repairs.

        Returns:
            Tuple of (validated DataFrame, ValidationResult).
        """
        issues = []
        fixes = []
        df_validated = df.copy()

        required_columns = ["Open", "High", "Low", "Close"]

        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
            return df, ValidationResult(
                is_valid=False, value=df, original_value=df, issues=issues, fixes_applied=fixes
            )

        # Check for NaN values
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            issues.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
            if repair:
                # Forward fill then backward fill
                df_validated[required_columns] = (
                    df_validated[required_columns].fillna(method="ffill").fillna(method="bfill")
                )
                fixes.append("Filled NaN values using forward/backward fill")

        # Check OHLC relationships
        invalid_high_low = df_validated["High"] < df_validated["Low"]
        if invalid_high_low.any():
            count = invalid_high_low.sum()
            issues.append(f"High < Low in {count} rows")
            if repair:
                # Swap high and low where invalid
                df_validated.loc[invalid_high_low, ["High", "Low"]] = df_validated.loc[
                    invalid_high_low, ["Low", "High"]
                ].values
                fixes.append(f"Swapped High/Low in {count} rows")

        # Check Close within High/Low range
        close_above_high = df_validated["Close"] > df_validated["High"]
        if close_above_high.any():
            count = close_above_high.sum()
            issues.append(f"Close > High in {count} rows")
            if repair:
                df_validated.loc[close_above_high, "High"] = df_validated.loc[
                    close_above_high, "Close"
                ]
                fixes.append(f"Adjusted High to match Close in {count} rows")

        close_below_low = df_validated["Close"] < df_validated["Low"]
        if close_below_low.any():
            count = close_below_low.sum()
            issues.append(f"Close < Low in {count} rows")
            if repair:
                df_validated.loc[close_below_low, "Low"] = df_validated.loc[
                    close_below_low, "Close"
                ]
                fixes.append(f"Adjusted Low to match Close in {count} rows")

        # Check for zero/negative prices
        for col in required_columns:
            zero_or_neg = df_validated[col] <= 0
            if zero_or_neg.any():
                count = zero_or_neg.sum()
                issues.append(f"Zero/negative values in {col}: {count} rows")
                if repair:
                    # Replace with previous valid value
                    df_validated.loc[zero_or_neg, col] = np.nan
                    df_validated[col] = (
                        df_validated[col].fillna(method="ffill").fillna(method="bfill")
                    )
                    fixes.append(f"Replaced zero/negative values in {col}")

        # Check for extreme price movements (>50% in one bar)
        if len(df_validated) > 1:
            returns = df_validated["Close"].pct_change()
            extreme_moves = returns.abs() > 0.5
            if extreme_moves.any():
                count = extreme_moves.sum()
                issues.append(f"Extreme price movements (>50%) in {count} bars")
                # Note: Not auto-fixing extreme moves as they might be legitimate

        # Volume validation if present
        if "Volume" in df.columns:
            neg_volume = df_validated["Volume"] < 0
            if neg_volume.any():
                count = neg_volume.sum()
                issues.append(f"Negative volume in {count} rows")
                if repair:
                    df_validated.loc[neg_volume, "Volume"] = 0
                    fixes.append(f"Set negative volume to 0 in {count} rows")

        return df_validated, ValidationResult(
            is_valid=len(issues) == 0,
            value=df_validated,
            original_value=df,
            issues=issues,
            fixes_applied=fixes,
        )

    def validate_returns_series(
        self, returns: pd.Series, max_return: float = 10.0, repair: bool = True
    ) -> tuple[pd.Series, ValidationResult]:
        """Validate a returns series.

        Args:
            returns: Series of returns.
            max_return: Maximum believable return.
            repair: Whether to attempt repairs.

        Returns:
            Tuple of (validated returns, ValidationResult).
        """
        issues = []
        fixes = []
        validated_returns = returns.copy()

        # Check for NaN
        nan_count = returns.isna().sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values")
            if repair:
                validated_returns = validated_returns.fillna(0)
                fixes.append(f"Filled {nan_count} NaN values with 0")

        # Check for infinite values
        inf_count = np.isinf(validated_returns).sum()
        if inf_count > 0:
            issues.append(f"Found {inf_count} infinite values")
            if repair:
                validated_returns[np.isinf(validated_returns)] = 0
                fixes.append(f"Replaced {inf_count} infinite values with 0")

        # Check for extreme returns
        extreme_returns = validated_returns.abs() > max_return
        if extreme_returns.any():
            count = extreme_returns.sum()
            issues.append(f"Found {count} extreme returns (>{max_return})")
            if repair:
                validated_returns[extreme_returns] = (
                    np.sign(validated_returns[extreme_returns]) * max_return
                )
                fixes.append(f"Clamped {count} extreme returns to ±{max_return}")

        return validated_returns, ValidationResult(
            is_valid=len(issues) == 0,
            value=validated_returns,
            original_value=returns,
            issues=issues,
            fixes_applied=fixes,
        )


# Singleton instances for convenience
_math_validator: MathValidator | None = None
_financial_validator: FinancialValidator | None = None
_data_validator: DataValidator | None = None


def get_math_validator(config: ValidationConfig | None = None) -> MathValidator:
    """Get singleton MathValidator instance.

    Args:
        config: Optional configuration for first initialization.

    Returns:
        MathValidator instance.
    """
    global _math_validator
    if _math_validator is None:
        _math_validator = MathValidator(config)
    return _math_validator


def get_financial_validator(config: ValidationConfig | None = None) -> FinancialValidator:
    """Get singleton FinancialValidator instance.

    Args:
        config: Optional configuration for first initialization.

    Returns:
        FinancialValidator instance.
    """
    global _financial_validator
    if _financial_validator is None:
        _financial_validator = FinancialValidator(config)
    return _financial_validator


def get_data_validator(config: ValidationConfig | None = None) -> DataValidator:
    """Get singleton DataValidator instance.

    Args:
        config: Optional configuration for first initialization.

    Returns:
        DataValidator instance.
    """
    global _data_validator
    if _data_validator is None:
        _data_validator = DataValidator(config)
    return _data_validator
