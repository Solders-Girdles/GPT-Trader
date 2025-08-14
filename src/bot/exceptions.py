"""
Centralized exception definitions for GPT-Trader.

This module provides a comprehensive set of custom exceptions for different
error scenarios in the trading system.
"""

from typing import Any


class GPTTraderError(Exception):
    """Base exception for all GPT-Trader errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(GPTTraderError):
    """Raised when there's a configuration issue."""

    pass


class DataError(GPTTraderError):
    """Raised when there's an issue with data sources or data quality."""

    pass


class StrategyError(GPTTraderError):
    """Raised when there's an issue with strategy execution."""

    pass


class OptimizationError(GPTTraderError):
    """Raised when there's an issue with optimization processes."""

    pass


class TradingError(GPTTraderError):
    """Raised when there's an issue with trading operations."""

    pass


class RiskError(GPTTraderError):
    """Raised when risk limits are exceeded or risk calculations fail."""

    pass


class PortfolioError(GPTTraderError):
    """Raised when there's an issue with portfolio management."""

    pass


class ValidationError(GPTTraderError):
    """Raised when data or parameters fail validation."""

    pass


class TimeoutError(GPTTraderError):
    """Raised when operations timeout."""

    pass


class InsufficientDataError(DataError):
    """Raised when there's insufficient data for analysis."""

    pass


class InvalidOHLCError(DataError):
    """Raised when OHLC data is invalid or corrupted."""

    pass


class StrategyNotFoundError(StrategyError):
    """Raised when a requested strategy is not found."""

    pass


class StrategyExecutionError(StrategyError):
    """Raised when strategy execution fails."""

    pass


class OptimizationTimeoutError(OptimizationError):
    """Raised when optimization times out."""

    pass


class OptimizationConvergenceError(OptimizationError):
    """Raised when optimization fails to converge."""

    pass


class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds for a trade."""

    pass


class OrderExecutionError(TradingError):
    """Raised when order execution fails."""

    pass


class RiskLimitExceededError(RiskError):
    """Raised when risk limits are exceeded."""

    pass


class PortfolioAllocationError(PortfolioError):
    """Raised when portfolio allocation fails."""

    pass


class InvalidParameterError(ValidationError):
    """Raised when parameters are invalid."""

    pass


class DataValidationError(ValidationError):
    """Raised when data validation fails."""

    pass


# Convenience functions for common error scenarios
def raise_if_invalid_data(data: Any, message: str) -> None:
    """Raise DataError if data is invalid."""
    if data is None or (hasattr(data, "__len__") and len(data) == 0):
        raise DataError(f"Invalid data: {message}")


def raise_if_missing_required_field(obj: Any, field_name: str) -> None:
    """Raise ValidationError if required field is missing."""
    if not hasattr(obj, field_name) or getattr(obj, field_name) is None:
        raise ValidationError(f"Missing required field: {field_name}")


def raise_if_out_of_range(value: float, min_val: float, max_val: float, field_name: str) -> None:
    """Raise ValidationError if value is out of range."""
    if not min_val <= value <= max_val:
        raise ValidationError(
            f"Value out of range for {field_name}: {value} (expected {min_val} to {max_val})"
        )
