"""Validation framework for GPT-Trader.

This module provides comprehensive validation and safe operations for:
- Mathematical operations (division, log, sqrt)
- Financial values (prices, quantities, returns)
- Data integrity (OHLCV validation)
"""

from .framework import (
    DataValidator,
    FinancialValidator,
    # Core classes
    MathValidator,
    ValidationConfig,
    ValidationLevel,
    ValidationResult,
    get_data_validator,
    get_financial_validator,
    # Singleton getters
    get_math_validator,
)

__all__ = [
    "MathValidator",
    "FinancialValidator",
    "DataValidator",
    "ValidationConfig",
    "ValidationResult",
    "ValidationLevel",
    "get_math_validator",
    "get_financial_validator",
    "get_data_validator",
]
