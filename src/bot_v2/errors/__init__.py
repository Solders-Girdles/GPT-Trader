"""
Centralized error handling for GPT-Trader V2

This module provides a comprehensive error hierarchy and handling system
for all feature slices, ensuring consistent error management across the system.
"""

import logging
import traceback
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class TradingError(Exception):
    """Base exception class for all trading-related errors"""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()

    def add_context(self, **kwargs):
        """Add additional context to the error"""
        self.context.update(kwargs)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback,
        }


class DataError(TradingError):
    """Raised when there are issues with market data"""

    def __init__(self, message: str, symbol: str | None = None, **kwargs) -> None:
        super().__init__(message, error_code="DATA_ERROR", **kwargs)
        if symbol:
            self.add_context(symbol=symbol)


class ConfigurationError(TradingError):
    """Raised when there are configuration issues"""

    def __init__(self, message: str, config_key: str | None = None, **kwargs) -> None:
        super().__init__(message, error_code="CONFIG_ERROR", recoverable=False, **kwargs)
        if config_key:
            self.add_context(config_key=config_key)


class ValidationError(TradingError):
    """Raised when input validation fails"""

    def __init__(self, message: str, field: str | None = None, value: Any = None, **kwargs) -> None:
        super().__init__(message, error_code="VALIDATION_ERROR", recoverable=False, **kwargs)
        if field:
            self.add_context(field=field, value=value)


class ExecutionError(TradingError):
    """Raised when trade execution fails"""

    def __init__(self, message: str, order_id: str | None = None, **kwargs) -> None:
        super().__init__(message, error_code="EXECUTION_ERROR", **kwargs)
        if order_id:
            self.add_context(order_id=order_id)


class NetworkError(TradingError):
    """Raised when network/API calls fail"""

    def __init__(
        self, message: str, url: str | None = None, status_code: int | None = None, **kwargs
    ) -> None:
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        if url:
            self.add_context(url=url, status_code=status_code)


class InsufficientFundsError(TradingError):
    """Raised when there are insufficient funds for a trade"""

    def __init__(self, message: str, required: float, available: float, **kwargs) -> None:
        super().__init__(message, error_code="INSUFFICIENT_FUNDS", recoverable=False, **kwargs)
        self.add_context(required=required, available=available, shortfall=required - available)


class StrategyError(TradingError):
    """Raised when strategy execution fails"""

    def __init__(self, message: str, strategy_name: str | None = None, **kwargs) -> None:
        super().__init__(message, error_code="STRATEGY_ERROR", **kwargs)
        if strategy_name:
            self.add_context(strategy_name=strategy_name)


class BacktestError(TradingError):
    """Raised when backtesting fails"""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_code="BACKTEST_ERROR", **kwargs)


class OptimizationError(TradingError):
    """Raised when optimization fails"""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_code="OPTIMIZATION_ERROR", **kwargs)


class RiskLimitExceeded(TradingError):
    """Raised when risk limits are exceeded"""

    def __init__(
        self, message: str, limit_type: str, limit_value: float, current_value: float, **kwargs
    ) -> None:
        super().__init__(message, error_code="RISK_LIMIT_EXCEEDED", recoverable=False, **kwargs)
        self.add_context(
            limit_type=limit_type,
            limit_value=limit_value,
            current_value=current_value,
            exceeded_by=current_value - limit_value,
        )


class TimeoutError(TradingError):
    """Raised when an operation times out"""

    def __init__(self, message: str, operation: str, timeout_seconds: float, **kwargs) -> None:
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.add_context(operation=operation, timeout_seconds=timeout_seconds)


class SliceIsolationError(TradingError):
    """Raised when slice isolation is violated"""

    def __init__(self, message: str, slice_name: str, violation: str, **kwargs) -> None:
        super().__init__(message, error_code="SLICE_ISOLATION_ERROR", recoverable=False, **kwargs)
        self.add_context(slice_name=slice_name, violation=violation)


# Error aggregation for multiple errors
class AggregateError(TradingError):
    """Container for multiple errors"""

    def __init__(self, message: str, errors: list[TradingError], **kwargs) -> None:
        super().__init__(message, error_code="AGGREGATE_ERROR", **kwargs)
        self.errors = errors
        self.add_context(error_count=len(errors))

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["errors"] = [e.to_dict() for e in self.errors]
        return result


# Helper functions
def handle_error(error: Exception, context: dict[str, Any] | None = None) -> TradingError:
    """Convert any exception to a TradingError with context"""
    if isinstance(error, TradingError):
        if context:
            error.add_context(**context)
        return error

    # Wrap non-trading errors
    wrapped = TradingError(
        message=str(error), error_code=error.__class__.__name__, context=context or {}
    )
    wrapped.traceback = traceback.format_exc()
    return wrapped


def log_error(error: TradingError, level: int = logging.ERROR) -> None:
    """Log an error with full context"""
    logger.log(level, f"{error.error_code}: {error.message}", extra={"error_data": error.to_dict()})


# Export all error types
__all__ = [
    "TradingError",
    "DataError",
    "ConfigurationError",
    "ValidationError",
    "ExecutionError",
    "NetworkError",
    "InsufficientFundsError",
    "StrategyError",
    "BacktestError",
    "OptimizationError",
    "RiskLimitExceeded",
    "TimeoutError",
    "SliceIsolationError",
    "AggregateError",
    "handle_error",
    "log_error",
]
