"""Enhanced exception handling for GPT-Trader.

This module provides:
- Structured exception hierarchy
- Automatic recovery mechanisms
- Retry logic with exponential backoff
- Circuit breaker patterns
- Exception decorators
"""

# Import from enhanced_exceptions
# Import decorators
from .decorators import (
    handle_exceptions,
    monitor_performance,
    safe_execution,
    validate_inputs,
    with_circuit_breaker,
    with_recovery,
    with_retry,
)
from .enhanced_exceptions import (
    CircuitBreaker,
    CriticalError,
    # Specific exceptions
    DataIntegrityError,
    # Context and enums
    ErrorContext,
    ErrorSeverity,
    # Handler and circuit breaker
    ExceptionHandler,
    # Base exceptions
    GPTTraderException,
    InsufficientCapitalError,
    NetworkError,
    OrderRejectedError,
    RecoverableError,
    RecoveryStrategy,
    RetryableError,
    RiskLimitError,
    TradingException,
    get_exception_handler,
)

# Legacy exception names for backward compatibility
# These map to the base GPTTraderException for now
ConfigurationError = GPTTraderException
DataError = DataIntegrityError
DataValidationError = DataIntegrityError
GPTTraderError = GPTTraderException
InsufficientDataError = DataIntegrityError
InsufficientFundsError = InsufficientCapitalError
InvalidOHLCError = DataIntegrityError
InvalidParameterError = GPTTraderException
OptimizationConvergenceError = GPTTraderException
OptimizationError = GPTTraderException
OptimizationTimeoutError = GPTTraderException
OrderExecutionError = OrderRejectedError
PortfolioAllocationError = TradingException
PortfolioError = TradingException
RiskError = RiskLimitError
RiskLimitExceededError = RiskLimitError
StrategyError = GPTTraderException
StrategyExecutionError = GPTTraderException
StrategyNotFoundError = GPTTraderException
TimeoutError = RetryableError
TradingError = TradingException
ValidationError = DataIntegrityError

__all__ = [
    # Base exceptions
    "GPTTraderException",
    "RecoverableError",
    "CriticalError",
    "RetryableError",
    # Specific exceptions
    "DataIntegrityError",
    "NetworkError",
    "TradingException",
    "InsufficientCapitalError",
    "OrderRejectedError",
    "RiskLimitError",
    # Context and enums
    "ErrorContext",
    "ErrorSeverity",
    "RecoveryStrategy",
    # Handler and circuit breaker
    "ExceptionHandler",
    "CircuitBreaker",
    "get_exception_handler",
    # Decorators
    "handle_exceptions",
    "monitor_performance",
    "safe_execution",
    "validate_inputs",
    "with_circuit_breaker",
    "with_recovery",
    "with_retry",
    # Legacy names
    "ConfigurationError",
    "DataError",
    "DataValidationError",
    "GPTTraderError",
    "InsufficientDataError",
    "InsufficientFundsError",
    "InvalidOHLCError",
    "InvalidParameterError",
    "OptimizationConvergenceError",
    "OptimizationError",
    "OptimizationTimeoutError",
    "OrderExecutionError",
    "PortfolioAllocationError",
    "PortfolioError",
    "RiskError",
    "RiskLimitExceededError",
    "StrategyError",
    "StrategyExecutionError",
    "StrategyNotFoundError",
    "TimeoutError",
    "TradingError",
    "ValidationError",
]
