"""
Security Validator for Bot V2 Trading System

Implements input validation, rate limiting, and trading-specific security checks
to protect against injection attacks and ensure safe trading operations.

This module acts as a facade, delegating to specialized security modules.
"""

from datetime import datetime
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

from .input_sanitizer import InputSanitizer, ValidationResult
from .numeric_validator import NumericValidator
from .order_validator import OrderValidator
from .rate_limiter import RateLimitConfig, RateLimiter
from .request_validator import RequestValidator
from .suspicious_activity_detector import SuspiciousActivityDetector
from .symbol_validator import SymbolValidator
from .trading_hours_validator import TradingHoursValidator

logger = get_logger(__name__, component="security")


class SecurityValidator:
    """
    Comprehensive security validation for trading operations.
    Prevents injection attacks and enforces trading limits.
    """

    # Regex patterns for validation (maintained for backward compatibility)
    PATTERNS = {
        "symbol": r"^[A-Z0-9]{1,10}(-[A-Z0-9]{2,10})?$",  # Trading symbols
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "alphanumeric": r"^[a-zA-Z0-9]+$",
        "numeric": r"^-?\d+(\.\d+)?$",
        "sql_injection": r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER|EXEC|EXECUTE|EVAL|OR)\b)|((?<![</])\b(SCRIPT|JAVASCRIPT)\b)",
        "xss_tags": r"<[^>]*>",
        "path_traversal": r"(\.\.[\\/])|(%2[fF])|(%5[cC])|(%2[eE]%2[eE])|(file://)",
    }

    # Trading limits (maintained for backward compatibility)
    TRADING_LIMITS = OrderValidator.TRADING_LIMITS

    # Rate limit configurations (maintained for backward compatibility)
    RATE_LIMITS = RateLimiter.RATE_LIMITS

    def __init__(self) -> None:
        self._rate_limiter = RateLimiter()

    def sanitize_string(self, input_str: str, max_length: int = 255) -> ValidationResult:
        """Sanitize string input and check for injection attempts."""
        return InputSanitizer.sanitize_string(input_str, max_length)

    def validate_symbol(self, symbol: str) -> ValidationResult:
        """Validate trading symbol"""
        return SymbolValidator.validate_symbol(symbol)

    def validate_numeric(
        self, value: Any, min_val: float | None = None, max_val: float | None = None
    ) -> ValidationResult:
        """Validate numeric input"""
        return NumericValidator.validate_numeric(value, min_val, max_val)

    def validate_order_request(
        self, order: dict[str, Any], account_value: float
    ) -> ValidationResult:
        """Validate trading order request."""
        return OrderValidator.validate_order_request(order, account_value)

    def check_rate_limit(self, identifier: str, limit_type: str) -> tuple[bool, str | None]:
        """Check if request is within rate limits."""
        return self._rate_limiter.check_rate_limit(identifier, limit_type)

    def check_trading_hours(
        self, symbol: str, timestamp: datetime | None = None
    ) -> ValidationResult:
        """Check if trading is allowed at current time"""
        return TradingHoursValidator.check_trading_hours(symbol, timestamp)

    def detect_suspicious_activity(self, user_id: str, activity: dict[str, Any]) -> bool:
        """Detect potentially suspicious trading activity."""
        return SuspiciousActivityDetector.detect_suspicious_activity(user_id, activity)

    def validate_request(self, request: dict[str, Any]) -> ValidationResult:
        """Comprehensive request validation."""
        return RequestValidator.validate_request(request)

    def clear_rate_limits(self, identifier: str | None = None) -> None:
        """Clear rate limit history"""
        self._rate_limiter.clear_rate_limits(identifier)


# Global instance
_validator = None


def get_validator() -> SecurityValidator:
    """Get the global validator instance"""
    global _validator
    if _validator is None:
        _validator = SecurityValidator()
    return _validator


# Convenience functions
def validate_order(order: dict, account_value: float) -> ValidationResult:
    """Validate trading order"""
    return get_validator().validate_order_request(order, account_value)


def check_rate_limit(identifier: str, limit_type: str) -> tuple[bool, str | None]:
    """Check rate limit"""
    return get_validator().check_rate_limit(identifier, limit_type)


def sanitize_input(input_str: str, max_length: int = 255) -> ValidationResult:
    """Sanitize string input"""
    return get_validator().sanitize_string(input_str, max_length)


# Re-export for backward compatibility
__all__ = [
    "SecurityValidator",
    "ValidationResult",
    "RateLimitConfig",
    "get_validator",
    "validate_order",
    "check_rate_limit",
    "sanitize_input",
]
