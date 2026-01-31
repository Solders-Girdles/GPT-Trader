"""Public entrypoint for security validation.

Use this module for security validation outside the security package.
"""

from __future__ import annotations

from .security_validator import (
    RateLimitConfig,
    SecurityValidator,
    ValidationResult,
    check_rate_limit,
    get_validator,
    sanitize_input,
    validate_order,
)

__all__ = [
    "RateLimitConfig",
    "SecurityValidator",
    "ValidationResult",
    "check_rate_limit",
    "get_validator",
    "sanitize_input",
    "validate_order",
]
