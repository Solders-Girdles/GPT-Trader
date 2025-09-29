"""
Security Module for Bot V2 Trading System

Provides comprehensive security features including:
- Secrets management with encryption
- JWT-based authentication and RBAC
- Input validation and injection prevention
- Rate limiting and suspicious activity detection
"""

from .auth_handler import (
    AuthHandler,
    Role,
    TokenPair,
    User,
    authenticate,
    check_permission,
    get_auth_handler,
    validate_token,
)
from .secrets_manager import (
    SecretsManager,
    get_api_key,
    get_secret,
    get_secrets_manager,
    store_secret,
)
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
    # Secrets Management
    "SecretsManager",
    "get_secrets_manager",
    "store_secret",
    "get_secret",
    "get_api_key",
    # Authentication
    "AuthHandler",
    "User",
    "Role",
    "TokenPair",
    "get_auth_handler",
    "authenticate",
    "validate_token",
    "check_permission",
    # Validation
    "SecurityValidator",
    "ValidationResult",
    "RateLimitConfig",
    "get_validator",
    "validate_order",
    "check_rate_limit",
    "sanitize_input",
]
