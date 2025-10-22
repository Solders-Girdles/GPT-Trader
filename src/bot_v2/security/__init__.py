"""
Security Module for Bot V2 Trading System

Provides comprehensive security features including:
- Secrets management with encryption (HashiCorp Vault/1Password support)
- CDP secrets provider with short-lived JWT generation
- IP allowlisting enforcement for API keys
- JWT-based authentication and RBAC
- Input validation and injection prevention
- Rate limiting and suspicious activity detection
"""

from bot_v2.security.auth_handler import (
    AuthHandler,
    Role,
    TokenPair,
    User,
    authenticate,
    check_permission,
    get_auth_handler,
    validate_token,
)
from bot_v2.security.cdp_secrets_provider import (
    CDPCredentials,
    CDPJWTToken,
    CDPSecretsProvider,
    generate_cdp_jwt,
    get_cdp_secrets_provider,
    store_cdp_credentials,
)
from bot_v2.security.ip_allowlist_enforcer import (
    IPAllowlistEnforcer,
    IPAllowlistRule,
    IPValidationResult,
    add_ip_allowlist_rule,
    get_ip_allowlist_enforcer,
    validate_ip,
)
from bot_v2.security.secrets_manager import (
    SecretsManager,
    get_api_key,
    get_secret,
    get_secrets_manager,
    store_secret,
)
from bot_v2.security.security_validator import (
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
    # CDP Secrets Provider
    "CDPSecretsProvider",
    "CDPCredentials",
    "CDPJWTToken",
    "get_cdp_secrets_provider",
    "store_cdp_credentials",
    "generate_cdp_jwt",
    # IP Allowlisting
    "IPAllowlistEnforcer",
    "IPAllowlistRule",
    "IPValidationResult",
    "get_ip_allowlist_enforcer",
    "add_ip_allowlist_rule",
    "validate_ip",
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
