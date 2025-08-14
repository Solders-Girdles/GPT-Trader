"""
Centralized Security Configuration Module

Provides security settings, policies, and constants for the GPT-Trader application.
All security-related configuration should be centralized here.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class SecurityLevel(Enum):
    """Application security levels"""

    DEVELOPMENT = "development"  # Relaxed security for development
    STAGING = "staging"  # Moderate security for testing
    PRODUCTION = "production"  # Maximum security for production


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""

    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    FERNET = "fernet"  # Symmetric encryption


class HashAlgorithm(Enum):
    """Supported hash algorithms"""

    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    ARGON2 = "argon2"  # For password hashing


@dataclass
class SecurityConfig:
    """Main security configuration"""

    # Environment
    security_level: SecurityLevel = SecurityLevel.PRODUCTION
    debug_mode: bool = False

    # Authentication
    jwt_secret_key: str = field(default_factory=lambda: os.environ.get("JWT_SECRET_KEY", ""))
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    jwt_refresh_expiry_days: int = 30

    # Password Policy
    min_password_length: int = 16
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_history_count: int = 5  # Remember last N passwords
    password_expiry_days: int = 90
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30

    # Session Management
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 3
    session_cookie_secure: bool = True
    session_cookie_httponly: bool = True
    session_cookie_samesite: str = "strict"

    # API Security
    api_rate_limit_per_minute: int = 60
    api_rate_limit_per_hour: int = 1000
    api_key_min_length: int = 32
    api_key_max_length: int = 128
    require_api_key: bool = True
    api_key_header: str = "X-API-Key"

    # CORS Settings
    cors_allowed_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    cors_allowed_methods: list[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    cors_allowed_headers: list[str] = field(
        default_factory=lambda: ["Content-Type", "Authorization", "X-API-Key"]
    )
    cors_max_age: int = 3600

    # Encryption
    encryption_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    encryption_key: str = field(default_factory=lambda: os.environ.get("ENCRYPTION_KEY", ""))
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    password_hash_algorithm: HashAlgorithm = HashAlgorithm.ARGON2

    # File Security
    allowed_upload_extensions: list[str] = field(
        default_factory=lambda: [".csv", ".json", ".txt", ".log"]
    )
    max_upload_size_mb: int = 10
    scan_uploads_for_malware: bool = True
    quarantine_suspicious_files: bool = True

    # Network Security
    allowed_ips: list[str] | None = None  # None = allow all
    blocked_ips: list[str] = field(default_factory=list)
    require_https: bool = True
    ssl_verify: bool = True
    proxy_trusted_ips: list[str] = field(
        default_factory=lambda: ["127.0.0.1", "::1"]  # Localhost only by default
    )

    # Audit & Logging
    log_authentication_events: bool = True
    log_authorization_events: bool = True
    log_data_access: bool = True
    log_configuration_changes: bool = True
    log_security_events: bool = True
    audit_log_retention_days: int = 90

    # Data Protection
    encrypt_data_at_rest: bool = True
    encrypt_data_in_transit: bool = True
    data_retention_days: int = 365
    anonymize_old_data: bool = True
    right_to_be_forgotten: bool = True  # GDPR compliance

    # Secrets Management
    secrets_rotation_days: int = 90
    require_secret_rotation: bool = True
    min_secret_length: int = 32
    secret_complexity_score: int = 4  # 1-5 scale

    # Security Headers
    enable_security_headers: bool = True
    hsts_max_age: int = 31536000  # 1 year
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    x_xss_protection: str = "1; mode=block"
    referrer_policy: str = "strict-origin-when-cross-origin"
    content_security_policy: str = "default-src 'self'"

    # Rate Limiting
    enable_rate_limiting: bool = True
    rate_limit_storage: str = "memory"  # memory, redis, database
    rate_limit_strategy: str = "fixed-window"  # fixed-window, sliding-window

    # Monitoring & Alerting
    enable_security_monitoring: bool = True
    alert_on_suspicious_activity: bool = True
    alert_channels: list[str] = field(default_factory=lambda: ["email", "slack"])
    security_email: str = field(default_factory=lambda: os.environ.get("SECURITY_EMAIL", ""))

    def __post_init__(self):
        """Validate and adjust configuration based on security level"""
        if self.security_level == SecurityLevel.DEVELOPMENT:
            self.debug_mode = True
            self.require_https = False
            self.ssl_verify = False
            self.api_rate_limit_per_minute = 1000
            self.max_login_attempts = 10
            self.session_timeout_minutes = 120

        elif self.security_level == SecurityLevel.STAGING:
            self.debug_mode = False
            self.api_rate_limit_per_minute = 120
            self.session_timeout_minutes = 60

        elif self.security_level == SecurityLevel.PRODUCTION:
            self.debug_mode = False
            self.require_https = True
            self.ssl_verify = True
            self.encrypt_data_at_rest = True
            self.encrypt_data_in_transit = True

        # Validate required secrets
        if self.security_level == SecurityLevel.PRODUCTION:
            if not self.jwt_secret_key:
                raise ValueError("JWT_SECRET_KEY environment variable is required in production")
            if len(self.jwt_secret_key) < 32:
                raise ValueError("JWT_SECRET_KEY must be at least 32 characters in production")


class SecurityPolicy(BaseModel):
    """Security policy rules"""

    # Access Control
    require_mfa: bool = Field(False, description="Require multi-factor authentication")
    mfa_methods: list[str] = Field(default=["totp", "sms"], description="Allowed MFA methods")

    # IP Restrictions
    whitelist_mode: bool = Field(False, description="Only allow whitelisted IPs")
    blacklist_mode: bool = Field(True, description="Block blacklisted IPs")
    geo_blocking_enabled: bool = Field(False, description="Enable geographic blocking")
    blocked_countries: list[str] = Field(default=[], description="ISO country codes to block")

    # Trading Restrictions
    max_daily_trades: int = Field(100, description="Maximum trades per day")
    max_position_size: float = Field(100000, description="Maximum position size in USD")
    max_portfolio_risk: float = Field(0.02, description="Maximum portfolio risk (2%)")
    require_stop_loss: bool = Field(True, description="Require stop loss on all trades")

    # Data Access
    pii_access_requires_approval: bool = Field(True, description="PII access needs approval")
    export_requires_approval: bool = Field(True, description="Data export needs approval")
    api_access_requires_approval: bool = Field(False, description="API access needs approval")

    # Compliance
    gdpr_compliant: bool = Field(True, description="GDPR compliance mode")
    ccpa_compliant: bool = Field(True, description="CCPA compliance mode")
    pci_compliant: bool = Field(False, description="PCI-DSS compliance mode")
    sox_compliant: bool = Field(False, description="SOX compliance mode")


class SecurityHeaders:
    """Security headers for HTTP responses"""

    @staticmethod
    def get_headers(config: SecurityConfig) -> dict[str, str]:
        """Get security headers based on configuration"""
        headers = {}

        if config.enable_security_headers:
            headers.update(
                {
                    "Strict-Transport-Security": f"max-age={config.hsts_max_age}; includeSubDomains",
                    "X-Frame-Options": config.x_frame_options,
                    "X-Content-Type-Options": config.x_content_type_options,
                    "X-XSS-Protection": config.x_xss_protection,
                    "Referrer-Policy": config.referrer_policy,
                    "Content-Security-Policy": config.content_security_policy,
                    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
                }
            )

        return headers


class RateLimitConfig:
    """Rate limiting configuration"""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.limits = {
            "global": f"{config.api_rate_limit_per_hour}/hour",
            "per_minute": f"{config.api_rate_limit_per_minute}/minute",
            "auth": "5/minute",  # Authentication attempts
            "data": "100/minute",  # Data API calls
            "trading": "10/minute",  # Trading operations
            "backtest": "5/minute",  # Backtesting requests
        }

    def get_limit(self, endpoint_type: str) -> str:
        """Get rate limit for endpoint type"""
        return self.limits.get(endpoint_type, self.limits["global"])


# Singleton instance
_security_config: SecurityConfig | None = None


def get_security_config() -> SecurityConfig:
    """Get or create security configuration"""
    global _security_config

    if _security_config is None:
        # Determine security level from environment
        env_level = os.environ.get("SECURITY_LEVEL", "production").lower()
        try:
            security_level = SecurityLevel(env_level)
        except ValueError:
            security_level = SecurityLevel.PRODUCTION

        _security_config = SecurityConfig(security_level=security_level)

    return _security_config


def reset_security_config():
    """Reset security configuration (mainly for testing)"""
    global _security_config
    _security_config = None


# Security constants
PBKDF2_ITERATIONS = 100000  # For key derivation
ARGON2_TIME_COST = 2  # Argon2 time cost
ARGON2_MEMORY_COST = 65536  # Argon2 memory cost (64MB)
ARGON2_PARALLELISM = 4  # Argon2 parallelism
BCRYPT_ROUNDS = 12  # BCrypt work factor

# Security paths
SECURITY_LOG_DIR = Path("/var/log/gpt-trader/security")
AUDIT_LOG_FILE = SECURITY_LOG_DIR / "audit.log"
AUTH_LOG_FILE = SECURITY_LOG_DIR / "auth.log"
ACCESS_LOG_FILE = SECURITY_LOG_DIR / "access.log"

# Create directories if they don't exist
SECURITY_LOG_DIR.mkdir(parents=True, exist_ok=True)


# Export commonly used functions
def is_production() -> bool:
    """Check if running in production mode"""
    config = get_security_config()
    return config.security_level == SecurityLevel.PRODUCTION


def is_development() -> bool:
    """Check if running in development mode"""
    config = get_security_config()
    return config.security_level == SecurityLevel.DEVELOPMENT


def requires_https() -> bool:
    """Check if HTTPS is required"""
    config = get_security_config()
    return config.require_https


def get_jwt_secret() -> str:
    """Get JWT secret key"""
    config = get_security_config()
    if not config.jwt_secret_key:
        if is_production():
            raise ValueError("JWT secret not configured")
        else:
            # Development fallback
            return "development-secret-change-in-production"
    return config.jwt_secret_key
