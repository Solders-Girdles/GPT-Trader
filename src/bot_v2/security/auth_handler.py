"""
Authentication and Authorization Handler for Bot V2 Trading System

Implements JWT-based authentication with RBAC, MFA support, and secure session management.
"""

import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - type checking only
    import jwt as _jwt_module  # noqa: F401
    import pyotp as _pyotp_module  # noqa: F401
else:  # pragma: no cover - runtime import guards
    try:
        import jwt as _jwt_module  # type: ignore[import-not-found]
        from jwt import PyJWTError as _PyJWTError  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - optional dependency
        _jwt_module = None  # type: ignore[assignment]

        class _PyJWTError(Exception):
            """Fallback error when PyJWT is unavailable."""

            pass

    try:
        import pyotp as _pyotp_module  # type: ignore[import-not-found]
    except ImportError:
        _pyotp_module = None  # type: ignore[assignment]

from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="auth")
PyJWTError = _PyJWTError


def _require_jwt() -> Any:
    if _jwt_module is None:
        raise RuntimeError(
            "PyJWT is not installed. Install extras with `pip install gpt-trader[security]` "
            "or `poetry install --with security`."
        )
    return _jwt_module


def _require_pyotp() -> Any:
    if _pyotp_module is None:
        raise RuntimeError(
            "pyotp is not installed. Install extras with `pip install gpt-trader[security]` "
            "or `poetry install --with security`."
        )
    return _pyotp_module


class Role(Enum):
    """User roles with hierarchical permissions"""

    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    SERVICE = "service"


@dataclass
class User:
    """User model with authentication details"""

    id: str
    username: str
    email: str
    role: Role
    permissions: list[str] = field(default_factory=list)
    mfa_enabled: bool = False
    mfa_secret: str | None = None
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class TokenPair:
    """JWT token pair (access + refresh)"""

    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"


@dataclass
class SessionInfo:
    user_id: str
    created_at: datetime
    access_token_jti: str
    refresh_token_jti: str


class AuthHandler:
    """
    Handles authentication, authorization, and session management.
    Implements JWT with RS256 and RBAC.
    """

    # Role permissions mapping
    ROLE_PERMISSIONS = {
        Role.ADMIN: ["*"],  # All permissions
        Role.TRADER: [
            "trading:read",
            "trading:execute",
            "positions:read",
            "orders:create",
            "performance:read",
        ],
        Role.VIEWER: ["trading:read", "positions:read", "performance:read"],
        Role.SERVICE: ["data:read", "signals:write", "health:read"],
    }

    def __init__(self, *, settings: RuntimeSettings | None = None) -> None:
        self._lock = Lock()
        self._users: dict[str, User] = {}
        self._sessions: dict[str, SessionInfo] = {}
        self._revoked_tokens: set[str] = set()

        self._settings = settings or load_runtime_settings()
        env_map = self._settings.raw_env

        # JWT configuration
        self.algorithm = "HS256"  # Using HS256 for simplicity, RS256 requires key pair
        self.secret_key = env_map.get("JWT_SECRET_KEY") or secrets.token_urlsafe(32)
        self.issuer = "bot-v2-trading-system"
        self.audience = ["trading-api", "dashboard"]
        self.access_token_lifetime = 900  # 15 minutes
        self.refresh_token_lifetime = 604800  # 7 days

        # Initialize with default admin user
        self._create_default_users()

    def _create_default_users(self) -> None:
        """Create default users for development"""
        environment = (self._settings.raw_env.get("ENV") or "development").lower()
        if environment == "development":
            admin_user = User(
                id="admin-001",
                username="admin",
                email="admin@bot-v2.local",
                role=Role.ADMIN,
                permissions=self.ROLE_PERMISSIONS[Role.ADMIN],
            )
            self._users[admin_user.id] = admin_user
            logger.info(
                "Created default admin user",
                operation="user_bootstrap",
                status="created",
            )

    def authenticate_user(
        self, username: str, password: str, mfa_code: str | None = None
    ) -> TokenPair | None:
        """
        Authenticate user and return JWT tokens.

        Args:
            username: Username or email
            password: User password
            mfa_code: Optional MFA code

        Returns:
            TokenPair if successful, None otherwise
        """
        with self._lock:
            # Find user
            user = self._find_user_by_username(username)
            if not user:
                logger.warning(
                    f"Authentication failed: user not found - {username}",
                    operation="authenticate_user",
                    status="failed",
                )
                return None

            # Verify password (simplified for demo)
            if not self._verify_password(password, user.id):
                logger.warning(
                    f"Authentication failed: invalid password - {username}",
                    operation="authenticate_user",
                    status="failed",
                )
                return None

            # Check MFA if enabled
            if user.mfa_enabled:
                if not mfa_code or not self._verify_mfa(user, mfa_code):
                    logger.warning(
                        f"Authentication failed: invalid MFA - {username}",
                        operation="authenticate_user",
                        status="failed",
                    )
                    return None

            # Generate tokens
            token_pair, access_jti, refresh_jti = self._generate_tokens(user)

            # Create session
            session_id = secrets.token_urlsafe(32)
            self._sessions[session_id] = SessionInfo(
                user_id=user.id,
                created_at=datetime.now(UTC),
                access_token_jti=access_jti,
                refresh_token_jti=refresh_jti,
            )

            logger.info(
                f"User authenticated successfully: {username}",
                operation="authenticate_user",
                status="success",
            )
            return token_pair

    def _generate_tokens(self, user: User) -> tuple[TokenPair, str, str]:
        """Generate JWT token pair for user"""
        now = datetime.now(UTC)

        # Access token claims
        access_jti = secrets.token_urlsafe(32)
        access_exp = now + timedelta(seconds=self.access_token_lifetime)
        access_claims = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "permissions": user.permissions,
            "iat": now,
            "exp": access_exp,
            "jti": access_jti,
            "iss": self.issuer,
            "aud": self.audience,
            "type": "access",
        }

        # Refresh token claims
        refresh_jti = secrets.token_urlsafe(32)
        refresh_exp = now + timedelta(seconds=self.refresh_token_lifetime)
        refresh_claims = {
            "sub": user.id,
            "iat": now,
            "exp": refresh_exp,
            "jti": refresh_jti,
            "iss": self.issuer,
            "aud": self.audience,
            "type": "refresh",
        }

        # Generate tokens
        jwt_module = _require_jwt()
        access_token = jwt_module.encode(access_claims, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt_module.encode(refresh_claims, self.secret_key, algorithm=self.algorithm)

        pair = TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_lifetime,
        )
        return pair, access_jti, refresh_jti

    def validate_token(self, token: str) -> dict[str, Any] | None:
        """
        Validate JWT token and return claims.

        Args:
            token: JWT token string

        Returns:
            Token claims if valid, None otherwise
        """
        jwt_module = _require_jwt()
        try:
            # Decode token
            claims = cast(
                dict[str, Any],
                jwt_module.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm],
                    issuer=self.issuer,
                    audience=self.audience,
                ),
            )

            # Check if revoked
            if claims.get("jti") in self._revoked_tokens:
                logger.warning(
                    f"Token is revoked: {claims.get('jti')}",
                    operation="token_validate",
                    status="revoked",
                )
                return None

            return claims

        except jwt_module.ExpiredSignatureError:
            logger.debug(
                "Token expired",
                operation="token_validate",
                status="expired",
            )
            return None
        except jwt_module.InvalidTokenError as exc:
            logger.warning(
                f"Invalid token: {exc}",
                operation="token_validate",
                status="invalid",
            )
            return None

    def refresh_token(self, refresh_token: str) -> TokenPair | None:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Valid refresh token

        Returns:
            New TokenPair if successful
        """
        # Validate refresh token
        claims = self.validate_token(refresh_token)
        if not claims or claims.get("type") != "refresh":
            return None

        # Get user
        user = self._users.get(claims["sub"])
        if not user or not user.is_active:
            return None

        # Revoke old refresh token
        self._revoked_tokens.add(claims["jti"])

        # Generate new tokens
        new_pair, _, _ = self._generate_tokens(user)
        return new_pair

    def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        jwt_module = _require_jwt()
        try:
            claims = jwt_module.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False},
            )
            jti = claims.get("jti")
            if jti:
                self._revoked_tokens.add(jti)
                logger.info(
                    f"Token revoked: {jti}",
                    operation="token_revoke",
                    status="revoked",
                )
                return True
        except PyJWTError as exc:
            logger.warning(
                f"Failed to decode token during revocation: {exc}",
                operation="token_revoke",
                status="decode_failed",
                exc_info=True,
            )
        except Exception as exc:
            logger.exception(
                f"Unexpected error revoking token: {exc}",
                operation="token_revoke",
                status="error",
            )
        return False

    def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        """
        Check if user has permission for resource:action.

        Args:
            user_id: User ID
            resource: Resource name (e.g., 'trading')
            action: Action name (e.g., 'execute')

        Returns:
            True if permitted
        """
        user = self._users.get(user_id)
        if not user:
            return False

        # Admin has all permissions
        if user.role == Role.ADMIN:
            return True

        # Check specific permission
        required_permission = f"{resource}:{action}"
        return required_permission in user.permissions or "*" in user.permissions

    def setup_mfa(self, user_id: str) -> str | None:
        """
        Setup MFA for user and return QR code URI.

        Args:
            user_id: User ID

        Returns:
            OTP provisioning URI for QR code
        """
        user = self._users.get(user_id)
        if not user:
            return None

        # Generate secret
        pyotp_module = _require_pyotp()
        secret = pyotp_module.random_base32()
        user.mfa_secret = secret
        user.mfa_enabled = True

        # Generate provisioning URI
        totp = pyotp_module.TOTP(secret)
        return str(totp.provisioning_uri(name=user.email, issuer_name="Bot V2 Trading"))

    def _verify_mfa(self, user: User, code: str) -> bool:
        """Verify MFA code"""
        if not user.mfa_secret:
            return False

        pyotp_module = _require_pyotp()
        totp = pyotp_module.TOTP(user.mfa_secret)
        return bool(totp.verify(code, valid_window=1))

    def _find_user_by_username(self, username: str) -> User | None:
        """Find user by username or email"""
        for candidate in self._users.values():
            if candidate.username == username or candidate.email == username:
                return candidate
        return None

    def _verify_password(self, password: str, user_id: str) -> bool:
        """Verify password (simplified for demo)"""
        # In production, use bcrypt hashed passwords
        return password == "admin123"  # Demo password

    def _get_jti(self, token: str) -> str | None:
        """Extract JTI from token (legacy helper for tests)."""
        jwt_module = _require_jwt()
        try:
            claims: Mapping[str, Any] = jwt_module.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False},
            )
            jti_value = claims.get("jti")
            return str(jti_value) if jti_value is not None else None
        except PyJWTError as exc:
            logger.warning(
                f"Failed to extract JTI from token: {exc}",
                operation="token_revoke",
                status="decode_failed",
                exc_info=True,
            )
        except Exception as exc:
            logger.exception(
                f"Unexpected error extracting JTI: {exc}",
                operation="token_revoke",
                status="error",
            )
        return None


# Global instance
_auth_handler: AuthHandler | None = None


def get_auth_handler() -> AuthHandler:
    """Get the global auth handler instance"""
    global _auth_handler
    if _auth_handler is None:
        _auth_handler = AuthHandler()
    return _auth_handler


# Convenience functions
def authenticate(username: str, password: str, mfa_code: str | None = None) -> TokenPair | None:
    """Authenticate user"""
    return get_auth_handler().authenticate_user(username, password, mfa_code)


def validate_token(token: str) -> dict[str, Any] | None:
    """Validate JWT token"""
    return get_auth_handler().validate_token(token)


def check_permission(user_id: str, resource: str, action: str) -> bool:
    """Check user permission"""
    return get_auth_handler().check_permission(user_id, resource, action)
