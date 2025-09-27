"""
Authentication and Authorization Handler for Bot V2 Trading System

Implements JWT-based authentication with RBAC, MFA support, and secure session management.
"""

import os
import jwt
from jwt import PyJWTError
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import bcrypt
import pyotp
from threading import Lock

logger = logging.getLogger(__name__)


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
    permissions: List[str] = field(default_factory=list)
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TokenPair:
    """JWT token pair (access + refresh)"""
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "Bearer"


class AuthHandler:
    """
    Handles authentication, authorization, and session management.
    Implements JWT with RS256 and RBAC.
    """
    
    # Role permissions mapping
    ROLE_PERMISSIONS = {
        Role.ADMIN: ["*"],  # All permissions
        Role.TRADER: [
            "trading:read", "trading:execute",
            "positions:read", "orders:create",
            "performance:read"
        ],
        Role.VIEWER: [
            "trading:read", "positions:read",
            "performance:read"
        ],
        Role.SERVICE: [
            "data:read", "signals:write",
            "health:read"
        ]
    }
    
    def __init__(self):
        self._lock = Lock()
        self._users = {}  # In production, use database
        self._sessions = {}  # Active sessions
        self._revoked_tokens = set()  # Revoked token JTIs
        
        # JWT configuration
        self.algorithm = "HS256"  # Using HS256 for simplicity, RS256 requires key pair
        self.secret_key = os.environ.get('JWT_SECRET_KEY', secrets.token_urlsafe(32))
        self.issuer = "bot-v2-trading-system"
        self.audience = ["trading-api", "dashboard"]
        self.access_token_lifetime = 900  # 15 minutes
        self.refresh_token_lifetime = 604800  # 7 days
        
        # Initialize with default admin user
        self._create_default_users()
    
    def _create_default_users(self):
        """Create default users for development"""
        if os.environ.get('ENV', 'development') == 'development':
            admin_user = User(
                id="admin-001",
                username="admin",
                email="admin@bot-v2.local",
                role=Role.ADMIN,
                permissions=self.ROLE_PERMISSIONS[Role.ADMIN]
            )
            self._users[admin_user.id] = admin_user
            logger.info("Created default admin user")
    
    def authenticate_user(self, username: str, password: str, mfa_code: Optional[str] = None) -> Optional[TokenPair]:
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
                logger.warning(f"Authentication failed: user not found - {username}")
                return None
            
            # Verify password (simplified for demo)
            if not self._verify_password(password, user.id):
                logger.warning(f"Authentication failed: invalid password - {username}")
                return None
            
            # Check MFA if enabled
            if user.mfa_enabled:
                if not mfa_code or not self._verify_mfa(user, mfa_code):
                    logger.warning(f"Authentication failed: invalid MFA - {username}")
                    return None
            
            # Generate tokens
            tokens = self._generate_tokens(user)
            
            # Create session
            session_id = secrets.token_urlsafe(32)
            self._sessions[session_id] = {
                'user_id': user.id,
                'created_at': datetime.now(timezone.utc),
                'access_token_jti': self._get_jti(tokens.access_token),
                'refresh_token_jti': self._get_jti(tokens.refresh_token)
            }
            
            logger.info(f"User authenticated successfully: {username}")
            return tokens
    
    def _generate_tokens(self, user: User) -> TokenPair:
        """Generate JWT token pair for user"""
        now = datetime.now(timezone.utc)
        
        # Access token claims
        access_jti = secrets.token_urlsafe(32)
        access_exp = now + timedelta(seconds=self.access_token_lifetime)
        access_claims = {
            'sub': user.id,
            'username': user.username,
            'role': user.role.value,
            'permissions': user.permissions,
            'iat': now,
            'exp': access_exp,
            'jti': access_jti,
            'iss': self.issuer,
            'aud': self.audience,
            'type': 'access'
        }
        
        # Refresh token claims
        refresh_jti = secrets.token_urlsafe(32)
        refresh_exp = now + timedelta(seconds=self.refresh_token_lifetime)
        refresh_claims = {
            'sub': user.id,
            'iat': now,
            'exp': refresh_exp,
            'jti': refresh_jti,
            'iss': self.issuer,
            'aud': self.audience,
            'type': 'refresh'
        }
        
        # Generate tokens
        access_token = jwt.encode(access_claims, self.secret_key, algorithm=self.algorithm)
        refresh_token = jwt.encode(refresh_claims, self.secret_key, algorithm=self.algorithm)
        
        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.access_token_lifetime
        )
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token and return claims.
        
        Args:
            token: JWT token string
            
        Returns:
            Token claims if valid, None otherwise
        """
        try:
            # Decode token
            claims = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience
            )
            
            # Check if revoked
            if claims.get('jti') in self._revoked_tokens:
                logger.warning(f"Token is revoked: {claims.get('jti')}")
                return None
            
            return claims
            
        except jwt.ExpiredSignatureError:
            logger.debug("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_token(self, refresh_token: str) -> Optional[TokenPair]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New TokenPair if successful
        """
        # Validate refresh token
        claims = self.validate_token(refresh_token)
        if not claims or claims.get('type') != 'refresh':
            return None
        
        # Get user
        user = self._users.get(claims['sub'])
        if not user or not user.is_active:
            return None
        
        # Revoke old refresh token
        self._revoked_tokens.add(claims['jti'])
        
        # Generate new tokens
        return self._generate_tokens(user)
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token"""
        try:
            claims = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False}
            )
            jti = claims.get('jti')
            if jti:
                self._revoked_tokens.add(jti)
                logger.info(f"Token revoked: {jti}")
                return True
        except PyJWTError as exc:
            logger.warning("Failed to decode token during revocation: %s", exc, exc_info=True)
        except Exception as exc:
            logger.exception("Unexpected error revoking token: %s", exc)
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
    
    def setup_mfa(self, user_id: str) -> Optional[str]:
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
        secret = pyotp.random_base32()
        user.mfa_secret = secret
        user.mfa_enabled = True
        
        # Generate provisioning URI
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(
            name=user.email,
            issuer_name='Bot V2 Trading'
        )
    
    def _verify_mfa(self, user: User, code: str) -> bool:
        """Verify MFA code"""
        if not user.mfa_secret:
            return False
        
        totp = pyotp.TOTP(user.mfa_secret)
        return totp.verify(code, valid_window=1)
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username or email"""
        for user in self._users.values():
            if user.username == username or user.email == username:
                return user
        return None
    
    def _verify_password(self, password: str, user_id: str) -> bool:
        """Verify password (simplified for demo)"""
        # In production, use bcrypt hashed passwords
        return password == "admin123"  # Demo password
    
    def _get_jti(self, token: str) -> Optional[str]:
        """Extract JTI from token"""
        try:
            claims = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_signature": False}
            )
            return claims.get('jti')
        except PyJWTError as exc:
            logger.warning("Failed to extract JTI from token: %s", exc, exc_info=True)
        except Exception as exc:
            logger.exception("Unexpected error extracting JTI: %s", exc)
        return None


# Global instance
_auth_handler = None


def get_auth_handler() -> AuthHandler:
    """Get the global auth handler instance"""
    global _auth_handler
    if _auth_handler is None:
        _auth_handler = AuthHandler()
    return _auth_handler


# Convenience functions
def authenticate(username: str, password: str, mfa_code: Optional[str] = None) -> Optional[TokenPair]:
    """Authenticate user"""
    return get_auth_handler().authenticate_user(username, password, mfa_code)


def validate_token(token: str) -> Optional[Dict[str, Any]]:
    """Validate JWT token"""
    return get_auth_handler().validate_token(token)


def check_permission(user_id: str, resource: str, action: str) -> bool:
    """Check user permission"""
    return get_auth_handler().check_permission(user_id, resource, action)
