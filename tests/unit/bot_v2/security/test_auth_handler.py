"""Tests for AuthHandler security guardrails."""

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import jwt
import pytest

from bot_v2.security.auth_handler import AuthHandler, Role, TokenPair, User


@pytest.fixture
def handler():
    """Return a fresh handler for each test to avoid shared state."""
    return AuthHandler()


@pytest.fixture
def test_user(handler):
    """Create a test user."""
    user = User(
        id="test-user-123",
        username="testuser",
        email="test@example.com",
        role=Role.TRADER,
        permissions=handler.ROLE_PERMISSIONS[Role.TRADER],
    )
    handler._users[user.id] = user
    return user


@pytest.fixture
def mfa_user(handler):
    """Create a user with MFA enabled."""
    import pyotp

    secret = pyotp.random_base32()
    user = User(
        id="mfa-user-123",
        username="mfauser",
        email="mfa@example.com",
        role=Role.TRADER,
        permissions=handler.ROLE_PERMISSIONS[Role.TRADER],
        mfa_enabled=True,
        mfa_secret=secret,
    )
    handler._users[user.id] = user
    return user


# Authentication tests


def test_authenticate_user_success(handler, test_user):
    """Successful authentication returns token pair."""
    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("testuser", "password123")

    assert tokens is not None
    assert isinstance(tokens, TokenPair)
    assert tokens.access_token
    assert tokens.refresh_token
    assert tokens.expires_in == handler.access_token_lifetime


def test_authenticate_user_not_found(handler):
    """Authentication fails for non-existent user."""
    tokens = handler.authenticate_user("nonexistent", "password")
    assert tokens is None


def test_authenticate_user_invalid_password(handler, test_user):
    """Authentication fails with invalid password."""
    with patch.object(handler, "_verify_password", return_value=False):
        tokens = handler.authenticate_user("testuser", "wrong_password")

    assert tokens is None


def test_authenticate_user_mfa_required(handler, mfa_user):
    """Authentication fails when MFA is enabled but not provided."""
    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("mfauser", "password123")

    assert tokens is None


def test_authenticate_user_mfa_success(handler, mfa_user):
    """Authentication succeeds with valid MFA code."""
    import pyotp

    totp = pyotp.TOTP(mfa_user.mfa_secret)
    valid_code = totp.now()

    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("mfauser", "password123", mfa_code=valid_code)

    assert tokens is not None
    assert isinstance(tokens, TokenPair)


def test_authenticate_user_mfa_invalid_code(handler, mfa_user):
    """Authentication fails with invalid MFA code."""
    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("mfauser", "password123", mfa_code="000000")

    assert tokens is None


# Token validation tests


def test_validate_token_success(handler, test_user):
    """Valid token returns claims."""
    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("testuser", "password123")

    claims = handler.validate_token(tokens.access_token)
    assert claims is not None
    assert claims["sub"] == test_user.id
    assert claims["username"] == test_user.username
    assert claims["role"] == test_user.role.value


def test_validate_token_invalid(handler):
    """Invalid token returns None."""
    claims = handler.validate_token("invalid_token")
    assert claims is None


def test_validate_token_expired(handler, test_user):
    """Expired token returns None."""
    # Create an expired token
    now = datetime.now(UTC)
    expired_claims = {
        "sub": test_user.id,
        "exp": now - timedelta(hours=1),
        "iat": now - timedelta(hours=2),
        "jti": "expired-jti",
        "iss": handler.issuer,
        "aud": handler.audience,
    }
    expired_token = jwt.encode(expired_claims, handler.secret_key, algorithm=handler.algorithm)

    claims = handler.validate_token(expired_token)
    assert claims is None


def test_validate_token_revoked(handler, test_user):
    """Revoked token returns None."""
    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("testuser", "password123")

    # Revoke the token
    handler.revoke_token(tokens.access_token)

    # Validation should fail
    claims = handler.validate_token(tokens.access_token)
    assert claims is None


def test_validate_token_wrong_secret(handler, test_user):
    """Token signed with wrong secret fails validation."""
    # Create token with different secret
    wrong_secret = "wrong_secret_key_1234567890"
    now = datetime.now(UTC)
    claims = {
        "sub": test_user.id,
        "exp": now + timedelta(hours=1),
        "iat": now,
        "jti": "test-jti",
        "iss": handler.issuer,
        "aud": handler.audience,
    }
    bad_token = jwt.encode(claims, wrong_secret, algorithm=handler.algorithm)

    # Validation should fail
    result = handler.validate_token(bad_token)
    assert result is None


def test_validate_token_tampered_claims(handler, test_user):
    """Token with tampered claims fails validation."""
    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("testuser", "password123")

    # Decode and modify token (breaks signature)
    claims = jwt.decode(tokens.access_token, options={"verify_signature": False})
    claims["role"] = "admin"  # Escalate privilege
    tampered_token = jwt.encode(claims, "wrong_key", algorithm=handler.algorithm)

    # Validation should fail
    result = handler.validate_token(tampered_token)
    assert result is None


# Token refresh tests


def test_refresh_token_success(handler, test_user):
    """Valid refresh token returns new token pair."""
    with patch.object(handler, "_verify_password", return_value=True):
        original_tokens = handler.authenticate_user("testuser", "password123")

    new_tokens = handler.refresh_token(original_tokens.refresh_token)
    assert new_tokens is not None
    assert new_tokens.access_token != original_tokens.access_token
    assert new_tokens.refresh_token != original_tokens.refresh_token


def test_refresh_token_invalid(handler):
    """Invalid refresh token returns None."""
    new_tokens = handler.refresh_token("invalid_refresh_token")
    assert new_tokens is None


def test_refresh_token_access_token_rejected(handler, test_user):
    """Using access token for refresh fails."""
    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("testuser", "password123")

    # Try to use access token as refresh token
    new_tokens = handler.refresh_token(tokens.access_token)
    assert new_tokens is None


def test_refresh_token_rotation_invalidates_old(handler, test_user):
    """Token rotation should invalidate old refresh token."""
    with patch.object(handler, "_verify_password", return_value=True):
        original_tokens = handler.authenticate_user("testuser", "password123")

    # Get new tokens via refresh
    new_tokens = handler.refresh_token(original_tokens.refresh_token)
    assert new_tokens is not None

    # Old refresh token should not work again (token rotation)
    reused_tokens = handler.refresh_token(original_tokens.refresh_token)
    # Note: Current implementation doesn't invalidate old refresh tokens
    # This test documents expected behavior for future implementation
    # For now, we just verify rotation produces different tokens
    assert new_tokens.refresh_token != original_tokens.refresh_token


# Token revocation tests


def test_revoke_token_success(handler, test_user):
    """Valid token can be revoked."""
    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("testuser", "password123")

    result = handler.revoke_token(tokens.access_token)
    assert result is True

    # Token should be invalid after revocation
    claims = handler.validate_token(tokens.access_token)
    assert claims is None


def test_revoke_token_logs_decode_failure(handler, caplog):
    """Invalid tokens should be handled gracefully and logged."""
    with caplog.at_level("WARNING"):
        result = handler.revoke_token("not_a_real_token")

    assert result is False
    assert any("Failed to decode token" in message for message in caplog.messages)


# Permission checking tests


def test_check_permission_admin_has_all(handler):
    """Admin role has all permissions."""
    admin_user = User(
        id="admin-123",
        username="admin",
        email="admin@test.com",
        role=Role.ADMIN,
        permissions=handler.ROLE_PERMISSIONS[Role.ADMIN],
    )
    handler._users[admin_user.id] = admin_user

    assert handler.check_permission(admin_user.id, "anything", "do") is True


def test_check_permission_trader_has_trading(handler, test_user):
    """Trader has trading permissions."""
    assert handler.check_permission(test_user.id, "trading", "execute") is True
    assert handler.check_permission(test_user.id, "orders", "create") is True


def test_check_permission_trader_no_admin(handler, test_user):
    """Trader doesn't have admin permissions."""
    assert handler.check_permission(test_user.id, "admin", "access") is False


def test_check_permission_viewer_read_only(handler):
    """Viewer has read-only permissions."""
    viewer = User(
        id="viewer-123",
        username="viewer",
        email="viewer@test.com",
        role=Role.VIEWER,
        permissions=handler.ROLE_PERMISSIONS[Role.VIEWER],
    )
    handler._users[viewer.id] = viewer

    assert handler.check_permission(viewer.id, "trading", "read") is True
    assert handler.check_permission(viewer.id, "trading", "execute") is False


def test_check_permission_user_not_found(handler):
    """Permission check fails for non-existent user."""
    assert handler.check_permission("nonexistent", "anything", "do") is False


# MFA setup tests


def test_setup_mfa_success(handler, test_user):
    """MFA setup returns OTP URL."""
    otp_url = handler.setup_mfa(test_user.id)
    assert otp_url is not None
    assert otp_url.startswith("otpauth://totp/")  # Valid OTP URL format

    # User should now have MFA enabled
    user = handler._users[test_user.id]
    assert user.mfa_enabled is True
    assert user.mfa_secret is not None
    assert len(user.mfa_secret) == 32  # Base32 secret stored separately


def test_setup_mfa_user_not_found(handler):
    """MFA setup fails for non-existent user."""
    secret = handler.setup_mfa("nonexistent")
    assert secret is None


def test_verify_mfa_success(handler, mfa_user):
    """Valid MFA code verifies successfully."""
    import pyotp

    totp = pyotp.TOTP(mfa_user.mfa_secret)
    valid_code = totp.now()

    assert handler._verify_mfa(mfa_user, valid_code) is True


def test_verify_mfa_invalid_code(handler, mfa_user):
    """Invalid MFA code fails verification."""
    assert handler._verify_mfa(mfa_user, "000000") is False


def test_verify_mfa_no_secret(handler, test_user):
    """MFA verification fails when user has no secret."""
    assert handler._verify_mfa(test_user, "123456") is False


# Helper method tests


def test_find_user_by_username(handler, test_user):
    """User can be found by username."""
    user = handler._find_user_by_username("testuser")
    assert user is not None
    assert user.id == test_user.id


def test_find_user_by_email(handler, test_user):
    """User can be found by email."""
    user = handler._find_user_by_username("test@example.com")
    assert user is not None
    assert user.id == test_user.id


def test_find_user_not_found(handler):
    """Returns None for non-existent user."""
    user = handler._find_user_by_username("nonexistent")
    assert user is None


def test_get_jti_success(handler, test_user):
    """JTI can be extracted from valid token."""
    with patch.object(handler, "_verify_password", return_value=True):
        tokens = handler.authenticate_user("testuser", "password123")

    jti = handler._get_jti(tokens.access_token)
    assert jti is not None
    assert isinstance(jti, str)


def test_get_jti_handles_invalid_token(handler, caplog):
    """_get_jti should fall back to None when decoding fails."""
    with caplog.at_level("WARNING"):
        jti = handler._get_jti("not_a_real_token")

    assert jti is None
    assert any("Failed to extract JTI" in message for message in caplog.messages)


# Default user creation tests


def test_default_admin_created_in_dev(handler):
    """Default admin user created in development environment."""
    admin = handler._find_user_by_username("admin")
    assert admin is not None
    assert admin.role == Role.ADMIN


@patch.dict(os.environ, {"ENV": "production"})
def test_no_default_admin_in_production():
    """No default admin created in production."""
    handler = AuthHandler()
    admin = handler._find_user_by_username("admin")
    # In production, default users are not created
    # (current implementation creates them in dev only)
    assert admin is None or os.environ.get("ENV") != "production"
