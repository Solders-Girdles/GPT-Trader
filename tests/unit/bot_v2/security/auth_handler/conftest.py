"""Shared fixtures for auth_handler tests."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot_v2.orchestration.runtime_settings import RuntimeSettings


@pytest.fixture
def auth_runtime_settings(runtime_settings_factory) -> RuntimeSettings:
    """Runtime settings tailored for auth_handler tests."""
    return runtime_settings_factory(
        env_overrides={
            "ENV": "development",
            "JWT_SECRET_KEY": "test-secret-key-for-jwt-signing-32-chars",
        }
    )


@pytest.fixture
def fake_jwt_module() -> MagicMock:
    """Fake jwt module for optional dependency guards."""
    jwt_mock = MagicMock()

    # Mock JWT encoding/decoding
    class MockJWT:
        def encode(self, payload: dict, key: str, algorithm: str = "HS256") -> str:
            return "fake.jwt.token"

        def decode(self, token: str, key: str, algorithms: list[str], **kwargs) -> dict:
            if token == "fake.jwt.token":
                return {
                    "sub": "test-user",
                    "jti": "test-jti",
                    "exp": 9999999999,
                    "iat": 1000000000,
                    "iss": "bot-v2-trading-system",
                    "aud": ["trading-api", "dashboard"],
                    "type": "access",
                }
            # Raise the correct exception type that AuthHandler expects
            raise jwt_mock.InvalidTokenError("Invalid token")

        class ExpiredSignatureError(Exception):
            pass

        class InvalidTokenError(Exception):
            pass

    # Create proper MagicMock instances for methods
    jwt_mock.encode = MagicMock(side_effect=MockJWT().encode)
    jwt_mock.decode = MagicMock(side_effect=MockJWT().decode)
    jwt_mock.ExpiredSignatureError = MockJWT.ExpiredSignatureError
    jwt_mock.InvalidTokenError = MockJWT.InvalidTokenError

    return jwt_mock


@pytest.fixture
def fake_pyotp_module() -> MagicMock:
    """Fake pyotp module for optional dependency guards."""
    pyotp_mock = MagicMock()

    class MockTOTP:
        def __init__(self, secret: str):
            self.secret = secret

        def verify(self, code: str, valid_window: int = 0) -> bool:
            return code == "123456"

        def provisioning_uri(self, name: str, issuer_name: str) -> str:
            return f"otpauth://totp/{issuer_name}:{name}?secret={self.secret}&issuer={issuer_name}"

    pyotp_mock.TOTP = MockTOTP
    pyotp_mock.random_base32 = lambda: "JBSWY3DPEHPK3PXP"

    return pyotp_mock


@pytest.fixture
def mock_auth_dependencies(
    monkeypatch: pytest.MonkeyPatch, fake_jwt_module: MagicMock, fake_pyotp_module: MagicMock
) -> None:
    """Mock optional dependencies for auth_handler tests."""
    monkeypatch.setitem(sys.modules, "jwt", fake_jwt_module)
    monkeypatch.setitem(sys.modules, "pyotp", fake_pyotp_module)


@pytest.fixture
def auth_handler(auth_runtime_settings: RuntimeSettings, mock_auth_dependencies: None) -> Any:
    """AuthHandler instance with mocked dependencies."""
    # Import after modules are patched
    from bot_v2.security import auth_handler

    # Explicitly assign the fake modules to the imported module
    auth_handler._jwt_module = sys.modules["jwt"]
    auth_handler._pyotp_module = sys.modules["pyotp"]

    # Update cached exception classes
    auth_handler.PyJWTError = Exception

    # Create instance with settings
    return auth_handler.AuthHandler(settings=auth_runtime_settings)


@pytest.fixture
def sample_user() -> dict[str, Any]:
    """Sample user data for testing."""
    from bot_v2.security.auth_handler import Role

    return {
        "id": "test-user-001",
        "username": "testuser",
        "email": "test@example.com",
        "role": Role.TRADER,
        "permissions": ["trading:read", "trading:execute", "positions:read"],
        "mfa_enabled": False,
        "is_active": True,
    }
