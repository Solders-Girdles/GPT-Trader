"""Token validation tests for AuthHandler."""

from __future__ import annotations

from typing import Any

import pytest

from bot_v2.security.auth_handler import AuthHandler


class TestTokenValidation:
    """Ensure token validation handles success and error paths."""

    def test_validate_valid_token(self, auth_handler: AuthHandler) -> None:
        claims = auth_handler.validate_token("fake.jwt.token")

        assert claims is not None
        assert claims["sub"] == "test-user"
        assert claims["jti"] == "test-jti"
        assert claims["type"] == "access"

    def test_validate_invalid_token(
        self, auth_handler: AuthHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            claims = auth_handler.validate_token("invalid.token")

        assert claims is None
        assert any("Invalid token" in message for message in caplog.messages)

    def test_validate_expired_token(
        self, auth_handler: AuthHandler, fake_jwt_module: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake_jwt_module.decode.side_effect = fake_jwt_module.ExpiredSignatureError("Token expired")

        with caplog.at_level("DEBUG"):
            claims = auth_handler.validate_token("expired.token")

        assert claims is None
        assert any("expired" in message for message in caplog.messages)

    def test_validate_revoked_token(self, auth_handler: AuthHandler) -> None:
        auth_handler._revoked_tokens.add("test-jti")

        assert auth_handler.validate_token("fake.jwt.token") is None

    def test_validate_missing_token(self, auth_handler: AuthHandler) -> None:
        assert auth_handler.validate_token("") is None
        assert auth_handler.validate_token(None) is None  # type: ignore[arg-type]
