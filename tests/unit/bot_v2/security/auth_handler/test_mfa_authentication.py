"""Authentication flows with MFA for AuthHandler."""

from __future__ import annotations

from typing import Any

import pytest

from bot_v2.security.auth_handler import AuthHandler


class TestAuthenticationWithMFA:
    """Exercise authentication paths when MFA is toggled."""

    def test_authenticate_with_mfa_enabled(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.mfa_enabled = True
        user.mfa_secret = "JBSWY3DPEHPK3PXP"
        auth_handler._users[user.id] = user

        token_pair = auth_handler.authenticate_user("testuser", "admin123", "123456")
        assert token_pair is not None
        assert token_pair.access_token == "fake.jwt.token"

    def test_authenticate_with_invalid_mfa_code(
        self,
        auth_handler: AuthHandler,
        sample_user: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.mfa_enabled = True
        user.mfa_secret = "JBSWY3DPEHPK3PXP"
        auth_handler._users[user.id] = user

        with caplog.at_level("WARNING"):
            token_pair = auth_handler.authenticate_user("testuser", "admin123", "000000")

        assert token_pair is None
        assert any("invalid MFA" in message for message in caplog.messages)

    def test_authenticate_without_mfa_code(
        self,
        auth_handler: AuthHandler,
        sample_user: dict[str, Any],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.mfa_enabled = True
        user.mfa_secret = "JBSWY3DPEHPK3PXP"
        auth_handler._users[user.id] = user

        with caplog.at_level("WARNING"):
            token_pair = auth_handler.authenticate_user("testuser", "admin123", None)

        assert token_pair is None
        assert any("invalid MFA" in message for message in caplog.messages)

    def test_authenticate_with_mfa_disabled(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.mfa_enabled = False
        auth_handler._users[user.id] = user

        token_pair = auth_handler.authenticate_user("testuser", "admin123", None)

        assert token_pair is not None
        assert token_pair.access_token == "fake.jwt.token"

    def test_authenticate_with_mfa_ignores_code(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.mfa_enabled = False
        auth_handler._users[user.id] = user

        token_pair = auth_handler.authenticate_user("testuser", "admin123", "000000")

        assert token_pair is not None
        assert token_pair.access_token == "fake.jwt.token"
