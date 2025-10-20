"""MFA setup tests for AuthHandler."""

from __future__ import annotations

from typing import Any

from bot_v2.security.auth_handler import AuthHandler


class TestMFASetup:
    """Ensure MFA provisioning state is persisted correctly."""

    def test_setup_mfa_for_valid_user(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        auth_handler._users[user.id] = user

        provisioning_uri = auth_handler.setup_mfa(user.id)

        assert provisioning_uri is not None
        assert "otpauth://totp/Bot V2 Trading" in provisioning_uri
        assert user.mfa_enabled is True
        assert user.mfa_secret == "JBSWY3DPEHPK3PXP"

    def test_setup_mfa_for_invalid_user(self, auth_handler: AuthHandler) -> None:
        assert auth_handler.setup_mfa("non-existent-user") is None

    def test_mfa_provisioning_uri_format(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        auth_handler._users[user.id] = user

        provisioning_uri = auth_handler.setup_mfa(user.id)

        assert "otpauth://totp/" in provisioning_uri
        assert f"Bot V2 Trading:{user.email}" in provisioning_uri
        assert "secret=JBSWY3DPEHPK3PXP" in provisioning_uri
        assert "issuer=Bot V2 Trading" in provisioning_uri
