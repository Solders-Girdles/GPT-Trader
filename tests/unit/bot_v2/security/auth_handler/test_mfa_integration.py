"""MFA integration tests for AuthHandler."""

from __future__ import annotations

from typing import Any

from bot_v2.security.auth_handler import AuthHandler


class TestMFAIntegration:
    """Confirm MFA changes interact safely with existing sessions."""

    def test_mfa_setup_affects_existing_sessions(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        auth_handler._users[user.id] = user

        token_pair = auth_handler.authenticate_user("testuser", "admin123", None)
        assert token_pair is not None

        provisioning_uri = auth_handler.setup_mfa(user.id)
        assert provisioning_uri is not None

        claims = auth_handler.validate_token(token_pair.access_token)
        assert claims is not None

    def test_mfa_persistence_across_instances(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        auth_handler._users[user.id] = user

        provisioning_uri = auth_handler.setup_mfa(user.id)
        assert provisioning_uri is not None

        stored_user = auth_handler._users[user.id]
        assert stored_user.mfa_enabled is True
        assert stored_user.mfa_secret == "JBSWY3DPEHPK3PXP"
