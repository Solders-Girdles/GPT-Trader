"""Token revocation tests for AuthHandler."""

from __future__ import annotations

from typing import Any

import pytest

from bot_v2.security.auth_handler import AuthHandler


class TestTokenRevocation:
    """Confirm revoke flows and JTI extraction."""

    def test_revoke_valid_token(
        self, auth_handler: AuthHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO"):
            result = auth_handler.revoke_token("fake.jwt.token")

        assert result is True
        assert "test-jti" in auth_handler._revoked_tokens
        assert any("Token revoked" in message for message in caplog.messages)

    def test_revoke_invalid_token_logs_error(
        self, auth_handler: AuthHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            result = auth_handler.revoke_token("not_a_real_token")

        assert result is False
        assert any("Failed to decode token" in message for message in caplog.messages)

    def test_get_jti_handles_invalid_token(
        self, auth_handler: AuthHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            jti = auth_handler._get_jti("not_a_real_token")

        assert jti is None
        assert any("Failed to extract JTI" in message for message in caplog.messages)

    def test_revoke_token_exception_handling(
        self, auth_handler: AuthHandler, fake_jwt_module: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake_jwt_module.decode.side_effect = Exception("Unexpected error")

        with caplog.at_level("DEBUG"):
            result = auth_handler.revoke_token("error.token")

        assert result is False
        assert any(
            "Failed to decode token during revocation" in message for message in caplog.messages
        )
