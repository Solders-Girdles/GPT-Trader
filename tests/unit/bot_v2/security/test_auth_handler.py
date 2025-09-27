"""Tests for AuthHandler security guardrails."""

import pytest

from bot_v2.security.auth_handler import AuthHandler


@pytest.fixture
def handler():
    """Return a fresh handler for each test to avoid shared state."""
    return AuthHandler()


def test_revoke_token_logs_decode_failure(handler, caplog):
    """Invalid tokens should be handled gracefully and logged."""
    with caplog.at_level("WARNING"):
        result = handler.revoke_token("not_a_real_token")

    assert result is False
    assert any("Failed to decode token" in message for message in caplog.messages)


def test_get_jti_handles_invalid_token(handler, caplog):
    """_get_jti should fall back to None when decoding fails."""
    with caplog.at_level("WARNING"):
        jti = handler._get_jti("not_a_real_token")

    assert jti is None
    assert any("Failed to extract JTI" in message for message in caplog.messages)
