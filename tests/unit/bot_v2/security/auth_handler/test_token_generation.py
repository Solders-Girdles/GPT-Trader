"""Token generation tests for AuthHandler."""

from __future__ import annotations

from typing import Any

from bot_v2.security.auth_handler import AuthHandler


class TestTokenGeneration:
    """Verify token creation and claim structure."""

    def test_generate_access_token_claims(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        token_pair, access_jti, refresh_jti = auth_handler._generate_tokens(user)

        assert token_pair.access_token == "fake.jwt.token"
        assert token_pair.refresh_token == "fake.jwt.token"
        assert token_pair.expires_in == auth_handler.access_token_lifetime
        assert token_pair.token_type == "Bearer"
        assert access_jti is not None
        assert refresh_jti is not None

    def test_generate_refresh_token_claims(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        token_pair, access_jti, refresh_jti = auth_handler._generate_tokens(user)

        assert access_jti != refresh_jti

    def test_token_expiration_times(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        token_pair, _, _ = auth_handler._generate_tokens(user)

        assert token_pair.expires_in == 900
