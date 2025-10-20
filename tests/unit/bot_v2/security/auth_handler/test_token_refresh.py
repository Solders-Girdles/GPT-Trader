"""Token refresh tests for AuthHandler."""

from __future__ import annotations

from typing import Any

from bot_v2.security.auth_handler import AuthHandler


class TestTokenRefresh:
    """Cover refresh flows and failure modes."""

    def test_refresh_valid_token(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any], fake_jwt_module: Any
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        auth_handler._users[user.id] = user

        def custom_decode(token: str, key: str, algorithms: list[str], **kwargs) -> dict:
            if token == "fake.jwt.token":
                return {
                    "sub": user.id,
                    "jti": "test-jti",
                    "exp": 9999999999,
                    "iat": 1000000000,
                    "iss": "bot-v2-trading-system",
                    "aud": ["trading-api", "dashboard"],
                    "type": "refresh",
                }
            raise fake_jwt_module.InvalidTokenError("Invalid token")

        fake_jwt_module.decode.side_effect = custom_decode

        token_pair = auth_handler.refresh_token("fake.jwt.token")

        assert token_pair is not None
        assert token_pair.access_token == "fake.jwt.token"
        assert token_pair.refresh_token == "fake.jwt.token"

    def test_refresh_invalid_token(self, auth_handler: AuthHandler) -> None:
        assert auth_handler.refresh_token("invalid.refresh.token") is None

    def test_refresh_wrong_token_type(
        self, auth_handler: AuthHandler, fake_jwt_module: Any
    ) -> None:
        fake_jwt_module.decode.return_value = {
            "sub": "test-user",
            "jti": "test-jti",
            "type": "access",
        }

        assert auth_handler.refresh_token("access.token") is None

    def test_refresh_inactive_user(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.is_active = False
        auth_handler._users[user.id] = user

        assert auth_handler.refresh_token("fake.jwt.token") is None

    def test_refresh_revokes_old_token(
        self, auth_handler: AuthHandler, fake_jwt_module: Any, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        auth_handler._users[user.id] = user

        def custom_decode(token: str, key: str, algorithms: list[str], **kwargs) -> dict:
            if token == "fake.jwt.token":
                return {
                    "sub": user.id,
                    "jti": "old-jti",
                    "exp": 9999999999,
                    "iat": 1000000000,
                    "iss": "bot-v2-trading-system",
                    "aud": ["trading-api", "dashboard"],
                    "type": "refresh",
                }
            raise fake_jwt_module.InvalidTokenError("Invalid token")

        fake_jwt_module.decode.side_effect = custom_decode

        auth_handler.refresh_token("fake.jwt.token")
        assert "old-jti" in auth_handler._revoked_tokens
