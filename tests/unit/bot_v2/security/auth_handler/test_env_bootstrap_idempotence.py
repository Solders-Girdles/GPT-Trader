"""Bootstrap idempotence tests for AuthHandler."""

from __future__ import annotations

import sys
from typing import Any

from bot_v2.security.auth_handler import Role, User


class TestBootstrapIdempotence:
    """Ensure repeated bootstrap calls do not duplicate users."""

    def _build_handler(self, auth_runtime_settings: Any) -> Any:
        from bot_v2.security import auth_handler

        auth_handler._jwt_module = sys.modules["jwt"]
        auth_handler._pyotp_module = sys.modules["pyotp"]
        return auth_handler.AuthHandler(settings=auth_runtime_settings)

    def test_multiple_bootstrap_calls_idempotent(
        self, auth_runtime_settings: Any, mock_auth_dependencies: None
    ) -> None:
        handler1 = self._build_handler(auth_runtime_settings)
        handler2 = self._build_handler(auth_runtime_settings)

        assert "admin-001" in handler1._users
        assert "admin-001" in handler2._users
        assert len(handler1._users) == 1
        assert len(handler2._users) == 1

    def test_bootstrap_with_existing_users(
        self, auth_runtime_settings: Any, mock_auth_dependencies: None
    ) -> None:
        handler = self._build_handler(auth_runtime_settings)

        custom_user = User(
            id="custom-001",
            username="customuser",
            email="custom@example.com",
            role=Role.TRADER,
            permissions=["trading:read"],
        )
        handler._users[custom_user.id] = custom_user

        assert "admin-001" in handler._users
        assert "custom-001" in handler._users
        assert len(handler._users) == 2
