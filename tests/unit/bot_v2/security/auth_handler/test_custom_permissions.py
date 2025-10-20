"""Custom permission tests for AuthHandler."""

from __future__ import annotations

from typing import Any

from bot_v2.security.auth_handler import AuthHandler, Role


class TestCustomPermissions:
    """Ensure manual permission overrides behave as expected."""

    def test_custom_permissions_override_role(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.role = Role.VIEWER
        user.permissions = ["trading:execute", "orders:create"]
        auth_handler._users[user.id] = user

        assert auth_handler.check_permission(user.id, "trading", "execute")
        assert auth_handler.check_permission(user.id, "orders", "create")
        assert not auth_handler.check_permission(user.id, "positions", "read")

    def test_empty_permissions_deny_all(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.role = Role.TRADER
        user.permissions = []
        auth_handler._users[user.id] = user

        assert not auth_handler.check_permission(user.id, "trading", "read")
        assert not auth_handler.check_permission(user.id, "trading", "execute")

    def test_permission_case_sensitivity(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.role = Role.TRADER
        user.permissions = auth_handler.ROLE_PERMISSIONS[Role.TRADER]
        auth_handler._users[user.id] = user

        assert auth_handler.check_permission(user.id, "trading", "read")
        assert not auth_handler.check_permission(user.id, "Trading", "read")
        assert not auth_handler.check_permission(user.id, "trading", "Read")
