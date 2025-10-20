"""Permission checking tests for AuthHandler."""

from __future__ import annotations

from typing import Any

from bot_v2.security.auth_handler import AuthHandler, Role


class TestPermissionChecking:
    """Exercise per-role permission checks."""

    def test_admin_has_all_permissions(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.role = Role.ADMIN
        auth_handler._users[user.id] = user

        assert auth_handler.check_permission(user.id, "trading", "execute")
        assert auth_handler.check_permission(user.id, "system", "admin")
        assert auth_handler.check_permission(user.id, "users", "delete")

    def test_trader_has_limited_permissions(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.role = Role.TRADER
        user.permissions = auth_handler.ROLE_PERMISSIONS[Role.TRADER]
        auth_handler._users[user.id] = user

        assert auth_handler.check_permission(user.id, "trading", "read")
        assert auth_handler.check_permission(user.id, "orders", "create")
        assert not auth_handler.check_permission(user.id, "users", "delete")

    def test_viewer_has_read_only_permissions(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.role = Role.VIEWER
        user.permissions = auth_handler.ROLE_PERMISSIONS[Role.VIEWER]
        auth_handler._users[user.id] = user

        assert auth_handler.check_permission(user.id, "trading", "read")
        assert not auth_handler.check_permission(user.id, "trading", "execute")

    def test_service_has_specific_permissions(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.role = Role.SERVICE
        user.permissions = auth_handler.ROLE_PERMISSIONS[Role.SERVICE]
        auth_handler._users[user.id] = user

        assert auth_handler.check_permission(user.id, "data", "read")
        assert not auth_handler.check_permission(user.id, "trading", "execute")

    def test_permission_check_nonexistent_user(self, auth_handler: AuthHandler) -> None:
        assert auth_handler.check_permission("nonexistent-user", "trading", "read") is False

    def test_permission_check_with_wildcard(
        self, auth_handler: AuthHandler, sample_user: dict[str, Any]
    ) -> None:
        from bot_v2.security.auth_handler import User

        user = User(**sample_user)
        user.permissions = ["*"]
        auth_handler._users[user.id] = user

        assert auth_handler.check_permission(user.id, "any", "action")
        assert auth_handler.check_permission(user.id, "system", "admin")
