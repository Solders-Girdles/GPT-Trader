"""Role definition tests for AuthHandler."""

from __future__ import annotations

from bot_v2.security.auth_handler import AuthHandler, Role


class TestRoleDefinitions:
    """Validate static permission lists on roles."""

    def test_admin_role_has_all_permissions(self, auth_handler: AuthHandler) -> None:
        permissions = auth_handler.ROLE_PERMISSIONS[Role.ADMIN]
        assert permissions == ["*"]

    def test_trader_role_permissions(self, auth_handler: AuthHandler) -> None:
        expected = {
            "trading:read",
            "trading:execute",
            "positions:read",
            "orders:create",
            "performance:read",
        }
        assert set(auth_handler.ROLE_PERMISSIONS[Role.TRADER]) == expected

    def test_viewer_role_permissions(self, auth_handler: AuthHandler) -> None:
        expected = {"trading:read", "positions:read", "performance:read"}
        assert set(auth_handler.ROLE_PERMISSIONS[Role.VIEWER]) == expected

    def test_service_role_permissions(self, auth_handler: AuthHandler) -> None:
        expected = {"data:read", "signals:write", "health:read"}
        assert set(auth_handler.ROLE_PERMISSIONS[Role.SERVICE]) == expected
