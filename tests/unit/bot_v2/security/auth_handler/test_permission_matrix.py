"""Permission matrix parametrized tests for AuthHandler."""

from __future__ import annotations

from typing import Any

import pytest

from bot_v2.security.auth_handler import AuthHandler, Role


@pytest.mark.parametrize(
    "role,resource,action,expected",
    [
        (Role.ADMIN, "trading", "read", True),
        (Role.ADMIN, "system", "admin", True),
        (Role.TRADER, "trading", "execute", True),
        (Role.TRADER, "users", "delete", False),
        (Role.VIEWER, "positions", "read", True),
        (Role.VIEWER, "orders", "create", False),
        (Role.SERVICE, "signals", "write", True),
        (Role.SERVICE, "trading", "execute", False),
    ],
)
def test_permission_matrix(
    auth_handler: AuthHandler,
    sample_user: dict[str, Any],
    role: Role,
    resource: str,
    action: str,
    expected: bool,
) -> None:
    from bot_v2.security.auth_handler import User

    user = User(**sample_user)
    user.role = role
    user.permissions = auth_handler.ROLE_PERMISSIONS[role]
    auth_handler._users[user.id] = user

    assert auth_handler.check_permission(user.id, resource, action) == expected
