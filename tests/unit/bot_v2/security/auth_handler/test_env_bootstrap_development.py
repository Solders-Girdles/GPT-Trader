"""Development environment bootstrap tests for AuthHandler."""

from __future__ import annotations

import sys
from typing import Any

import pytest


class TestDevelopmentBootstrap:
    """Ensure development settings create the default admin user."""

    def _build_handler(self, auth_runtime_settings: Any) -> Any:
        from bot_v2.security import auth_handler

        auth_handler._jwt_module = sys.modules["jwt"]
        auth_handler._pyotp_module = sys.modules["pyotp"]
        return auth_handler.AuthHandler(settings=auth_runtime_settings)

    def test_development_env_creates_default_admin(
        self, auth_runtime_settings: Any, mock_auth_dependencies: None
    ) -> None:
        handler = self._build_handler(auth_runtime_settings)
        assert "admin-001" in handler._users

    def test_default_admin_permissions(
        self, auth_runtime_settings: Any, mock_auth_dependencies: None
    ) -> None:
        handler = self._build_handler(auth_runtime_settings)
        admin_user = handler._users["admin-001"]

        assert admin_user.role.value == "admin"
        assert "*" in admin_user.permissions

    def test_default_admin_authentication(
        self, auth_runtime_settings: Any, mock_auth_dependencies: None
    ) -> None:
        handler = self._build_handler(auth_runtime_settings)

        token_pair = handler.authenticate_user("admin", "admin123")
        assert token_pair is not None
        assert token_pair.access_token == "fake.jwt.token"

    def test_default_admin_wrong_password(
        self, auth_runtime_settings: Any, mock_auth_dependencies: None
    ) -> None:
        handler = self._build_handler(auth_runtime_settings)
        assert handler.authenticate_user("admin", "wrongpassword") is None

    def test_env_bootstrap_logging(
        self,
        auth_runtime_settings: Any,
        mock_auth_dependencies: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from bot_v2.security import auth_handler

        auth_handler._jwt_module = sys.modules["jwt"]
        auth_handler._pyotp_module = sys.modules["pyotp"]

        with caplog.at_level("INFO"):
            auth_handler.AuthHandler(settings=auth_runtime_settings)

        assert any("Created default admin user" in message for message in caplog.messages)
        assert any("user_bootstrap" in message for message in caplog.messages)

    def test_env_bootstrap_with_custom_settings(
        self, monkeypatch: pytest.MonkeyPatch, mock_auth_dependencies: None
    ) -> None:
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("JWT_SECRET_KEY", "custom-secret-key")

        from bot_v2.orchestration.runtime_settings import load_runtime_settings

        settings = load_runtime_settings()
        handler = self._build_handler(settings)

        assert "admin-001" in handler._users
        assert handler.secret_key == "custom-secret-key"

    def test_missing_env_defaults_to_development(
        self, monkeypatch: pytest.MonkeyPatch, mock_auth_dependencies: None
    ) -> None:
        monkeypatch.delenv("ENV", raising=False)

        from bot_v2.orchestration.runtime_settings import load_runtime_settings

        settings = load_runtime_settings()
        handler = self._build_handler(settings)

        assert "admin-001" in handler._users
