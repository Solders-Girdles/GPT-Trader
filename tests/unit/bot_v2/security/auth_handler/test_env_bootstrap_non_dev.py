"""Non-development bootstrap tests for AuthHandler."""

from __future__ import annotations

import sys

import pytest

from bot_v2.security.auth_handler import AuthHandler


class TestNonDevelopmentBootstrap:
    """Confirm non-development environments skip default-user creation."""

    def test_production_env_no_default_user(
        self, monkeypatch: pytest.MonkeyPatch, mock_auth_dependencies: None
    ) -> None:
        monkeypatch.setenv("ENV", "production")

        from bot_v2.orchestration.runtime_settings import load_runtime_settings

        settings = load_runtime_settings()
        handler = AuthHandler(settings=settings)

        assert "admin-001" not in handler._users
        assert len(handler._users) == 0

    def test_env_bootstrap_case_insensitive(
        self, monkeypatch: pytest.MonkeyPatch, mock_auth_dependencies: None
    ) -> None:
        for env_value in ["DEVELOPMENT", "Development", "development"]:
            monkeypatch.setenv("ENV", env_value)

            from bot_v2.orchestration.runtime_settings import load_runtime_settings

            settings = load_runtime_settings()

            from bot_v2.security import auth_handler

            auth_handler._jwt_module = sys.modules["jwt"]
            auth_handler._pyotp_module = sys.modules["pyotp"]

            handler = auth_handler.AuthHandler(settings=settings)
            assert "admin-001" in handler._users

    def test_non_development_env_no_bootstrap(
        self, monkeypatch: pytest.MonkeyPatch, mock_auth_dependencies: None
    ) -> None:
        test_envs = ["production", "staging", "test", "ci"]

        for env_value in test_envs:
            monkeypatch.setenv("ENV", env_value)

            from bot_v2.orchestration.runtime_settings import load_runtime_settings

            settings = load_runtime_settings()

            from bot_v2.security import auth_handler

            auth_handler._jwt_module = sys.modules["jwt"]
            auth_handler._pyotp_module = sys.modules["pyotp"]

            handler = auth_handler.AuthHandler(settings=settings)
            assert "admin-001" not in handler._users
            assert len(handler._users) == 0
