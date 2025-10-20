"""Bootstrap and cipher initialisation error tests for SecretsManager."""

from __future__ import annotations

from typing import Any

import pytest

from bot_v2.security.secrets_manager import SecretsManager


class TestBootstrapErrors:
    """Validate bootstrap behaviour when configuration is invalid."""

    def test_missing_encryption_key_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", "")

        from bot_v2.orchestration.runtime_settings import load_runtime_settings

        settings = load_runtime_settings()

        with pytest.raises(ValueError, match="ENCRYPTION_KEY must be set in production"):
            SecretsManager(settings=settings)

    def test_invalid_encryption_key_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BOT_V2_ENCRYPTION_KEY", "invalid_key_format")

        from bot_v2.orchestration.runtime_settings import load_runtime_settings

        settings = load_runtime_settings()

        with pytest.raises(ValueError, match="Invalid encryption key"):
            SecretsManager(settings=settings)

    def test_cipher_not_initialized_error(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        manager._cipher_suite = None

        with pytest.raises(RuntimeError, match="Encryption subsystem not initialised"):
            manager._require_cipher()
