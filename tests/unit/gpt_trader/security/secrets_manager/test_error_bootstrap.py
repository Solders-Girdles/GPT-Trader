"""Bootstrap and cipher initialisation error tests for SecretsManager."""

from __future__ import annotations

from typing import Any

import pytest

from gpt_trader.security.secrets_manager import SecretsManager


class TestBootstrapErrors:
    """Validate bootstrap behaviour when configuration is invalid."""

    def test_missing_encryption_key_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("GPT_TRADER_ENCRYPTION_KEY", raising=False)

        with pytest.raises(ValueError, match="ENCRYPTION_KEY must be set in production"):
            SecretsManager(vault_enabled=False)

    def test_invalid_encryption_key_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", "invalid_key_format")

        with pytest.raises(ValueError, match="Invalid encryption key"):
            SecretsManager(vault_enabled=False)

    def test_cipher_not_initialized_error(self, secrets_manager_with_fallback: Any) -> None:
        manager = secrets_manager_with_fallback
        manager._cipher_suite = None

        with pytest.raises(RuntimeError, match="Encryption subsystem not initialised"):
            manager._require_cipher()
