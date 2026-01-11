"""Shared fixtures for secrets_manager tests."""

from __future__ import annotations

from typing import Any

import pytest
from cryptography.fernet import Fernet

from gpt_trader.app.config import BotConfig

pytest_plugins = [
    "tests.unit.gpt_trader.security.secrets_manager.crypto_fixtures",
    "tests.unit.gpt_trader.security.secrets_manager.file_fixtures",
    "tests.unit.gpt_trader.security.secrets_manager.vault_fixtures",
]


@pytest.fixture
def secrets_bot_config(bot_config_factory) -> BotConfig:
    """Bot config tailored for secrets_manager tests."""
    return bot_config_factory()


@pytest.fixture
def secrets_runtime_settings(
    secrets_bot_config: BotConfig, monkeypatch: pytest.MonkeyPatch
) -> BotConfig:
    """Backward-compatible fixture for tests expecting RuntimeSettings.

    Returns BotConfig (the replacement for RuntimeSettings).
    Sets up required environment variables for encryption.
    """
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", Fernet.generate_key().decode())
    return secrets_bot_config


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Advance time deterministically without real sleep."""
    import time

    current = {"value": time.time()}

    def advance(seconds: float) -> None:
        current["value"] += seconds

    monkeypatch.setattr(time, "time", lambda: current["value"])
    monkeypatch.setattr(time, "sleep", lambda seconds: advance(seconds))

    return advance
