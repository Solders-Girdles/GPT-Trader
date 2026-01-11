from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet

from gpt_trader.app.config import BotConfig


@pytest.fixture
def secrets_dir(tmp_path: Path) -> Path:
    """Temporary secrets directory for file-based storage tests."""
    secrets_dir = tmp_path / ".gpt_trader" / "secrets"
    secrets_dir.mkdir(parents=True, exist_ok=True)
    return secrets_dir


@pytest.fixture
def mock_secrets_storage(secrets_dir: Path) -> dict[str, dict[str, Any]]:
    """In-memory secrets storage for testing."""
    return {}


@pytest.fixture
def sample_secrets() -> dict[str, dict[str, Any]]:
    """Sample secrets data for testing."""
    return {
        "brokers/coinbase": {
            "api_key": "test-api-key-123",
            "secret": "test-secret-456",
            "passphrase": "test-passphrase",
        },
        "brokers/binance": {
            "api_key": "binance-api-key-789",
            "secret": "binance-secret-012",
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "username": "testuser",
            "password": "testpassword",
        },
    }


@pytest.fixture
def secrets_manager_with_fallback(
    secrets_bot_config: BotConfig,
    patched_require_fernet: None,
    monkeypatch: pytest.MonkeyPatch,
    secrets_dir: Path,
) -> Any:
    """SecretsManager instance with file fallback (no vault)."""
    from gpt_trader.security.secrets_manager import SecretsManager

    # Force vault to be disabled
    monkeypatch.setenv("VAULT_TOKEN", "")
    # Set required env vars
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("GPT_TRADER_ENCRYPTION_KEY", Fernet.generate_key().decode())

    return SecretsManager(vault_enabled=False, config=secrets_bot_config, secrets_dir=secrets_dir)


@pytest.fixture
def corrupted_file_data(secrets_dir: Path) -> Path:
    """Create a corrupted encrypted file for testing error handling."""
    corrupted_file = secrets_dir / "test_corrupted.enc"
    corrupted_file.write_bytes(b"corrupted-data-that-cannot-be-decrypted")
    return corrupted_file
