from __future__ import annotations

import pytest
from cryptography.fernet import Fernet


@pytest.fixture
def deterministic_fernet() -> Fernet:
    """Deterministic Fernet instance for predictable tests."""
    # Use a fixed key for deterministic encryption/decryption
    test_key = b"Z9mB9nKp8sVqLyWsC5uE4oHj7gFdR2aL3xYiN6wTzQc="
    return Fernet(test_key)


@pytest.fixture
def patched_require_fernet(deterministic_fernet: Fernet, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch _require_fernet to return deterministic instance."""
    from gpt_trader.security import secrets_manager

    def mock_require_fernet() -> type[Fernet]:
        return type(deterministic_fernet)

    monkeypatch.setattr(secrets_manager, "_require_fernet", mock_require_fernet)
