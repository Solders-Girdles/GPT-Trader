"""SecretsManager log redaction behaviours."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import pytest

LOGGER_NAME = "gpt_trader.security.secrets_manager"
LOG_REF_PATTERN = re.compile(r"\b(?:secret|encrypted-file):[0-9a-f]{12}\b")


def _captured_log_text(caplog: pytest.LogCaptureFixture) -> str:
    record_fragments: list[str] = [caplog.text]
    for record in caplog.records:
        record_fragments.append(record.getMessage())
        for value in record.__dict__.values():
            if isinstance(value, str):
                record_fragments.append(value)
    return "\n".join(record_fragments)


def test_file_store_and_load_logs_redact_secret_namespace_and_file_path(
    secrets_manager_with_fallback: Any,
    sample_secrets: dict[str, dict[str, Any]],
    secrets_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = secrets_manager_with_fallback
    secret_path = "brokers/coinbase"
    expected_file = secrets_dir / "brokers_coinbase.enc"

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert manager.store_secret(secret_path, sample_secrets[secret_path]) is True
        manager.clear_cache()
        assert manager.get_secret(secret_path) == sample_secrets[secret_path]
        manager.clear_cache()
        assert manager.get_secret("brokers/coinbase-missing") is None

    captured_logs = _captured_log_text(caplog)
    raw_secret_digest = hashlib.sha256(secret_path.encode("utf-8")).hexdigest()[:12]
    raw_file_digest = hashlib.sha256(str(expected_file).encode("utf-8")).hexdigest()[:12]

    assert secret_path not in captured_logs
    assert str(expected_file) not in captured_logs
    assert "brokers_coinbase.enc" not in captured_logs
    assert "test-api-key-123" not in captured_logs
    assert "test-secret-456" not in captured_logs
    assert "secret:" in captured_logs
    assert "encrypted-file:" in captured_logs
    assert f"secret:{raw_secret_digest}" not in captured_logs
    assert f"encrypted-file:{raw_file_digest}" not in captured_logs
    assert LOG_REF_PATTERN.search(captured_logs) is not None
    assert "Successfully wrote encrypted secret file" in caplog.messages
    assert "Successfully decrypted and loaded secret" in caplog.messages
    assert "Encrypted secret file does not exist" in caplog.messages


def test_file_read_error_logs_redact_secret_namespace_and_os_error_path(
    secrets_manager_with_fallback: Any,
    secrets_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = secrets_manager_with_fallback
    secret_path = "brokers/coinbase"
    expected_file = secrets_dir / "brokers_coinbase.enc"

    assert manager.store_secret(secret_path, {"api_key": "secret-value"}) is True
    manager.clear_cache()

    original_read_bytes = Path.read_bytes

    def failing_read_bytes(self: Path) -> bytes:
        if self.name.endswith(".enc"):
            raise OSError(f"Read error for {self}")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        assert manager.get_secret(secret_path) is None

    captured_logs = _captured_log_text(caplog)
    raw_secret_digest = hashlib.sha256(secret_path.encode("utf-8")).hexdigest()[:12]
    raw_file_digest = hashlib.sha256(str(expected_file).encode("utf-8")).hexdigest()[:12]

    assert secret_path not in captured_logs
    assert str(expected_file) not in captured_logs
    assert "brokers_coinbase.enc" not in captured_logs
    assert "secret-value" not in captured_logs
    assert f"secret:{raw_secret_digest}" not in captured_logs
    assert f"encrypted-file:{raw_file_digest}" not in captured_logs
    assert LOG_REF_PATTERN.search(captured_logs) is not None
    assert "Failed to read encrypted secret file" in caplog.messages
    assert "OSError" in captured_logs
