"""File I/O error handling tests for SecretsManager."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TestFileIOErrors:
    """Exercise corrupted files, permission issues, and disk failures."""

    def test_corrupted_file_detection(
        self, secrets_manager_with_fallback: Any, corrupted_file_data: Path
    ) -> None:
        manager = secrets_manager_with_fallback

        assert manager.get_secret("test_corrupted") is None

    def test_invalid_json_payload(
        self, secrets_manager_with_fallback: Any, secrets_dir: Path
    ) -> None:
        manager = secrets_manager_with_fallback

        invalid_file = secrets_dir / "test_invalid.enc"
        invalid_file.write_bytes(b"invalid_encrypted_data")

        assert manager.get_secret("test_invalid") is None

    def test_file_permission_error(
        self, secrets_manager_with_fallback: Any, secrets_dir: Path, monkeypatch: Any
    ) -> None:
        manager = secrets_manager_with_fallback

        original_write_bytes = Path.write_bytes

        def failing_write_bytes(self, data):
            if self.name.endswith(".enc"):
                raise PermissionError("Permission denied")
            return original_write_bytes(self, data)

        monkeypatch.setattr(Path, "write_bytes", failing_write_bytes)

        assert manager.store_secret("test/permission", {"key": "value"}) is False

    def test_file_read_error(
        self, secrets_manager_with_fallback: Any, secrets_dir: Path, monkeypatch: Any
    ) -> None:
        manager = secrets_manager_with_fallback

        manager.store_secret("test/read_error", {"key": "value"})
        manager.clear_cache()

        original_read_bytes = Path.read_bytes

        def failing_read_bytes(self):
            if self.name.endswith(".enc"):
                raise OSError("Read error")
            return original_read_bytes(self)

        monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)

        assert manager.get_secret("test/read_error") is None

    def test_disk_space_error(
        self, secrets_manager_with_fallback: Any, secrets_dir: Path, monkeypatch: Any
    ) -> None:
        manager = secrets_manager_with_fallback

        original_write_bytes = Path.write_bytes

        def disk_full_error(self, data):
            if self.name.endswith(".enc"):
                raise OSError("No space left on device")
            return original_write_bytes(self, data)

        monkeypatch.setattr(Path, "write_bytes", disk_full_error)

        assert manager.store_secret("test/disk_full", {"key": "value"}) is False
