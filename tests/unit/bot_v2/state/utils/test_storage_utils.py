"""Unit tests for atomic storage helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from bot_v2.state.utils.storage import (
    AtomicFileStorage,
    ensure_directory,
    get_file_age_seconds,
)


def test_atomic_write_read_and_delete(tmp_path: Path) -> None:
    """Atomic writes should persist data and support full lifecycle operations."""
    storage = AtomicFileStorage(tmp_path)
    written_path = storage.write_atomic("payload.bin", b"data")

    assert written_path.read_bytes() == b"data"
    assert storage.exists("payload.bin") is True
    assert storage.read("payload.bin") == b"data"
    assert storage.get_file_size("payload.bin") == len(b"data")

    stored_files = storage.list_files("*.bin")
    assert stored_files == [written_path]

    assert storage.delete("payload.bin") is True
    assert storage.delete("payload.bin") is False


def test_atomic_write_cleans_up_temp_files_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failures during rename should remove temporary artifacts."""
    storage = AtomicFileStorage(tmp_path)

    def fail_replace(self: Path, _target: Path) -> None:
        raise RuntimeError("rename failure")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(RuntimeError):
        storage.write_atomic("failure.bin", b"oops")

    temp_files = list(tmp_path.glob(".tmp_failure.bin_*"))
    assert temp_files == []
    assert not (tmp_path / "failure.bin").exists()


def test_atomic_write_cleanup_handles_unlink_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup errors should not mask the original failure."""
    storage = AtomicFileStorage(tmp_path)

    def fail_replace(self: Path, _target: Path) -> None:
        raise RuntimeError("rename failure")

    def fail_unlink(self: Path, missing_ok: bool = False) -> None:  # noqa: ARG001
        raise OSError("unlink failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(RuntimeError):
        storage.write_atomic("cleanup.bin", b"payload")


def test_atomic_read_missing_file_raises(tmp_path: Path) -> None:
    """Reading a missing file should propagate FileNotFoundError."""
    storage = AtomicFileStorage(tmp_path)
    with pytest.raises(FileNotFoundError):
        storage.read("missing.bin")


def test_ensure_directory_creates_path(tmp_path: Path) -> None:
    """ensure_directory should create nested paths when absent."""
    nested = tmp_path / "a" / "b"
    created = ensure_directory(nested)
    assert created == nested
    assert created.exists()
    assert created.is_dir()


def test_get_file_age_seconds(tmp_path: Path) -> None:
    """File age utility should return a non-negative age for existing files."""
    file_path = tmp_path / "age.txt"
    file_path.write_text("data")

    age = get_file_age_seconds(file_path)

    assert age >= 0
