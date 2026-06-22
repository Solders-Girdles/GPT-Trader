"""Tests for durability utilities."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from gpt_trader.persistence.durability import (
    CorruptionError,
    WriteResult,
    atomic_write_file,
    atomic_write_json,
    check_sqlite_integrity,
    compute_checksum,
    read_json_with_checksum,
    verify_checksum,
)


class TestWriteResult:
    """Tests for WriteResult dataclass."""

    def test_ok_result(self) -> None:
        result = WriteResult.ok(row_id=42, checksum="abc123")
        assert result.success is True
        assert result.row_id == 42
        assert result.checksum == "abc123"
        assert result.error is None

    def test_fail_result(self) -> None:
        result = WriteResult.fail("Database locked")
        assert result.success is False
        assert result.error == "Database locked"
        assert result.row_id is None


class TestChecksum:
    """Tests for checksum functions."""

    def test_compute_checksum_bytes(self) -> None:
        checksum = compute_checksum(b"hello world")
        assert len(checksum) == 64  # SHA-256 hex
        assert checksum == compute_checksum(b"hello world")  # Deterministic

    def test_compute_checksum_string(self) -> None:
        checksum = compute_checksum("hello world")
        assert len(checksum) == 64

    def test_compute_checksum_dict(self) -> None:
        data = {"a": 1, "b": 2}
        checksum = compute_checksum(data)
        assert len(checksum) == 64

        # Same data in different order should produce same checksum
        data2 = {"b": 2, "a": 1}
        assert compute_checksum(data2) == checksum

    def test_verify_checksum_valid(self) -> None:
        data = {"key": "value"}
        checksum = compute_checksum(data)
        assert verify_checksum(data, checksum) is True

    def test_verify_checksum_invalid(self) -> None:
        data = {"key": "value"}
        assert verify_checksum(data, "wrong_checksum") is False


class TestAtomicFileWrite:
    """Tests for atomic file operations."""

    def test_atomic_write_file_bytes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.bin"
            atomic_write_file(path, b"\x00\x01\x02")

            assert path.exists()
            assert path.read_bytes() == b"\x00\x01\x02"

    def test_atomic_write_file_string(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            atomic_write_file(path, "hello world")

            assert path.exists()
            assert path.read_text() == "hello world"

    def test_atomic_write_file_creates_directories(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "test.txt"
            atomic_write_file(path, "content")

            assert path.exists()

    def test_atomic_write_file_overwrites(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.txt"
            path.write_text("original")

            atomic_write_file(path, "updated")
            assert path.read_text() == "updated"

    def test_atomic_write_json(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}

            atomic_write_json(path, data)

            with open(path) as f:
                loaded = json.load(f)
            assert loaded == data

    def test_atomic_write_json_with_checksum(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value"}

            checksum = atomic_write_json(path, data, include_checksum=True)

            assert checksum is not None
            with open(path) as f:
                loaded = json.load(f)
            assert "_checksum" in loaded
            assert loaded["_checksum"] == checksum


class TestReadJsonWithChecksum:
    """Tests for JSON reading with checksum verification."""

    def test_read_valid_json_with_checksum(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value"}
            atomic_write_json(path, data, include_checksum=True)

            loaded, valid = read_json_with_checksum(path)

            assert loaded == {"key": "value"}
            assert valid is True

    def test_read_json_without_checksum(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value"}
            atomic_write_json(path, data, include_checksum=False)

            loaded, valid = read_json_with_checksum(path)

            assert loaded == {"key": "value"}
            assert valid is True  # No checksum = valid by default

    def test_read_corrupted_json_detects_mismatch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            # Write with checksum
            data = {"key": "value"}
            atomic_write_json(path, data, include_checksum=True)

            # Corrupt the file by modifying the value but not checksum
            content = json.loads(path.read_text())
            content["key"] = "corrupted"
            path.write_text(json.dumps(content))

            loaded, valid = read_json_with_checksum(path)

            assert loaded["key"] == "corrupted"
            assert valid is False

    def test_read_invalid_json_raises_error(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            path.write_text("not valid json {{{")

            with pytest.raises(CorruptionError):
                read_json_with_checksum(path)


class TestSqliteIntegrity:
    """Tests for SQLite integrity checking."""

    def test_check_integrity_valid_database(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO test VALUES (1)")
            conn.close()

            is_ok, issues = check_sqlite_integrity(db_path)

            assert is_ok is True
            assert issues == []

    def test_check_integrity_nonexistent_database(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nonexistent.db"

            is_ok, issues = check_sqlite_integrity(db_path)

            assert is_ok is True  # Non-existent is "ok" (will be created)
            assert issues == []

    def test_check_integrity_corrupted_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "corrupted.db"
            # Write garbage that isn't a valid SQLite database
            db_path.write_bytes(b"not a sqlite database" * 100)

            is_ok, issues = check_sqlite_integrity(db_path)

            assert is_ok is False
            assert len(issues) > 0
