"""
Durability utilities for production-grade persistence.

Provides:
- Atomic file writes (write-rename pattern)
- Checksum calculation and validation
- Database integrity checking
- Write operation wrappers with error handling
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="durability")


class PersistenceError(Exception):
    """Base exception for persistence errors."""

    pass


class WriteError(PersistenceError):
    """Raised when a write operation fails."""

    pass


class CorruptionError(PersistenceError):
    """Raised when data corruption is detected."""

    pass


class RecoveryError(PersistenceError):
    """Raised when recovery fails."""

    pass


def _sqlite_sidecar_paths(database_path: Path) -> tuple[Path, Path]:
    """Return SQLite WAL sidecar paths for a database path."""
    return (
        database_path.with_name(f"{database_path.name}-wal"),
        database_path.with_name(f"{database_path.name}-shm"),
    )


def _remove_sqlite_sidecars(database_path: Path) -> bool:
    removed = True
    for sidecar_path in _sqlite_sidecar_paths(database_path):
        try:
            sidecar_path.unlink()
        except FileNotFoundError:
            continue
        except OSError as e:
            logger.error(
                "Failed to remove SQLite sidecar",
                operation="database_repair",
                sidecar=str(sidecar_path),
                error=str(e),
            )
            removed = False
    return removed


def _remove_sqlite_file_set(database_path: Path) -> bool:
    removed = _remove_sqlite_sidecars(database_path)
    try:
        database_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.error(
            "Failed to remove SQLite database file",
            operation="database_repair",
            path=str(database_path),
            error=str(e),
        )
        removed = False
    return removed


def _copy_sqlite_file_set(database_path: Path, backup_path: Path) -> bool:
    if not _remove_sqlite_sidecars(backup_path):
        return False
    try:
        shutil.copy2(database_path, backup_path)
        for source_sidecar, backup_sidecar in zip(
            _sqlite_sidecar_paths(database_path), _sqlite_sidecar_paths(backup_path), strict=True
        ):
            if source_sidecar.exists():
                shutil.copy2(source_sidecar, backup_sidecar)
    except OSError as e:
        logger.error(
            "Failed to backup corrupted database",
            operation="database_repair",
            error=str(e),
        )
        return False
    return True


def _quote_sqlite_identifier(identifier: str) -> str:
    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'


def _sqlite_table_columns(
    connection: sqlite3.Connection,
    table_name: str,
    *,
    schema: str = "main",
) -> list[str]:
    quoted_schema = _quote_sqlite_identifier(schema)
    quoted_table = _quote_sqlite_identifier(table_name)
    cursor = connection.execute(f"PRAGMA {quoted_schema}.table_info({quoted_table})")
    return [row[1] for row in cursor.fetchall()]


def _log_sqlite_recovery_skip(table_name: str) -> None:
    logger.warning(
        "Could not recover table without common columns",
        operation="database_repair",
        table=table_name,
    )


def _copy_common_sqlite_columns(
    connection: sqlite3.Connection,
    table_name: str,
    *,
    source_schema: str,
) -> bool:
    destination_columns = _sqlite_table_columns(connection, table_name)
    source_columns = set(_sqlite_table_columns(connection, table_name, schema=source_schema))
    common_columns = [column for column in destination_columns if column in source_columns]
    if not common_columns:
        _log_sqlite_recovery_skip(table_name)
        return False

    quoted_table = _quote_sqlite_identifier(table_name)
    quoted_source_schema = _quote_sqlite_identifier(source_schema)
    quoted_columns = ", ".join(_quote_sqlite_identifier(column) for column in common_columns)
    connection.execute(
        f"INSERT OR IGNORE INTO {quoted_table} ({quoted_columns}) "
        f"SELECT {quoted_columns} FROM {quoted_source_schema}.{quoted_table}"
    )
    return True


@dataclass(frozen=True)
class WriteResult:
    """Result of a write operation."""

    success: bool
    row_id: int | None = None
    checksum: str | None = None
    error: str | None = None

    @classmethod
    def ok(cls, row_id: int | None = None, checksum: str | None = None) -> WriteResult:
        """Create successful result."""
        return cls(success=True, row_id=row_id, checksum=checksum)

    @classmethod
    def fail(cls, error: str) -> WriteResult:
        """Create failed result."""
        return cls(success=False, error=error)


def compute_checksum(data: bytes | str | dict[str, Any]) -> str:
    """
    Compute SHA-256 checksum for data.

    Args:
        data: Bytes, string, or dict to checksum

    Returns:
        Hex-encoded SHA-256 checksum
    """
    if isinstance(data, dict):
        # Canonical JSON for deterministic hashing
        data = json.dumps(data, sort_keys=True, separators=(",", ":"))
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_checksum(data: bytes | str | dict[str, Any], expected: str) -> bool:
    """
    Verify data matches expected checksum.

    Args:
        data: Data to verify
        expected: Expected checksum

    Returns:
        True if checksum matches
    """
    actual = compute_checksum(data)
    return actual == expected


def _fsync_parent_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        logger.debug("Skipping directory fsync because O_DIRECTORY is unavailable")
        return

    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | directory_flag)
    except OSError as e:
        logger.debug("Skipping directory fsync for %s: %s", path.parent, e)
        return

    try:
        os.fsync(dir_fd)
    except OSError as e:
        logger.debug("Directory fsync failed for %s: %s", path.parent, e)
    finally:
        os.close(dir_fd)


def atomic_write_file(path: Path, content: bytes | str, *, fsync: bool = True) -> None:
    """
    Write file atomically using a write-replace pattern.

    Creates a temporary file in the same directory, writes content,
    optionally fsyncs, then atomically replaces the target path.

    Args:
        path: Target file path
        content: Content to write
        fsync: Whether to fsync before rename (default: True for durability)

    Raises:
        WriteError: If write fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "wb" if isinstance(content, bytes) else "w"
    encoding = None if isinstance(content, bytes) else "utf-8"

    try:
        # Create temp file in same directory for atomic replacement
        fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, mode, encoding=encoding) as f:
                f.write(content)
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())

            # Atomic replacement; unlike os.rename, this overwrites on Windows.
            os.replace(temp_path, path)

            # Fsync directory to persist the replacement when supported.
            if fsync:
                _fsync_parent_directory(path)

        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError as cleanup_error:
                logger.debug("Failed to cleanup temp file %s: %s", temp_path, cleanup_error)
            raise

    except Exception as e:
        raise WriteError(f"Failed to write {path}: {e}") from e


def atomic_write_json(
    path: Path,
    data: dict[str, Any] | list[Any],
    *,
    include_checksum: bool = False,
) -> str | None:
    """
    Write JSON file atomically.

    Args:
        path: Target file path
        data: Data to serialize
        include_checksum: If True, adds _checksum field to data

    Returns:
        Checksum if include_checksum is True, else None

    Raises:
        WriteError: If write fails
    """
    checksum = None

    if include_checksum and isinstance(data, dict):
        # Compute checksum of data without _checksum field
        data_copy = {k: v for k, v in data.items() if k != "_checksum"}
        checksum = compute_checksum(data_copy)
        data = {**data_copy, "_checksum": checksum}

    content = json.dumps(data, indent=2, sort_keys=True)
    atomic_write_file(path, content)

    return checksum


def read_json_with_checksum(path: Path) -> tuple[dict[str, Any], bool]:
    """
    Read JSON file and verify embedded checksum.

    Args:
        path: JSON file path

    Returns:
        Tuple of (data without _checksum, checksum_valid)

    Raises:
        CorruptionError: If file is corrupted or unreadable
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise CorruptionError(f"Invalid JSON in {path}: {e}") from e
    except OSError as e:
        raise CorruptionError(f"Cannot read {path}: {e}") from e

    if not isinstance(data, dict):
        return data, True  # No checksum for non-dict

    stored_checksum = data.pop("_checksum", None)
    if stored_checksum is None:
        return data, True  # No checksum stored

    expected = compute_checksum(data)
    valid = stored_checksum == expected

    if not valid:
        logger.warning(
            "Checksum mismatch in file",
            operation="checksum_verify",
            path=str(path),
            stored=stored_checksum[:16],
            expected=expected[:16],
        )

    return data, valid


def check_sqlite_integrity(database_path: Path) -> tuple[bool, list[str]]:
    """
    Run SQLite integrity check on database.

    Args:
        database_path: Path to SQLite database

    Returns:
        Tuple of (integrity_ok, list of issues)
    """
    issues: list[str] = []

    if not database_path.exists():
        return True, []  # Non-existent is "ok" (will be created)

    try:
        conn = sqlite3.connect(str(database_path))
        try:
            cursor = conn.execute("PRAGMA integrity_check")
            results = cursor.fetchall()

            for row in results:
                result = row[0]
                if result != "ok":
                    issues.append(result)

            return len(issues) == 0, issues
        finally:
            conn.close()
    except sqlite3.Error as e:
        return False, [f"Database error: {e}"]


def repair_sqlite_database(database_path: Path, backup_path: Path | None = None) -> bool:
    """
    Attempt to repair a corrupted SQLite database.

    Uses the ".recover" command if available (SQLite 3.29+),
    otherwise dumps and recreates the database.

    Args:
        database_path: Path to corrupted database
        backup_path: Optional path to backup corrupted database

    Returns:
        True if repair succeeded
    """
    if not database_path.exists():
        return True

    # Backup corrupted database
    if backup_path is None:
        backup_path = database_path.with_suffix(".corrupted")

    if not _copy_sqlite_file_set(database_path, backup_path):
        return False

    logger.info(
        "Backed up corrupted database",
        operation="database_repair",
        original=str(database_path),
        backup=str(backup_path),
    )

    # Try to recover data via an ATTACH strategy to avoid N+1 queries and memory loads
    temp_db_path = database_path.with_suffix(".tmp")
    if not _remove_sqlite_file_set(temp_db_path):
        return False

    try:
        conn = sqlite3.connect(str(temp_db_path))
        recovered_count = 0
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            # Initialize with known schemas to preserve them
            schema_path = Path(__file__).parent / "schema.sql"
            if schema_path.exists():
                conn.executescript(schema_path.read_text())

            # Attach corrupted backup to stream data directly at SQLite level
            conn.execute("ATTACH DATABASE ? AS corrupted", (str(backup_path),))

            cursor = conn.execute(
                "SELECT name, sql FROM corrupted.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = cursor.fetchall()

            for table_name, table_sql in tables:
                try:
                    # If the table wasn't created by schema.sql, recreate it from the original schema
                    if table_sql:
                        try:
                            conn.execute(table_sql)
                        except sqlite3.OperationalError:
                            # Table may already exist from schema.sql execution above
                            pass

                    if _copy_common_sqlite_columns(conn, table_name, source_schema="corrupted"):
                        recovered_count += 1
                except sqlite3.Error:
                    logger.warning(
                        "Could not recover table",
                        operation="database_repair",
                        table=table_name,
                    )
            # Commit the copied data to the new database before closing
            conn.commit()
            conn.execute("PRAGMA main.wal_checkpoint(TRUNCATE)")
        finally:
            try:
                conn.execute("DETACH DATABASE corrupted")
            except sqlite3.Error:
                pass
            conn.close()

        if not _remove_sqlite_sidecars(temp_db_path):
            raise RecoveryError("Could not remove recovered database WAL sidecars")
        if not _remove_sqlite_sidecars(database_path):
            raise RecoveryError("Could not remove original database WAL sidecars")

        # Replace corrupted database with recovered temporary database
        temp_db_path.replace(database_path)

        logger.info(
            "Database recreated",
            operation="database_repair",
            tables_recovered=recovered_count,
        )
        return True

    except Exception as e:
        logger.error(
            "Database repair failed",
            operation="database_repair",
            error=str(e),
        )
        # Cleanup temp database on failure
        _remove_sqlite_file_set(temp_db_path)
        return False


__all__ = [
    "CorruptionError",
    "PersistenceError",
    "RecoveryError",
    "WriteError",
    "WriteResult",
    "atomic_write_file",
    "atomic_write_json",
    "check_sqlite_integrity",
    "compute_checksum",
    "read_json_with_checksum",
    "repair_sqlite_database",
    "verify_checksum",
]
