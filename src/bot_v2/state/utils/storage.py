"""Shared storage backend utilities for state management.

Consolidates common storage patterns used across checkpoint and backup systems.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


class AtomicFileStorage:
    """Provides atomic file write operations using temp file + rename pattern.

    This ensures that files are never partially written or corrupted if
    the write operation is interrupted.
    """

    def __init__(self, base_dir: str | Path):
        """Initialize atomic file storage.

        Args:
            base_dir: Base directory for storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_atomic(self, filename: str, data: bytes) -> Path:
        """Write data atomically to a file.

        Args:
            filename: Target filename (relative to base_dir)
            data: Bytes to write

        Returns:
            Path to written file
        """
        target_path = self.base_dir / filename

        # Write to temp file first
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.base_dir, prefix=f".tmp_{filename}_", suffix=".tmp"
        )

        try:
            # Write data to temp file
            with os.fdopen(temp_fd, "wb") as f:
                f.write(data)

            # Atomic rename
            temp_path_obj = Path(temp_path)
            temp_path_obj.replace(target_path)

            return target_path

        except Exception:
            # Clean up temp file on error
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def read(self, filename: str) -> bytes:
        """Read data from file.

        Args:
            filename: Filename to read (relative to base_dir)

        Returns:
            File contents as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = self.base_dir / filename
        with open(file_path, "rb") as f:
            return f.read()

    def exists(self, filename: str) -> bool:
        """Check if file exists.

        Args:
            filename: Filename to check

        Returns:
            True if file exists
        """
        return (self.base_dir / filename).exists()

    def delete(self, filename: str) -> bool:
        """Delete file if it exists.

        Args:
            filename: Filename to delete

        Returns:
            True if file was deleted, False if it didn't exist
        """
        file_path = self.base_dir / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_files(self, pattern: str = "*") -> list[Path]:
        """List files matching pattern.

        Args:
            pattern: Glob pattern (default: all files)

        Returns:
            List of matching file paths
        """
        return sorted(self.base_dir.glob(pattern))

    def get_file_size(self, filename: str) -> int:
        """Get file size in bytes.

        Args:
            filename: Filename to check

        Returns:
            File size in bytes

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = self.base_dir / filename
        return file_path.stat().st_size


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_age_seconds(path: str | Path) -> float:
    """Get age of file in seconds.

    Args:
        path: File path

    Returns:
        Age in seconds since last modification
    """
    import time

    file_path = Path(path)
    return time.time() - file_path.stat().st_mtime