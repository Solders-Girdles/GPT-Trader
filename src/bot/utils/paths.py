"""
Consolidated path and file utilities for GPT-Trader.

This module provides centralized path handling and file management
utilities that were previously duplicated across multiple files.
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path


class PathUtils:
    """Path and directory utilities."""

    @staticmethod
    def ensure_directory(path: str | Path) -> Path:
        """Ensure directory exists, create if it doesn't.

        Args:
            path: Directory path to ensure

        Returns:
            Path object
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @staticmethod
    def create_timestamped_directory(base_path: str | Path, prefix: str = "") -> Path:
        """Create directory with timestamp suffix.

        Args:
            base_path: Base directory path
            prefix: Optional prefix for directory name

        Returns:
            Path to created directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{prefix}_{timestamp}" if prefix else timestamp

        dir_path = Path(base_path) / dir_name
        return PathUtils.ensure_directory(dir_path)

    @staticmethod
    def get_project_root() -> Path:
        """Get project root directory.

        Returns:
            Path to project root
        """
        # Look for pyproject.toml or setup.py to identify project root
        current = Path.cwd()

        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
                return current
            current = current.parent

        # Fallback to current directory
        return Path.cwd()

    @staticmethod
    def get_data_directory() -> Path:
        """Get data directory, create if doesn't exist.

        Returns:
            Path to data directory
        """
        data_dir = PathUtils.get_project_root() / "data"
        return PathUtils.ensure_directory(data_dir)

    @staticmethod
    def get_logs_directory() -> Path:
        """Get logs directory, create if doesn't exist.

        Returns:
            Path to logs directory
        """
        logs_dir = PathUtils.get_project_root() / "logs"
        return PathUtils.ensure_directory(logs_dir)

    @staticmethod
    def get_config_directory() -> Path:
        """Get config directory, create if doesn't exist.

        Returns:
            Path to config directory
        """
        config_dir = PathUtils.get_project_root() / "config"
        return PathUtils.ensure_directory(config_dir)


class FileUtils:
    """File management utilities."""

    @staticmethod
    def safe_copy(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
        """Safely copy file with error handling.

        Args:
            src: Source file path
            dst: Destination file path
            overwrite: Whether to overwrite existing files

        Returns:
            Path to destination file

        Raises:
            ValueError: If operation fails
        """
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise ValueError(f"Source file does not exist: {src}")

        if not src_path.is_file():
            raise ValueError(f"Source is not a file: {src}")

        if dst_path.exists() and not overwrite:
            raise ValueError(f"Destination file already exists: {dst}")

        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return dst_path
        except Exception as e:
            raise ValueError(f"Failed to copy {src} to {dst}: {e}") from e

    @staticmethod
    def safe_move(src: str | Path, dst: str | Path, overwrite: bool = False) -> Path:
        """Safely move file with error handling.

        Args:
            src: Source file path
            dst: Destination file path
            overwrite: Whether to overwrite existing files

        Returns:
            Path to destination file

        Raises:
            ValueError: If operation fails
        """
        src_path = Path(src)
        dst_path = Path(dst)

        if not src_path.exists():
            raise ValueError(f"Source file does not exist: {src}")

        if dst_path.exists() and not overwrite:
            raise ValueError(f"Destination file already exists: {dst}")

        try:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            return dst_path
        except Exception as e:
            raise ValueError(f"Failed to move {src} to {dst}: {e}") from e

    @staticmethod
    def cleanup_old_files(directory: str | Path, max_age_days: int = 30, pattern: str = "*") -> int:
        """Clean up old files in directory.

        Args:
            directory: Directory to clean up
            max_age_days: Maximum age in days
            pattern: File pattern to match

        Returns:
            Number of files cleaned up
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0

        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                try:
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                except OSError:
                    continue  # Skip files we can't delete

        return cleaned_count

    @staticmethod
    def get_file_size(path: str | Path) -> int:
        """Get file size in bytes.

        Args:
            path: File path

        Returns:
            File size in bytes

        Raises:
            ValueError: If file doesn't exist
        """
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"File does not exist: {path}")
        return file_path.stat().st_size

    @staticmethod
    def is_file_recent(path: str | Path, max_age_hours: int = 24) -> bool:
        """Check if file was modified recently.

        Args:
            path: File path
            max_age_hours: Maximum age in hours

        Returns:
            True if file is recent, False otherwise
        """
        file_path = Path(path)
        if not file_path.exists():
            return False

        cutoff_time = time.time() - (max_age_hours * 60 * 60)
        return file_path.stat().st_mtime > cutoff_time


class FileFinder:
    """Utilities for finding files."""

    @staticmethod
    def find_files_by_pattern(
        directory: str | Path, pattern: str, recursive: bool = True
    ) -> list[Path]:
        """Find files matching pattern.

        Args:
            directory: Directory to search
            pattern: Glob pattern to match
            recursive: Whether to search recursively

        Returns:
            List of matching file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return []

        if recursive:
            return sorted(dir_path.rglob(pattern))
        else:
            return sorted(dir_path.glob(pattern))

    @staticmethod
    def find_latest_file(directory: str | Path, pattern: str = "*") -> Path | None:
        """Find most recently modified file matching pattern.

        Args:
            directory: Directory to search
            pattern: Glob pattern to match

        Returns:
            Path to latest file or None if no matches
        """
        files = FileFinder.find_files_by_pattern(directory, pattern, recursive=False)
        if not files:
            return None

        return max(files, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def find_files_by_extension(
        directory: str | Path, extension: str, recursive: bool = True
    ) -> list[Path]:
        """Find files with specific extension.

        Args:
            directory: Directory to search
            extension: File extension (with or without dot)
            recursive: Whether to search recursively

        Returns:
            List of matching file paths
        """
        if not extension.startswith("."):
            extension = "." + extension

        pattern = f"*{extension}"
        return FileFinder.find_files_by_pattern(directory, pattern, recursive)

    @staticmethod
    def find_empty_files(directory: str | Path, recursive: bool = True) -> list[Path]:
        """Find empty files in directory.

        Args:
            directory: Directory to search
            recursive: Whether to search recursively

        Returns:
            List of empty file paths
        """
        all_files = FileFinder.find_files_by_pattern(directory, "*", recursive)
        return [f for f in all_files if f.is_file() and f.stat().st_size == 0]


class ArchiveUtils:
    """Archive and compression utilities."""

    @staticmethod
    def create_backup(source: str | Path, backup_dir: str | Path | None = None) -> Path:
        """Create timestamped backup of file or directory.

        Args:
            source: Source path to backup
            backup_dir: Directory for backup (defaults to source parent)

        Returns:
            Path to backup
        """
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"Source does not exist: {source}")

        if backup_dir is None:
            backup_dir = source_path.parent / "backups"
        else:
            backup_dir = Path(backup_dir)

        PathUtils.ensure_directory(backup_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.name}_{timestamp}"
        backup_path = backup_dir / backup_name

        if source_path.is_file():
            return FileUtils.safe_copy(source_path, backup_path)
        else:
            shutil.copytree(source_path, backup_path)
            return backup_path
