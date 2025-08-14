#!/usr/bin/env python3
"""Backtest File Retention and Cleanup Script

SOT-PRE-009: Add file retention policy for backtests

Automated cleanup of backtest CSV files with intelligent retention policies:
- Keep files less than 7 days old
- Keep top 10 best performing backtests (by Sharpe ratio) regardless of age
- Archive important results to data/backtests/archive/
- Delete others older than 7 days
- File size monitoring and compression
- Cron-compatible with logging and dry-run mode

Usage:
    python scripts/cleanup_backtests.py --dry-run  # Preview actions
    python scripts/cleanup_backtests.py            # Execute cleanup
    python scripts/cleanup_backtests.py --archive-only  # Only archive, no delete
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from bot.logging import _ensure_root_config
except ImportError:
    # Fallback logging setup if bot.logging not available
    def _ensure_root_config() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


class BacktestFile(NamedTuple):
    """Represents a backtest file with its metadata."""

    path: Path
    created: datetime
    size_bytes: int
    file_type: str  # 'optimization' or 'portfolio'
    sharpe_ratio: float | None
    cagr: float | None
    is_important: bool = False


class BacktestCleanup:
    """Manages backtest file cleanup with intelligent retention policies."""

    def __init__(self, base_path: Path, dry_run: bool = False):
        self.base_path = base_path
        self.dry_run = dry_run
        self.archive_path = base_path / "archive"
        self.logger = logging.getLogger("backtest_cleanup")

        # Retention policy configuration
        self.retention_days = 7
        self.keep_best_count = 10
        self.size_warn_gb = 1.0
        self.size_archive_gb = 2.0

        # Stats tracking
        self.stats = {
            "files_scanned": 0,
            "files_kept_recent": 0,
            "files_kept_best": 0,
            "files_archived": 0,
            "files_deleted": 0,
            "bytes_cleaned": 0,
            "bytes_archived": 0,
        }

    def setup_directories(self) -> None:
        """Ensure required directories exist."""
        if not self.dry_run:
            self.archive_path.mkdir(exist_ok=True)
            self.logger.info(f"Archive directory ready: {self.archive_path}")

    def parse_timestamp_from_filename(self, filename: str) -> datetime | None:
        """Extract timestamp from backtest filename.

        Expected formats:
        - OPT_trend_breakout_20250809-164511_overall.csv
        - PORT_enhanced_trend_breakout_20250810-185209.csv
        """
        try:
            # Find the timestamp pattern YYYYMMDD-HHMMSS
            import re

            pattern = r"(\d{8})-(\d{6})"
            match = re.search(pattern, filename)
            if match:
                date_str, time_str = match.groups()
                return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
        except Exception as e:
            self.logger.warning(f"Failed to parse timestamp from {filename}: {e}")
        return None

    def extract_metrics_from_csv(self, file_path: Path) -> tuple[float | None, float | None]:
        """Extract Sharpe ratio and CAGR from CSV file.

        Returns:
            Tuple of (sharpe_ratio, cagr) or (None, None) if not found
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if not rows:
                    return None, None

                # For optimization files, get best row
                if "mean_sharpe" in rows[0]:
                    best_row = max(rows, key=lambda r: float(r.get("mean_sharpe", 0)))
                    sharpe = float(best_row.get("mean_sharpe", 0))
                    cagr = float(best_row.get("mean_cagr", 0))
                    return sharpe, cagr

                # For portfolio files, calculate from equity curve
                elif "equity" in rows[0]:
                    equities = [float(r["equity"]) for r in rows if r["equity"]]
                    if len(equities) > 1:
                        returns = [
                            (equities[i] - equities[i - 1]) / equities[i - 1]
                            for i in range(1, len(equities))
                            if equities[i - 1] != 0
                        ]

                        if returns:
                            import statistics

                            mean_return = statistics.mean(returns)
                            std_return = statistics.stdev(returns) if len(returns) > 1 else 0
                            sharpe = mean_return / std_return if std_return != 0 else 0

                            # Annualized CAGR approximation
                            total_return = (equities[-1] - equities[0]) / equities[0]
                            periods = (
                                len(equities) / 252
                            )  # Assume daily data, 252 trading days/year
                            cagr = (1 + total_return) ** (1 / periods) - 1 if periods > 0 else 0

                            return sharpe, cagr
        except Exception as e:
            self.logger.debug(f"Failed to extract metrics from {file_path}: {e}")

        return None, None

    def scan_backtest_files(self) -> list[BacktestFile]:
        """Scan and categorize all backtest files."""
        files = []
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for file_path in self.base_path.glob("*.csv"):
            if not file_path.is_file():
                continue

            self.stats["files_scanned"] += 1

            # Get file metadata
            stat = file_path.stat()
            created = datetime.fromtimestamp(stat.st_mtime)
            size_bytes = stat.st_size

            # Parse timestamp from filename (more reliable than file mtime)
            parsed_time = self.parse_timestamp_from_filename(file_path.name)
            if parsed_time:
                created = parsed_time

            # Determine file type
            file_type = "optimization" if file_path.name.startswith("OPT_") else "portfolio"

            # Extract performance metrics
            sharpe_ratio, cagr = self.extract_metrics_from_csv(file_path)

            # Mark as important if recent or has good metrics
            is_recent = created >= cutoff_date
            is_good_performer = sharpe_ratio is not None and sharpe_ratio > 1.0

            backtest_file = BacktestFile(
                path=file_path,
                created=created,
                size_bytes=size_bytes,
                file_type=file_type,
                sharpe_ratio=sharpe_ratio,
                cagr=cagr,
                is_important=is_recent or is_good_performer,
            )

            files.append(backtest_file)

        return files

    def select_files_to_keep(
        self, files: list[BacktestFile]
    ) -> tuple[list[BacktestFile], list[BacktestFile]]:
        """Select which files to keep vs clean up.

        Returns:
            Tuple of (files_to_keep, files_to_cleanup)
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        files_to_keep = []
        files_to_cleanup = []

        # 1. Keep all recent files (< retention_days old)
        recent_files = [f for f in files if f.created >= cutoff_date]
        files_to_keep.extend(recent_files)
        self.stats["files_kept_recent"] = len(recent_files)

        # 2. From older files, keep top performers by Sharpe ratio
        older_files = [f for f in files if f.created < cutoff_date]

        # Sort by Sharpe ratio (descending), handle None values
        older_files_with_sharpe = [f for f in older_files if f.sharpe_ratio is not None]
        older_files_with_sharpe.sort(key=lambda f: f.sharpe_ratio, reverse=True)

        # Keep top N best performers
        best_performers = older_files_with_sharpe[: self.keep_best_count]
        files_to_keep.extend(best_performers)
        self.stats["files_kept_best"] = len(best_performers)

        # Mark remaining older files for cleanup
        kept_paths = {f.path for f in files_to_keep}
        files_to_cleanup = [f for f in files if f.path not in kept_paths]

        return files_to_keep, files_to_cleanup

    def archive_file(self, backtest_file: BacktestFile) -> bool:
        """Archive a file with compression.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create archive filename with metadata
            archive_name = (
                f"{backtest_file.path.stem}_{backtest_file.created.strftime('%Y%m%d')}.csv.gz"
            )
            archive_path = self.archive_path / archive_name

            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would archive: {backtest_file.path} -> {archive_path}")
                return True

            # Compress and archive
            with open(backtest_file.path, "rb") as f_in:
                with gzip.open(archive_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Create metadata file
            metadata = {
                "original_path": str(backtest_file.path),
                "created": backtest_file.created.isoformat(),
                "size_bytes": backtest_file.size_bytes,
                "file_type": backtest_file.file_type,
                "sharpe_ratio": backtest_file.sharpe_ratio,
                "cagr": backtest_file.cagr,
                "archived_date": datetime.now().isoformat(),
            }

            metadata_path = archive_path.with_suffix(".csv.gz.meta")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Archived: {backtest_file.path} -> {archive_path}")
            self.stats["files_archived"] += 1
            self.stats["bytes_archived"] += backtest_file.size_bytes
            return True

        except Exception as e:
            self.logger.error(f"Failed to archive {backtest_file.path}: {e}")
            return False

    def delete_file(self, backtest_file: BacktestFile) -> bool:
        """Delete a file safely.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would delete: {backtest_file.path}")
                return True

            backtest_file.path.unlink()
            self.logger.info(f"Deleted: {backtest_file.path}")
            self.stats["files_deleted"] += 1
            self.stats["bytes_cleaned"] += backtest_file.size_bytes
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete {backtest_file.path}: {e}")
            return False

    def check_size_limits(self, files: list[BacktestFile]) -> None:
        """Check and report on size limits."""
        total_size_bytes = sum(f.size_bytes for f in files)
        total_size_gb = total_size_bytes / (1024**3)

        self.logger.info(f"Total backtest files size: {total_size_gb:.2f} GB ({len(files)} files)")

        if total_size_gb > self.size_archive_gb:
            self.logger.warning(
                f"Size exceeds archive threshold ({self.size_archive_gb} GB). "
                "Consider more aggressive cleanup or manual review."
            )
        elif total_size_gb > self.size_warn_gb:
            self.logger.warning(
                f"Size approaching warning threshold ({self.size_warn_gb} GB). "
                "Monitor growth and consider cleanup frequency."
            )

    def run_cleanup(self, archive_only: bool = False) -> None:
        """Execute the cleanup process."""
        self.logger.info("Starting backtest cleanup process...")
        self.logger.info(f"Base path: {self.base_path}")
        self.logger.info(f"Retention days: {self.retention_days}")
        self.logger.info(f"Keep best count: {self.keep_best_count}")
        self.logger.info(f"Dry run: {self.dry_run}")
        self.logger.info(f"Archive only: {archive_only}")

        # Setup
        self.setup_directories()

        # Scan files
        self.logger.info("Scanning backtest files...")
        all_files = self.scan_backtest_files()

        if not all_files:
            self.logger.info("No backtest files found.")
            return

        self.logger.info(f"Found {len(all_files)} backtest files")

        # Check size limits
        self.check_size_limits(all_files)

        # Select files to keep vs cleanup
        files_to_keep, files_to_cleanup = self.select_files_to_keep(all_files)

        self.logger.info(f"Files to keep: {len(files_to_keep)}")
        self.logger.info(f"Files to clean up: {len(files_to_cleanup)}")

        if not files_to_cleanup:
            self.logger.info("No files need cleanup.")
            return

        # Process cleanup files
        for backtest_file in files_to_cleanup:
            # Decide whether to archive or delete based on importance
            if (
                backtest_file.is_important
                or backtest_file.sharpe_ratio
                and backtest_file.sharpe_ratio > 0.5
            ):
                # Archive important or decent performing files
                if self.archive_file(backtest_file):
                    if not archive_only:
                        self.delete_file(backtest_file)
            else:
                # Delete poor performers directly
                if not archive_only:
                    self.delete_file(backtest_file)

        # Report results
        self.report_results()

    def report_results(self) -> None:
        """Report cleanup results."""
        self.logger.info("=== Cleanup Results ===")
        self.logger.info(f"Files scanned: {self.stats['files_scanned']}")
        self.logger.info(f"Files kept (recent): {self.stats['files_kept_recent']}")
        self.logger.info(f"Files kept (best performers): {self.stats['files_kept_best']}")
        self.logger.info(f"Files archived: {self.stats['files_archived']}")
        self.logger.info(f"Files deleted: {self.stats['files_deleted']}")
        self.logger.info(f"Bytes cleaned: {self.stats['bytes_cleaned'] / (1024**2):.1f} MB")
        self.logger.info(f"Bytes archived: {self.stats['bytes_archived'] / (1024**2):.1f} MB")


def main():
    """Main entry point for the cleanup script."""
    parser = argparse.ArgumentParser(
        description="Clean up old backtest files with intelligent retention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cleanup_backtests.py --dry-run          # Preview actions
  python scripts/cleanup_backtests.py                    # Execute cleanup
  python scripts/cleanup_backtests.py --archive-only     # Archive only, no delete
  python scripts/cleanup_backtests.py --retention-days 14 # Custom retention
        """,
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Preview actions without making changes"
    )

    parser.add_argument(
        "--archive-only", action="store_true", help="Only archive files, don't delete them"
    )

    parser.add_argument(
        "--retention-days", type=int, default=7, help="Days to keep recent files (default: 7)"
    )

    parser.add_argument(
        "--keep-best", type=int, default=10, help="Number of best performers to keep (default: 10)"
    )

    parser.add_argument("--base-path", type=Path, help="Custom base path for backtest files")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging output")

    args = parser.parse_args()

    # Setup logging
    _ensure_root_config()
    logger = logging.getLogger("backtest_cleanup")

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Determine base path
    if args.base_path:
        base_path = args.base_path
    else:
        # Default to project's data/backtests directory
        script_dir = Path(__file__).parent
        base_path = script_dir.parent / "data" / "backtests"

    if not base_path.exists():
        logger.error(f"Backtest directory does not exist: {base_path}")
        sys.exit(1)

    # Create and run cleanup
    cleanup = BacktestCleanup(base_path, dry_run=args.dry_run)
    cleanup.retention_days = args.retention_days
    cleanup.keep_best_count = args.keep_best

    try:
        cleanup.run_cleanup(archive_only=args.archive_only)
        logger.info("Cleanup completed successfully")

        # Exit with non-zero code in dry-run mode for cron detection
        if args.dry_run:
            sys.exit(2)  # Special exit code for dry run

    except KeyboardInterrupt:
        logger.info("Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
