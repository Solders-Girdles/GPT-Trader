#!/usr/bin/env python3
"""
Workspace cleanup utility.

Prunes tool caches, rotates large runtime artifacts, and archives oversized logs
so the repository stays lean for day-to-day development.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_LOG = REPO_ROOT / "logs" / "cleanup_audit.log"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean local workspace artifacts.")
    parser.add_argument(
        "--apply", action="store_true", help="Perform changes instead of a dry-run."
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=14,
        help="Retention window for archived log files in days.",
    )
    parser.add_argument(
        "--keep-rotations",
        type=int,
        default=2,
        help="Number of rotated log files to keep before archiving.",
    )
    parser.add_argument(
        "--log-threshold-mb",
        type=int,
        default=64,
        help="Rotate runtime logs larger than this size (MB).",
    )
    parser.add_argument(
        "--data-threshold-mb",
        type=int,
        default=96,
        help="Rotate event-store JSONL files larger than this size (MB).",
    )
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=60,
        help="Only rotate files untouched for this many minutes.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (audit log is still written).",
    )
    parser.add_argument(
        "--preserve-hypothesis",
        action="store_true",
        help="Keep .hypothesis directory (preserves property-based test examples).",
    )
    return parser.parse_args()


@dataclass
class CleanupSession:
    apply: bool
    quiet: bool = False
    audit_path: Path = AUDIT_LOG
    actions: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "APPLY" if self.apply else "DRY-RUN"
        self._write("INFO", "Cleanup started", {"mode": mode, "root": str(REPO_ROOT)})

    def _write(self, level: str, message: str, extra: dict | None = None) -> None:
        record = {"timestamp": utcnow().isoformat(), "level": level, "message": message}
        if extra:
            record.update(extra)

        with self.audit_path.open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(record, ensure_ascii=True) + "\n")

        if not self.quiet:
            console_line = f"[{level}] {message}"
            if extra:
                detail = " ".join(
                    f"{key}={value}" for key, value in extra.items() if key not in {"timestamp"}
                )
                if detail:
                    console_line = f"{console_line} | {detail}"
            print(console_line)

    def info(self, message: str, extra: dict | None = None) -> None:
        self._write("INFO", message, extra)

    def warn(self, message: str, extra: dict | None = None) -> None:
        self._write("WARN", message, extra)

    def error(self, message: str, extra: dict | None = None) -> None:
        self._write("ERROR", message, extra)

    def record_action(
        self,
        action: str,
        target: Path | str,
        status: str,
        *,
        level: str = "INFO",
        details: dict | None = None,
    ) -> None:
        payload = {"action": action, "target": str(target), "status": status}
        if details:
            payload.update(details)

        self.actions.append(
            {"action": action, "target": str(target), "status": status, "details": details or {}}
        )
        self._write(level, f"{action} -> {status}", payload)

    def record_error(self, action: str, target: Path | str, exc: Exception) -> None:
        details = {"error": type(exc).__name__, "reason": str(exc)}
        self.errors.append({"action": action, "target": str(target), **details})
        self._write("ERROR", f"{action} failed", {"target": str(target), **details})

    def finish(self) -> None:
        summary = {
            "actions_recorded": len(self.actions),
            "errors": len(self.errors),
            "mode": "APPLY" if self.apply else "DRY-RUN",
        }
        level = "SUCCESS" if not self.errors else "WARN"
        self._write(level, "Cleanup complete", summary)


def remove_path(session: CleanupSession, path: Path, *, category: str, reason: str) -> None:
    if not path.exists():
        return

    status = "planned"
    try:
        if session.apply:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            status = "removed"
        session.record_action(
            "remove", path, status, details={"category": category, "reason": reason}
        )
    except Exception as exc:  # pragma: no cover - defensive
        session.record_error("remove", path, exc)


def compress_rotation(
    session: CleanupSession, source: Path, archive_dir: Path, reason: str
) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_name = source.name + ".gz"
    archive_path = archive_dir / archive_name
    if archive_path.exists():
        timestamp = utcnow().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"{source.name}.{timestamp}.gz"

    status = "planned"
    try:
        if session.apply:
            with source.open("rb") as src, gzip.open(archive_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            source.unlink()
            status = "archived"

        session.record_action(
            "archive",
            source,
            status,
            details={"destination": str(archive_path), "category": "runtime_log", "reason": reason},
        )
    except Exception as exc:  # pragma: no cover - defensive
        session.record_error("archive", source, exc)


def rotate_large_file(
    session: CleanupSession,
    path: Path,
    archive_dir: Path,
    *,
    threshold_mb: int,
    stale_minutes: int,
    category: str,
) -> None:
    if not path.exists():
        return

    stat = path.stat()
    size_mb = stat.st_size / (1024 * 1024)
    if size_mb < threshold_mb:
        return

    stale_cutoff = utcnow() - timedelta(minutes=stale_minutes)
    if datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc) > stale_cutoff:
        session.record_action(
            "rotate-skip",
            path,
            "active",
            details={
                "category": category,
                "size_mb": round(size_mb, 2),
                "reason": "recently_modified",
            },
        )
        return

    archive_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes)
    base_name = path.name[: -len(suffix)] if suffix else path.name
    timestamp = utcnow().strftime("%Y%m%d_%H%M%S")
    archive_base = f"{base_name.rstrip('_')}_{timestamp}{suffix}"
    archive_path = archive_dir / archive_base
    compressed_path = archive_dir / f"{archive_base}.gz"

    status = "planned"
    try:
        if session.apply:
            path.rename(archive_path)
            with archive_path.open("rb") as src, gzip.open(compressed_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            archive_path.unlink()
            path.touch()
            status = "rotated"

        session.record_action(
            "rotate",
            path,
            status,
            details={
                "category": category,
                "size_mb": round(size_mb, 2),
                "archive": str(compressed_path),
                "threshold_mb": threshold_mb,
            },
        )
    except Exception as exc:  # pragma: no cover - defensive
        session.record_error("rotate", path, exc)


def parse_rotation(file_path: Path) -> tuple[str, int] | None:
    parts = file_path.name.split(".")
    if len(parts) < 3:
        return None
    rotation = parts[-1]
    if not rotation.isdigit():
        return None
    base_name = ".".join(parts[:-1])
    return base_name, int(rotation)


def clean_tool_caches(session: CleanupSession, *, preserve_hypothesis: bool = False) -> None:
    cache_dirs = [
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".benchmarks",
    ]
    if not preserve_hypothesis:
        cache_dirs.append(".hypothesis")

    for rel in cache_dirs:
        remove_path(
            session, REPO_ROOT / rel, category="tool_cache", reason="regenerated_by_tooling"
        )


def clean_pycache_directories(session: CleanupSession) -> None:
    """Remove __pycache__ directories from source and test trees."""
    # Remove root-level __pycache__
    root_pycache = REPO_ROOT / "__pycache__"
    if root_pycache.exists():
        remove_path(session, root_pycache, category="python_cache", reason="bytecode_cache")

    # Remove __pycache__ from source and test trees
    search_roots = [REPO_ROOT / "src", REPO_ROOT / "tests"]
    for root in search_roots:
        if not root.exists():
            continue
        for pycache in root.rglob("__pycache__"):
            remove_path(session, pycache, category="python_cache", reason="bytecode_cache")


def clean_coverage_artifacts(session: CleanupSession) -> None:
    targets = [
        "htmlcov",
        "var/results/coverage/html",
        ".coverage",
        "coverage.json",
        "pip-audit-report.json",
    ]
    for rel in targets:
        remove_path(
            session, REPO_ROOT / rel, category="coverage_artifact", reason="regenerated_by_tests"
        )


def clean_var_logs(
    session: CleanupSession,
    *,
    keep_rotations: int,
    retention_days: int,
    threshold_mb: int,
    stale_minutes: int,
) -> None:
    log_dir = REPO_ROOT / "var" / "logs"
    if not log_dir.exists():
        session.info("Skipping runtime log cleanup; var/logs does not exist.")
        return

    archive_dir = log_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    retention_cutoff = utcnow() - timedelta(days=retention_days)

    def prune_archives(directory: Path) -> None:
        for entry in directory.glob("*.gz"):
            mtime = datetime.fromtimestamp(entry.stat().st_mtime, tz=timezone.utc)
            if mtime < retention_cutoff:
                remove_path(
                    session,
                    entry,
                    category="runtime_log_archive",
                    reason=f"older_than_{retention_days}d",
                )

    prune_archives(log_dir)
    prune_archives(archive_dir)

    rotations: dict[str, list[tuple[int, Path]]] = {}
    for entry in log_dir.iterdir():
        if entry.is_dir():
            continue

        if entry.suffix == ".gz":
            continue

        rotation_info = parse_rotation(entry)
        if rotation_info:
            key, rotation = rotation_info
            rotations.setdefault(key, []).append((rotation, entry))
            continue

    for base_name, entries in rotations.items():
        entries.sort(key=lambda item: item[0])
        for rotation_index, path in entries[keep_rotations:]:
            compress_rotation(
                session,
                path,
                archive_dir,
                reason=f"rotation_{rotation_index}_beyond_keep_{keep_rotations}",
            )

    for pattern in ("*.log", "*.jsonl"):
        for path in log_dir.glob(pattern):
            rotate_large_file(
                session,
                path,
                archive_dir,
                threshold_mb=threshold_mb,
                stale_minutes=stale_minutes,
                category="runtime_log",
            )


def clean_var_data(
    session: CleanupSession,
    *,
    threshold_mb: int,
    stale_minutes: int,
) -> None:
    data_dir = REPO_ROOT / "var" / "data"
    if not data_dir.exists():
        session.info("Skipping runtime data cleanup; var/data does not exist.")
        return

    archive_dir = data_dir / "archive"
    for jsonl_path in data_dir.rglob("*.jsonl"):
        rotate_large_file(
            session,
            jsonl_path,
            archive_dir,
            threshold_mb=threshold_mb,
            stale_minutes=stale_minutes,
            category="event_store",
        )


def clean_empty_dirs(session: CleanupSession, directories: Iterable[Path]) -> None:
    for directory in directories:
        if directory.exists() and not any(directory.iterdir()):
            remove_path(session, directory, category="empty_directory", reason="no_contents")


def main() -> None:
    args = parse_args()
    session = CleanupSession(apply=args.apply, quiet=args.quiet)

    clean_tool_caches(session, preserve_hypothesis=args.preserve_hypothesis)
    clean_pycache_directories(session)
    clean_coverage_artifacts(session)
    clean_var_logs(
        session,
        keep_rotations=args.keep_rotations,
        retention_days=args.retention_days,
        threshold_mb=args.log_threshold_mb,
        stale_minutes=args.stale_minutes,
    )
    clean_var_data(
        session,
        threshold_mb=args.data_threshold_mb,
        stale_minutes=args.stale_minutes,
    )

    clean_empty_dirs(
        session,
        [
            REPO_ROOT / "htmlcov",
            REPO_ROOT / "var" / "results" / "coverage" / "html",
            REPO_ROOT / "var" / "data" / "archive",
            REPO_ROOT / "var" / "logs" / "archive",
        ],
    )

    session.finish()


if __name__ == "__main__":
    main()
