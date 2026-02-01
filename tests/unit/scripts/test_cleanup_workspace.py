from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest

from scripts.maintenance import cleanup_workspace


@pytest.fixture
def cleanup_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> cleanup_workspace.CleanupSession:
    monkeypatch.setattr(cleanup_workspace, "REPO_ROOT", tmp_path)
    audit_path = tmp_path / "logs" / "cleanup_audit.log"
    return cleanup_workspace.CleanupSession(
        apply=True,
        quiet=True,
        audit_path=audit_path,
    )


def test_clean_tool_caches_preserve_hypothesis(
    cleanup_session: cleanup_workspace.CleanupSession, tmp_path: Path
) -> None:
    (tmp_path / ".pytest_cache").mkdir(parents=True)
    (tmp_path / ".hypothesis").mkdir(parents=True)

    cleanup_workspace.clean_tool_caches(cleanup_session, preserve_hypothesis=True)

    assert (tmp_path / ".pytest_cache").exists() is False
    assert (tmp_path / ".hypothesis").exists()


def test_clean_coverage_artifacts_removes_files(
    cleanup_session: cleanup_workspace.CleanupSession, tmp_path: Path
) -> None:
    (tmp_path / "htmlcov").mkdir()
    (tmp_path / "var" / "results" / "coverage" / "html").mkdir(parents=True)
    (tmp_path / ".coverage").write_text("data")
    (tmp_path / "coverage.json").write_text("{}")
    (tmp_path / "pip-audit-report.json").write_text("{}")

    cleanup_workspace.clean_coverage_artifacts(cleanup_session)

    assert (tmp_path / "htmlcov").exists() is False
    assert (tmp_path / "var" / "results" / "coverage" / "html").exists() is False
    assert (tmp_path / ".coverage").exists() is False
    assert (tmp_path / "coverage.json").exists() is False
    assert (tmp_path / "pip-audit-report.json").exists() is False


def test_clean_var_logs_prunes_archived_gz(
    cleanup_session: cleanup_workspace.CleanupSession, tmp_path: Path
) -> None:
    archive_dir = tmp_path / "var" / "logs" / "archive"
    archive_dir.mkdir(parents=True)
    old_archive = archive_dir / "old.log.gz"
    old_archive.write_bytes(b"old")
    old_time = (cleanup_workspace.utcnow() - timedelta(days=3)).timestamp()
    os.utime(old_archive, (old_time, old_time))

    fresh_archive = archive_dir / "fresh.log.gz"
    fresh_archive.write_bytes(b"fresh")

    cleanup_workspace.clean_var_logs(
        cleanup_session,
        keep_rotations=1,
        retention_days=1,
        threshold_mb=256,
        stale_minutes=60,
    )

    assert old_archive.exists() is False
    assert fresh_archive.exists()


def test_rotate_large_file_archives_stale(
    cleanup_session: cleanup_workspace.CleanupSession, tmp_path: Path
) -> None:
    log_dir = tmp_path / "var" / "logs"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "runtime.log"
    log_path.write_bytes(b"x" * 2048)
    old_time = (cleanup_workspace.utcnow() - timedelta(minutes=90)).timestamp()
    os.utime(log_path, (old_time, old_time))

    archive_dir = log_dir / "archive"

    cleanup_workspace.rotate_large_file(
        cleanup_session,
        log_path,
        archive_dir,
        threshold_mb=0,
        stale_minutes=60,
        category="runtime_log",
    )

    assert log_path.exists()
    assert log_path.stat().st_size == 0
    assert list(archive_dir.glob("runtime_*.log.gz"))
