from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from scripts.monitoring import manage_logs


@pytest.fixture
def log_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> manage_logs.LogManager:
    monkeypatch.setenv("COINBASE_TRADER_LOG_DIR", str(tmp_path))
    return manage_logs.LogManager()


def test_rotate_logs_by_size(log_manager: manage_logs.LogManager) -> None:
    log_path = log_manager.log_dir / "app.log"
    log_path.write_bytes(b"x" * (1024 * 1024 + 5))

    rotated = log_manager.rotate_logs(max_size_mb=1, max_age_hours=24)

    assert len(rotated) == 1
    assert rotated[0].startswith("app_")
    assert rotated[0].endswith(".log")
    assert log_path.exists()
    assert log_path.stat().st_size == 0

    compressed = list(log_manager.log_dir.glob("app_*.log.gz"))
    assert len(compressed) == 1


def test_rotate_logs_by_age(log_manager: manage_logs.LogManager) -> None:
    log_path = log_manager.log_dir / "old.log"
    log_path.write_text("old log")
    old_time = (datetime.now() - timedelta(hours=2)).timestamp()
    os.utime(log_path, (old_time, old_time))

    rotated = log_manager.rotate_logs(max_size_mb=10, max_age_hours=1)

    assert rotated == ["old.log"]
    assert log_path.exists() is False
    archived = log_manager.archive_dir / "old.log.gz"
    assert archived.exists()


def test_generate_summary_counts_plain_and_gz(log_manager: manage_logs.LogManager) -> None:
    content = "INFO order placed\nWARNING low balance\nERROR trade failed\nFill executed\n"
    log_path = log_manager.log_dir / "activity.log"
    log_path.write_text(content)

    summary = log_manager.generate_summary(log_path)

    assert summary["lines"] == 4
    assert summary["errors"] == 1
    assert summary["warnings"] == 1
    assert summary["orders"] == 1
    assert summary["trades"] == 2
    assert summary["error"] == ""

    compressed_path = log_manager.compress_log(log_path)
    compressed_summary = log_manager.generate_summary(compressed_path)

    assert compressed_summary["lines"] == 4
    assert compressed_summary["errors"] == 1
    assert compressed_summary["warnings"] == 1
    assert compressed_summary["orders"] == 1
    assert compressed_summary["trades"] == 2
    assert compressed_summary["error"] == ""


def test_generate_summary_missing_file(log_manager: manage_logs.LogManager) -> None:
    missing_path = log_manager.log_dir / "missing.log"

    summary = log_manager.generate_summary(missing_path)

    assert summary["lines"] == 0
    assert summary["size_bytes"] == 0
    assert summary["error"]


def test_cleanup_old_logs_removes_archives(log_manager: manage_logs.LogManager) -> None:
    old_archive = log_manager.archive_dir / "old.log.gz"
    old_archive.write_bytes(b"old")
    old_time = (datetime.now() - timedelta(days=2)).timestamp()
    os.utime(old_archive, (old_time, old_time))

    fresh_archive = log_manager.archive_dir / "fresh.log.gz"
    fresh_archive.write_bytes(b"fresh")

    removed = log_manager.cleanup_old_logs(keep_days=1)

    assert removed == ["old.log.gz"]
    assert old_archive.exists() is False
    assert fresh_archive.exists()


def test_get_active_logs_reports_sizes(log_manager: manage_logs.LogManager) -> None:
    large_log = log_manager.log_dir / "a.log"
    large_log.write_bytes(b"x" * (1024 * 1024 + 1))
    small_log = log_manager.log_dir / "b.log"
    small_log.write_text("line 1\nline 2\n")

    logs = log_manager.get_active_logs()

    assert [log["name"] for log in logs] == ["a.log", "b.log"]
    assert logs[0]["lines"] == "large"
    assert logs[1]["lines"] == 2
