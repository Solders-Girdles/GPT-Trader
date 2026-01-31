from __future__ import annotations

from datetime import datetime as real_datetime
from pathlib import Path

import pytest

import scripts.ci.check_dedupe_manifest as check_dedupe_manifest


def _base_cluster() -> dict[str, object]:
    return {
        "type": "source_fanout",
        "files": ["tests/unit/example_test.py"],
        "decision": "merge",
        "priority": "high",
        "expected_file_delta": -1,
    }


def _write_manifest(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_manifest_missing_returns_empty(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr(check_dedupe_manifest, "MANIFEST_PATH", missing)

    assert check_dedupe_manifest.load_manifest() == {}


def test_load_manifest_empty_raises(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "dedupe.yaml"
    _write_manifest(manifest_path, "")
    monkeypatch.setattr(check_dedupe_manifest, "MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="Manifest is empty"):
        check_dedupe_manifest.load_manifest()


def test_load_manifest_top_level_mapping_required(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "dedupe.yaml"
    _write_manifest(manifest_path, "- item\n")
    monkeypatch.setattr(check_dedupe_manifest, "MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="Manifest must be a mapping"):
        check_dedupe_manifest.load_manifest()


def test_load_manifest_version_must_be_one(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "dedupe.yaml"
    _write_manifest(manifest_path, "version: 2\nclusters: {}\n")
    monkeypatch.setattr(check_dedupe_manifest, "MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="Manifest version must be 1"):
        check_dedupe_manifest.load_manifest()


def test_load_manifest_clusters_must_be_mapping(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "dedupe.yaml"
    _write_manifest(manifest_path, "version: 1\nclusters: []\n")
    monkeypatch.setattr(check_dedupe_manifest, "MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="Manifest `clusters` must be a mapping"):
        check_dedupe_manifest.load_manifest()


def test_validate_manifest_empty_is_ok() -> None:
    assert check_dedupe_manifest.validate_manifest({}) == []


def test_validate_manifest_reports_missing_required_fields() -> None:
    manifest = {"clusters": {"alpha": {}}}

    problems = check_dedupe_manifest.validate_manifest(manifest)

    assert set(problems) == {
        "Cluster alpha: missing 'type' field",
        "Cluster alpha: missing 'files' field",
        "Cluster alpha: missing 'decision' field",
        "Cluster alpha: missing 'priority' field",
        "Cluster alpha: missing 'expected_file_delta' field",
    }


def test_validate_manifest_reports_invalid_fields() -> None:
    cluster = _base_cluster()
    cluster.update(
        {
            "files": "tests/unit/example_test.py",
            "decision": "keep",
            "priority": "urgent",
            "status": "blocked",
            "type": "unknown",
        }
    )
    manifest = {"clusters": {"alpha": cluster}}

    problems = check_dedupe_manifest.validate_manifest(manifest)

    assert set(problems) == {
        "Cluster alpha: 'files' must be a list",
        "Cluster alpha: invalid decision 'keep'",
        "Cluster alpha: invalid priority 'urgent'",
        "Cluster alpha: invalid status 'blocked'",
        "Cluster alpha: invalid type 'unknown'",
    }


def test_validate_manifest_cluster_must_be_mapping() -> None:
    manifest = {"clusters": {"alpha": ["not-a-mapping"]}}

    problems = check_dedupe_manifest.validate_manifest(manifest)

    assert problems == ["Cluster alpha: must be a mapping"]


def test_validate_manifest_in_progress_missing_owner() -> None:
    cluster = _base_cluster()
    cluster.update({"status": "in_progress", "pr_url": "https://example.com/pr"})
    manifest = {"clusters": {"alpha": cluster}}

    problems = check_dedupe_manifest.validate_manifest(manifest)

    assert problems == ["Cluster alpha: in_progress cluster missing 'owner'"]


def test_validate_manifest_in_progress_invalid_added_date() -> None:
    cluster = _base_cluster()
    cluster.update({"status": "in_progress", "owner": "alice", "added": "01-01-2024"})
    manifest = {"clusters": {"alpha": cluster}}

    problems = check_dedupe_manifest.validate_manifest(manifest)

    assert problems == ["Cluster alpha: invalid 'added' date format (expected YYYY-MM-DD)"]


def test_validate_manifest_in_progress_stale_without_pr_url(monkeypatch) -> None:
    fixed_now = real_datetime(2024, 2, 15)

    class FixedDateTime:
        @classmethod
        def now(cls) -> real_datetime:
            return fixed_now

        @classmethod
        def strptime(cls, value: str, fmt: str) -> real_datetime:
            return real_datetime.strptime(value, fmt)

    monkeypatch.setattr(check_dedupe_manifest, "datetime", FixedDateTime)

    cluster = _base_cluster()
    cluster.update({"status": "in_progress", "owner": "alice", "added": "2024-01-01"})
    manifest = {"clusters": {"alpha": cluster}}

    problems = check_dedupe_manifest.validate_manifest(manifest)

    days_old = (fixed_now - real_datetime(2024, 1, 1)).days
    assert problems == [
        "Cluster alpha: in_progress for "
        f"{days_old} days without pr_url (max: {check_dedupe_manifest.MAX_IN_PROGRESS_DAYS} days)"
    ]


def test_validate_manifest_done_requires_owner_and_pr_url() -> None:
    cluster = _base_cluster()
    cluster.update({"status": "done"})
    manifest = {"clusters": {"alpha": cluster}}

    problems = check_dedupe_manifest.validate_manifest(manifest)

    assert set(problems) == {
        "Cluster alpha: done cluster missing 'owner'",
        "Cluster alpha: done cluster missing 'pr_url'",
    }
