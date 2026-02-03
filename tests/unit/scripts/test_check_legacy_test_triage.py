"""Tests for scripts/ci/check_legacy_test_triage.py."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import yaml

import scripts.ci.check_legacy_test_triage as check_legacy_test_triage

if TYPE_CHECKING:
    from pathlib import Path


def _write_file(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_manifest(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        content = payload
    else:
        content = yaml.safe_dump(payload, sort_keys=False)
    path.write_text(content, encoding="utf-8")


class TestLoadManifest:
    """Coverage for load_manifest()."""

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            (["entry"], "mapping at the top-level"),
            ({"version": 2, "tests": {}}, "version must be 1"),
            ({"version": 1, "tests": []}, "`tests` must be a mapping"),
        ],
    )
    def test_invalid_manifest_payload_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        payload: Any,
        match: str,
    ) -> None:
        manifest_path = tmp_path / "legacy_tests.yaml"
        monkeypatch.setattr(check_legacy_test_triage, "MANIFEST_PATH", manifest_path)
        _write_manifest(manifest_path, payload)

        with pytest.raises(ValueError, match=match):
            check_legacy_test_triage.load_manifest()


class TestValidateManifestEntries:
    """Coverage for validate_manifest_entries()."""

    def test_bad_paths_are_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_file(tmp_path / "tests" / "unit" / "test_prefix.py")
        _write_file(tmp_path / "docs" / "test_outside.py")
        _write_file(tmp_path / "tests" / "unit" / "test_note.txt")

        manifest = {
            "./tests/unit/test_prefix.py": {
                "action": "delete",
                "reason": "legacy",
                "status": "todo",
            },
            "docs/test_outside.py": {
                "action": "delete",
                "reason": "legacy",
                "status": "todo",
            },
            "tests/unit/test_note.txt": {
                "action": "delete",
                "reason": "legacy",
                "status": "todo",
            },
        }

        problems = check_legacy_test_triage.validate_manifest_entries(manifest)

        assert any("remove leading './'" in problem for problem in problems)
        assert any(
            "docs/test_outside.py: must live under tests/" in problem
            for problem in problems
        )
        assert any(
            "tests/unit/test_note.txt: must be a .py file" in problem
            for problem in problems
        )

    def test_invalid_action_status_and_missing_reason_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_file(tmp_path / "tests" / "unit" / "test_action_status.py")
        _write_file(tmp_path / "tests" / "unit" / "test_missing_reason.py")

        manifest = {
            "tests/unit/test_action_status.py": {
                "action": "remove",
                "reason": "legacy",
                "status": "bad",
            },
            "tests/unit/test_missing_reason.py": {
                "action": "delete",
                "reason": " ",
                "status": "todo",
            },
        }

        problems = check_legacy_test_triage.validate_manifest_entries(manifest)

        assert any("status must be one of" in problem for problem in problems)
        assert any("action must be one of" in problem for problem in problems)
        assert any("reason is required" in problem for problem in problems)


class TestValidateMarkerAlignment:
    """Coverage for validate_marker_alignment()."""

    def test_missing_manifest_and_mismatched_marker_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        marked_files = {
            "tests/unit/test_missing_manifest.py": "legacy_delete",
            "tests/unit/test_mismatch.py": "legacy_delete",
        }
        manifest = {
            "tests/unit/test_mismatch.py": {"action": "modernize"},
        }

        problems = check_legacy_test_triage.validate_marker_alignment(
            manifest, marked_files
        )

        assert any(
            "tests/unit/test_missing_manifest.py: has legacy_delete marker but is missing"
            in problem
            for problem in problems
        )
        assert any(
            "tests/unit/test_mismatch.py: marker legacy_delete does not match manifest action 'modernize'"
            in problem
            for problem in problems
        )

    def test_missing_required_marker_flagged_except_done_missing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_file(tmp_path / "tests" / "unit" / "test_requires_marker.py")

        manifest = {
            "tests/unit/test_requires_marker.py": {
                "action": "delete",
                "status": "todo",
            },
            "tests/unit/test_done_removed.py": {
                "action": "delete",
                "status": "done",
            },
        }

        problems = check_legacy_test_triage.validate_marker_alignment(manifest, {})

        assert any(
            "tests/unit/test_requires_marker.py: manifest action 'delete' requires marker legacy_delete"
            in problem
            for problem in problems
        )
        assert not any("tests/unit/test_done_removed.py" in problem for problem in problems)
