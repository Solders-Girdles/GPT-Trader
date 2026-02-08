from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from scripts.agents import regenerate_all

SAMPLE_GENERATORS: list[tuple[str, str, str]] = [
    ("fake_schema.py", "schemas", "Fake schema"),
    ("fake_testing.py", "testing", "Fake testing"),
]


def _ensure_committed_dirs(base: Path, generators: Sequence[tuple[str, str, str]]) -> None:
    for _, output_dir, _ in generators:
        (base / output_dir).mkdir(parents=True, exist_ok=True)


def _patch_git_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(regenerate_all.shutil, "which", lambda _: "/usr/bin/git")


def _patch_regenerate_all(monkeypatch: pytest.MonkeyPatch, success: bool = True) -> None:
    def fake_regenerate_all(
        *,
        verbose: bool,
        generators: Sequence[tuple[str, str, str]],
        output_root: Path,
        reasoning_validate: bool,
        reasoning_strict: bool,
    ) -> tuple[list[regenerate_all.GeneratorResult], bool]:
        for _, output_dir, _ in generators:
            (output_root / output_dir).mkdir(parents=True, exist_ok=True)

        results = [
            regenerate_all.GeneratorResult(
                script=script_name,
                success=success,
                duration=0.1,
                output_dir=output_dir,
                error="" if success else "generator failure",
            )
            for script_name, output_dir, _ in generators
        ]
        return results, success

    monkeypatch.setattr(regenerate_all, "regenerate_all", fake_regenerate_all)


@pytest.mark.parametrize("generators", [SAMPLE_GENERATORS])
def test_verify_freshness_reports_all_up_to_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    generators: Sequence[tuple[str, str, str]],
) -> None:
    var_root = tmp_path / "var_agents"
    monkeypatch.setattr(regenerate_all, "VAR_AGENTS_DIR", var_root)
    _ensure_committed_dirs(var_root, generators)

    _patch_git_presence(monkeypatch)
    _patch_regenerate_all(monkeypatch, success=True)
    monkeypatch.setattr(regenerate_all, "_diff_directories", lambda *_: None)

    result = regenerate_all.verify_freshness(generators=generators)

    assert result == 0
    captured = capsys.readouterr()
    assert "All context files are up-to-date." in captured.err


def test_verify_freshness_reports_stale_and_limits_diff_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generators = SAMPLE_GENERATORS
    var_root = tmp_path / "var_agents"
    monkeypatch.setattr(regenerate_all, "VAR_AGENTS_DIR", var_root)
    _ensure_committed_dirs(var_root, generators)

    _patch_git_presence(monkeypatch)
    _patch_regenerate_all(monkeypatch, success=True)

    diff_calls: list[str] = []

    def fake_diff(committed: Path, generated: Path) -> str | None:
        diff_calls.append(committed.name)
        return "diff output" if committed.name == generators[0][1] else None

    monkeypatch.setattr(regenerate_all, "_diff_directories", fake_diff)

    result = regenerate_all.verify_freshness(generators=generators)

    assert result == 1
    assert diff_calls == [generators[0][1], generators[1][1]]

    captured = capsys.readouterr()
    assert "STALE: Agent context files have changed!" in captured.err
    assert "Run 'uv run agent-regenerate' and commit the changes." in captured.err
    assert "diff output" in captured.err


def test_verify_freshness_bails_on_generator_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generators = SAMPLE_GENERATORS
    var_root = tmp_path / "var_agents"
    monkeypatch.setattr(regenerate_all, "VAR_AGENTS_DIR", var_root)
    _ensure_committed_dirs(var_root, generators)

    _patch_git_presence(monkeypatch)
    _patch_regenerate_all(monkeypatch, success=False)

    diff_called = False

    def fake_diff(_: Path, __: Path) -> str | None:
        nonlocal diff_called
        diff_called = True
        return None

    monkeypatch.setattr(regenerate_all, "_diff_directories", fake_diff)

    result = regenerate_all.verify_freshness(generators=generators)

    assert result == 1
    assert not diff_called
    captured = capsys.readouterr()
    assert "Some generators failed. Cannot verify freshness." in captured.err
