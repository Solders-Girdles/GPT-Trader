from __future__ import annotations

from pathlib import Path

from scripts.ci import check_orchestration_imports as guard

# Built by concatenation so this test file does not itself trip the guard, which
# scans tests/ for the literal pattern.
_ORCH = "gpt_trader." + "orchestration"


def _make_repo(tmp_path: Path) -> Path:
    for root in guard.SEARCH_ROOTS:
        (tmp_path / root).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_no_violations_returns_empty(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "src" / "clean.py").write_text(
        "from gpt_trader.app import container\n", encoding="utf-8"
    )

    assert guard.find_violations(repo) == []


def test_detects_from_import(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "src" / "legacy.py").write_text(f"from {_ORCH} import runner\n", encoding="utf-8")

    violations = guard.find_violations(repo)

    assert len(violations) == 1
    assert "src/legacy.py:1" in violations[0]


def test_detects_plain_import_in_tests_and_scripts(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "tests" / "test_legacy.py").write_text(f"import {_ORCH}\n", encoding="utf-8")
    (repo / "scripts" / "tool.py").write_text(f"import {_ORCH}.helpers\n", encoding="utf-8")

    violations = guard.find_violations(repo)

    assert len(violations) == 2


def test_detects_comma_separated_import(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "scripts" / "tool.py").write_text(f"import os, {_ORCH}\n", encoding="utf-8")

    violations = guard.find_violations(repo)

    assert len(violations) == 1
    assert "scripts/tool.py:1" in violations[0]


def test_ignores_similarly_named_module(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "src" / "future.py").write_text(f"import {_ORCH}_v2\n", encoding="utf-8")

    assert guard.find_violations(repo) == []
