from __future__ import annotations

from pathlib import Path

import scripts.ci.check_deprecation_registry as check_deprecation_registry


def _set_repo_root(monkeypatch, repo_root: Path) -> None:
    """Point the script's repo_root computation at a temp repo."""

    fake_script_path = repo_root / "scripts" / "ci" / "check_deprecation_registry.py"
    monkeypatch.setattr(check_deprecation_registry, "__file__", str(fake_script_path))


def test_missing_docs_file_reports_error(monkeypatch, capsys, tmp_path: Path) -> None:
    _set_repo_root(monkeypatch, tmp_path)

    (tmp_path / "src").mkdir(parents=True)

    rc = check_deprecation_registry.main()
    out = capsys.readouterr().out

    assert rc == 1
    assert "::error::Missing docs/DEPRECATIONS.md" in out


def test_docs_without_src_paths_reports_error(monkeypatch, capsys, tmp_path: Path) -> None:
    _set_repo_root(monkeypatch, tmp_path)

    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "docs" / "DEPRECATIONS.md").write_text("No paths here\n", encoding="utf-8")

    rc = check_deprecation_registry.main()
    out = capsys.readouterr().out

    assert rc == 1
    assert "::error::No src paths found in docs/DEPRECATIONS.md" in out


def test_valid_registry_passes(monkeypatch, capsys, tmp_path: Path) -> None:
    _set_repo_root(monkeypatch, tmp_path)

    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "docs").mkdir(parents=True)

    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "shim.py").write_text(
        """\
# This is a deprecation shim
DeprecationWarning
""",
        encoding="utf-8",
    )

    (tmp_path / "docs" / "DEPRECATIONS.md").write_text(
        "- src/pkg/shim.py\n",
        encoding="utf-8",
    )

    rc = check_deprecation_registry.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "Deprecation registry check passed." in out


def test_missing_registry_entry_fails_with_list(monkeypatch, capsys, tmp_path: Path) -> None:
    _set_repo_root(monkeypatch, tmp_path)

    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "docs").mkdir(parents=True)

    (tmp_path / "src" / "a.py").write_text("DeprecationWarning\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text(".. deprecated:: 1.0\n", encoding="utf-8")

    # Only register one of the two shims
    (tmp_path / "docs" / "DEPRECATIONS.md").write_text(
        "src/a.py\n",
        encoding="utf-8",
    )

    rc = check_deprecation_registry.main()
    out = capsys.readouterr().out

    assert rc == 1
    assert "::error::Deprecation shims missing from docs/DEPRECATIONS.md:" in out
    assert " - src/b.py" in out
