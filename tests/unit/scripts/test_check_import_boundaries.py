from __future__ import annotations

from pathlib import Path

import scripts.ci.check_import_boundaries as check_import_boundaries


def _configure_rule(monkeypatch: object, repo_root: Path) -> Path:
    src_root = repo_root / "src"
    rule = check_import_boundaries.ImportRule(
        name="features_no_tui_imports",
        description="Feature slices must not import the TUI layer.",
        source_root=src_root / "gpt_trader" / "features",
        forbidden_prefixes=("gpt_trader.tui",),
    )
    monkeypatch.setattr(check_import_boundaries, "REPO_ROOT", repo_root)
    monkeypatch.setattr(check_import_boundaries, "SRC_ROOT", src_root)
    monkeypatch.setattr(check_import_boundaries, "RULES", (rule,))
    return rule.source_root


def _write_file(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_absolute_import_violation(tmp_path, monkeypatch, capsys) -> None:
    features_root = _configure_rule(monkeypatch, tmp_path)
    _write_file(
        tmp_path,
        "src/gpt_trader/features/alpha/absolute_violation.py",
        "from gpt_trader.tui.widgets import Widget\n",
    )

    result = check_import_boundaries.scan([str(features_root)])
    captured = capsys.readouterr()

    assert result == 1
    assert "imports gpt_trader.tui.widgets.Widget" in captured.out
    assert "violation(s) found" in captured.out


def test_relative_import_violation(tmp_path, monkeypatch, capsys) -> None:
    features_root = _configure_rule(monkeypatch, tmp_path)
    _write_file(
        tmp_path,
        "src/gpt_trader/features/alpha/relative_violation.py",
        "from ...tui.widgets import Widget\n",
    )

    result = check_import_boundaries.scan([str(features_root)])
    captured = capsys.readouterr()

    assert result == 1
    assert "imports gpt_trader.tui.widgets.Widget" in captured.out
    assert "violation(s) found" in captured.out


def test_allowed_imports_pass(tmp_path, monkeypatch, capsys) -> None:
    features_root = _configure_rule(monkeypatch, tmp_path)
    _write_file(
        tmp_path,
        "src/gpt_trader/features/alpha/allowed_absolute.py",
        "from gpt_trader.features.alpha import service\n",
    )
    _write_file(
        tmp_path,
        "src/gpt_trader/features/alpha/allowed_relative.py",
        "from ..beta import helper\n",
    )

    result = check_import_boundaries.scan([str(features_root)])
    captured = capsys.readouterr()

    assert result == 0
    assert "Import boundary guard passed." in captured.out
