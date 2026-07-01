from __future__ import annotations

from pathlib import Path

import scripts.ci.check_import_boundaries as check_import_boundaries


def _configure_rule(monkeypatch: object, repo_root: Path) -> Path:
    src_root = repo_root / "src"
    rule = check_import_boundaries.ImportRule(
        name="features_no_entrypoint_imports",
        description="Feature slices must not import entrypoint layers or the DI container.",
        source_root=src_root / "gpt_trader" / "features",
        forbidden_prefixes=check_import_boundaries.ENTRYPOINT_IMPORT_PREFIXES,
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
        "from gpt_trader.cli.widgets import Widget\n",
    )

    result = check_import_boundaries.scan([str(features_root)])
    captured = capsys.readouterr()

    assert result == 1
    assert "imports gpt_trader.cli.widgets.Widget" in captured.out
    assert "violation(s) found" in captured.out


def test_relative_import_violation(tmp_path, monkeypatch, capsys) -> None:
    features_root = _configure_rule(monkeypatch, tmp_path)
    _write_file(
        tmp_path,
        "src/gpt_trader/features/alpha/relative_violation.py",
        "from ...cli.widgets import Widget\n",
    )

    result = check_import_boundaries.scan([str(features_root)])
    captured = capsys.readouterr()

    assert result == 1
    assert "imports gpt_trader.cli.widgets.Widget" in captured.out
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


def test_container_import_violation(tmp_path, monkeypatch, capsys) -> None:
    features_root = _configure_rule(monkeypatch, tmp_path)
    _write_file(
        tmp_path,
        "src/gpt_trader/features/alpha/container_violation.py",
        "from gpt_trader.app.container import ApplicationContainer\n",
    )

    result = check_import_boundaries.scan([str(features_root)])
    captured = capsys.readouterr()

    assert result == 1
    assert "imports gpt_trader.app.container.ApplicationContainer" in captured.out


def test_default_rules_cover_lower_layer_entrypoint_guards() -> None:
    rule_names = {rule.name for rule in check_import_boundaries.RULES}

    # Feature slices plus every shared infrastructure package must be guarded so
    # the dependency direction (lower layers never import entrypoints) cannot
    # silently regress.
    expected = {
        "features_no_entrypoint_imports",
        "monitoring_no_entrypoint_imports",
        "persistence_no_entrypoint_imports",
        "security_no_entrypoint_imports",
        "core_no_entrypoint_imports",
        "logging_no_entrypoint_imports",
        "utilities_no_entrypoint_imports",
        "validation_no_entrypoint_imports",
        "errors_no_entrypoint_imports",
        "backtesting_no_entrypoint_imports",
        "config_no_entrypoint_imports",
    }
    assert expected <= rule_names


def test_every_guarded_package_exists_on_disk() -> None:
    # A typo'd package name would create a rule whose source_root never exists,
    # silently guarding nothing. Each guarded package must be a real directory.
    for rule in check_import_boundaries.RULES:
        assert rule.source_root.is_dir(), f"missing guarded package: {rule.source_root}"
