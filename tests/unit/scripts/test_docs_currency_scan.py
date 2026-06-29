from __future__ import annotations

from pathlib import Path

import pytest
from scripts.maintenance import docs_currency_scan as scan


def test_normalize_item_filters_glossary_noise() -> None:
    assert scan.normalize_item("env_var", "MVP") is None
    assert scan.normalize_item("env_var", "MOCK_BROKER") == "MOCK_BROKER"
    assert (
        scan.normalize_item("path", "runtime_data/canary/reports") == "runtime_data/canary/reports"
    )


def test_verify_path_marks_runtime_paths_uncertain(tmp_path: Path) -> None:
    state = scan.ScanState(repo_root=tmp_path)
    result = scan.verify_path(state, "runtime_data/canary/reports/daily_report_2026-01-01.txt")
    assert result.status == "uncertain"


def test_verify_env_var_accepts_makefile_variable(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text(
        "READINESS_REPORT_DIR?=runtime_data/reports\n", encoding="utf-8"
    )
    state = scan.load_state(tmp_path, fetch_help=False)
    result = scan.verify_env_var(state, "READINESS_REPORT_DIR")
    assert result.status == "ok"
    assert "Makefile" in result.method


def test_verify_command_accepts_python_module(tmp_path: Path) -> None:
    module_dir = tmp_path / "src" / "gpt_trader" / "cli"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "__main__.py").write_text("print('ok')\n", encoding="utf-8")
    state = scan.ScanState(repo_root=tmp_path)
    result = scan.verify_command(state, "python -m gpt_trader.cli")
    assert result.status == "ok"


def test_verify_command_accepts_python_module_file(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "gpt_trader"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tool.py").write_text("print('ok')\n", encoding="utf-8")
    state = scan.ScanState(repo_root=tmp_path)
    result = scan.verify_command(state, "python -m gpt_trader.tool")
    assert result.status == "ok"


def test_verify_command_rejects_python_package_without_main(tmp_path: Path) -> None:
    # A package with only __init__.py is importable but NOT runnable via `python -m`.
    pkg = tmp_path / "src" / "gpt_trader" / "pkgonly"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    state = scan.ScanState(repo_root=tmp_path)
    result = scan.verify_command(state, "python -m gpt_trader.pkgonly")
    assert result.status == "missing"


def test_scan_docs_on_fixture_repo_produces_report(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sample.md").write_text(
        "Run `uv run gpt-trader run` and read `src/gpt_trader/app/missing_module.py`.\n",
        encoding="utf-8",
    )
    doc_files, extracted, discrepancies = scan.scan_docs(tmp_path, fetch_help=False)
    assert [path.name for path in doc_files] == ["sample.md"]
    assert extracted, "expected at least one extracted reference"
    assert any(item.category == "path" for item in extracted)
    report = scan.render_report(
        doc_files=doc_files,
        extracted=extracted,
        discrepancies=discrepancies,
        repo_root=tmp_path,
    )
    assert "# Docs-to-Code Currency Scan Report" in report
    assert "Docs processed:" in report
    out = tmp_path / "report.md"
    out.write_text(report, encoding="utf-8")
    assert out.stat().st_size > 0


@pytest.mark.parametrize(
    ("item", "expected_status"),
    [
        ("gpt_trader.orchestration", "stale"),
        ("scripts/run_spot_profile.py", "stale"),
    ],
)
def test_verify_module_or_path_marks_removed_items_stale(item: str, expected_status: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    state = scan.load_state(repo_root, fetch_help=False)
    if item.startswith("scripts/"):
        result = scan.verify_path(state, item)
    else:
        result = scan.verify_module(state, item, "docs/DEPRECATIONS.md")
    assert result.status == expected_status


def test_short_cli_flags_are_extracted(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text("Use `-k` to filter, `--profile` to select.\n", encoding="utf-8")
    flags = {
        item.item for item in scan.extract_from_doc(doc, tmp_path) if item.category == "cli_flag"
    }
    assert "-k" in flags
    assert "--profile" in flags


def test_unknown_short_flag_is_uncertain_not_missing(tmp_path: Path) -> None:
    state = scan.ScanState(repo_root=tmp_path)
    result = scan.verify_cli_flag(state, "-k", "docs/some.md")
    assert result.status == "uncertain"


def test_unknown_flag_is_uncertain_when_help_skipped(tmp_path: Path) -> None:
    state = scan.load_state(tmp_path, fetch_help=False)
    result = scan.verify_cli_flag(state, "--made-up-flag", "docs/some.md")
    assert result.status == "uncertain"


def test_verify_command_handles_uv_run_python_module(tmp_path: Path) -> None:
    module_dir = tmp_path / "src" / "gpt_trader" / "cli"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "__main__.py").write_text("print('ok')\n", encoding="utf-8")
    state = scan.ScanState(repo_root=tmp_path)
    result = scan.verify_command(state, "uv run python -m gpt_trader.cli")
    assert result.status == "ok"


def test_verify_module_source_fallback_resolves_flat_module(tmp_path: Path) -> None:
    pkg = tmp_path / "src" / "myproj" / "features"
    pkg.mkdir(parents=True)
    (pkg / "symbols.py").write_text("def derive_thing():\n    return 1\n", encoding="utf-8")
    state = scan.ScanState(repo_root=tmp_path)
    result = scan.verify_module(state, "myproj.features.symbols.derive_thing", "docs/x.md")
    assert result.status == "uncertain"
    assert result.method == "source grep"


def test_render_report_collapses_newlines_in_notes(tmp_path: Path) -> None:
    ext = scan.ExtractedItem(
        source_doc="docs/x.md",
        category="module",
        item="gpt_trader.foo",
        item_type="gpt_trader",
    )
    result = scan.VerificationResult("missing", "import", "Traceback:\nLine1\nLine2")
    report = scan.render_report(
        doc_files=[], extracted=[ext], discrepancies=[(ext, result)], repo_root=tmp_path
    )
    rows = [line for line in report.splitlines() if "**missing**" in line]
    assert len(rows) == 1
    assert "Traceback: Line1 Line2" in rows[0]
