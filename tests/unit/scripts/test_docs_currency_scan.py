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


def test_scan_docs_on_real_repo_produces_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    doc_files, extracted, discrepancies = scan.scan_docs(repo_root, fetch_help=False)
    assert len(doc_files) >= 40
    assert len(extracted) > 100
    report = scan.render_report(
        doc_files=doc_files,
        extracted=extracted,
        discrepancies=discrepancies,
        repo_root=repo_root,
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
