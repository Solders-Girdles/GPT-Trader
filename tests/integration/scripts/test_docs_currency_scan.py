"""Integration coverage for the docs-to-code currency scanner.

This walks the real ``docs/`` tree, so it is gated behind the ``integration``
marker (skipped by default) and asserts stable invariants rather than
churn-sensitive counts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.maintenance import docs_currency_scan as scan

pytestmark = pytest.mark.integration


def test_scan_docs_on_real_repo_produces_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    doc_files, extracted, discrepancies = scan.scan_docs(repo_root, fetch_help=False)

    # Invariants that hold regardless of normal docs churn.
    assert doc_files, "expected to discover docs/*.md files"
    assert all(path.suffix == ".md" for path in doc_files)
    assert extracted, "expected to extract at least one named reference"
    assert len(discrepancies) <= len(extracted)
    valid_categories = {"command", "path", "env_var", "cli_flag", "module", "script"}
    assert {item.category for item in extracted} <= valid_categories

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


def test_currency_suppressions_stay_live_and_gate_is_clean() -> None:
    """Every suppression must match a real finding, and --fail-on missing,stale
    must be clean on the repo. Uses fetch_help=True so CLI-flag findings resolve
    to `missing` rather than the `uncertain` fallback used when help is skipped."""
    repo_root = Path(__file__).resolve().parents[3]
    _, _, discrepancies = scan.scan_docs(repo_root, fetch_help=True)
    assert scan.unused_suppressions(discrepancies) == set()
    assert scan.gating_findings(discrepancies, {"missing", "stale"}) == []
