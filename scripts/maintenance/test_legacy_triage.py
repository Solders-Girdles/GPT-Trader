#!/usr/bin/env python3
"""Legacy test triage helper.

Purpose:
- Identify tests that still follow legacy patterns (unittest-style, script-style, etc.)
- Track decisions (delete vs modernize) in `tests/_triage/legacy_tests.yaml`
- Surface those decisions as pytest markers via the root `conftest.py` hook

This script is intentionally heuristic and non-destructive: it does not edit tests.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = PROJECT_ROOT / "tests"
MANIFEST_PATH = TESTS_ROOT / "_triage" / "legacy_tests.yaml"


@dataclass(frozen=True)
class FileSignals:
    has_tests: bool
    has_non_test_defs: bool
    uses_unittest: bool
    has_main_guard: bool
    has_print: bool
    in_integration_dir: bool
    has_integration_marker: bool
    mentions_legacy: bool


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
        return True
    return False


def _extract_signals(path: Path) -> FileSignals:
    content = _safe_read_text(path)
    in_integration_dir = "tests/integration/" in path.as_posix().replace("\\", "/")
    has_tests = False
    has_non_test_defs = False
    uses_unittest = False
    has_main_guard = False
    has_print = False
    # Integration tests are marked by directory via root `conftest.py`.
    has_integration_marker = in_integration_dir
    mentions_legacy = False

    try:
        tree = ast.parse(content, filename=str(path))
    except SyntaxError:
        # Treat parse failures as "unknown" rather than breaking triage.
        return FileSignals(
            has_tests=False,
            has_non_test_defs=False,
            uses_unittest=False,
            has_main_guard=False,
            has_print=False,
            in_integration_dir=in_integration_dir,
            has_integration_marker=False,
            mentions_legacy=False,
        )

    docstring = ast.get_docstring(tree) or ""
    lowered = docstring.lower()
    mentions_legacy = any(token in lowered for token in ("legacy", "deprecated", "migration"))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                has_tests = True
            else:
                has_non_test_defs = True
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                for child in node.body:
                    if isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and child.name.startswith("test_"):
                        has_tests = True
                        break
            else:
                has_non_test_defs = True

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "TestCase":
                    uses_unittest = True
                if (
                    isinstance(base, ast.Attribute)
                    and isinstance(base.value, ast.Name)
                    and base.value.id == "unittest"
                    and base.attr == "TestCase"
                ):
                    uses_unittest = True
        if _is_main_guard(node):
            has_main_guard = True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                has_print = True
        if isinstance(node, ast.Attribute):
            if getattr(node, "attr", "") == "integration" and "pytest.mark.integration" in content:
                has_integration_marker = True

    return FileSignals(
        has_tests=has_tests,
        has_non_test_defs=has_non_test_defs,
        uses_unittest=uses_unittest,
        has_main_guard=has_main_guard,
        has_print=has_print,
        in_integration_dir=in_integration_dir,
        has_integration_marker=has_integration_marker,
        mentions_legacy=mentions_legacy,
    )


def _load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"version": 1, "tests": {}}
    try:
        import yaml
    except Exception:
        return {"version": 1, "tests": {}}
    data = yaml.safe_load(_safe_read_text(MANIFEST_PATH)) or {}
    if not isinstance(data, dict):
        return {"version": 1, "tests": {}}
    tests = data.get("tests")
    if not isinstance(tests, dict):
        data["tests"] = {}
    return data


def _iter_test_files() -> list[Path]:
    # Match pytest discovery defaults in pytest.ini (test_*.py)
    return sorted(TESTS_ROOT.rglob("test_*.py"))


def _suggest_actions(path: Path, signals: FileSignals) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []

    if not signals.has_tests:
        if signals.has_non_test_defs:
            suggestions.append(
                {
                    "action": "modernize",
                    "reason": "Matches pytest discovery (test_*.py) but contains only helpers; rename/move so pytest doesn't collect it.",
                }
            )
        else:
            suggestions.append(
                {
                    "action": "delete",
                    "reason": "No tests are collected from this module; remove placeholder or move notes to docs.",
                }
            )

    # Script/unittest style is almost always worth modernizing.
    if signals.uses_unittest:
        suggestions.append(
            {
                "action": "modernize",
                "reason": "Uses unittest-style tests; prefer pytest fixtures/asserts for consistency.",
            }
        )
    if signals.has_main_guard:
        suggestions.append(
            {
                "action": "modernize",
                "reason": 'Contains `if __name__ == "__main__"` script runner; remove for test-only modules.',
            }
        )
    if signals.has_print:
        suggestions.append(
            {
                "action": "modernize",
                "reason": "Uses print() in tests; prefer assertions/log capture to keep test output clean.",
            }
        )
    # Integration tests are enforced by directory, not per-file decorators.

    # Flag likely legacy mentions for human review (but don't prescribe delete).
    if signals.mentions_legacy and not suggestions:
        suggestions.append(
            {
                "action": "review",
                "reason": "Docstring mentions legacy/deprecated/migration; confirm it still covers supported behavior.",
            }
        )

    return suggestions


def _format_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    summary = report["summary"]
    lines.append("Legacy test triage report")
    lines.append(f"Test files: {summary['total_files']}")
    lines.append(f"Manifest entries: {summary['triaged_files']}")
    lines.append(f"Untriaged actionable: {summary['untriaged_actionable']}")
    lines.append(f"Untriaged review-only: {summary['untriaged_review_only']}")
    lines.append(f"Manifest issues: {summary['manifest_issues']}")
    lines.append("")

    if report["untriaged_actionable_candidates"]:
        lines.append("Untriaged actionable candidates")
        for item in report["untriaged_actionable_candidates"]:
            lines.append(f"- {item['path']}")
            for suggestion in item["suggestions"]:
                lines.append(f"  - suggest: {suggestion['action']} ({suggestion['reason']})")
            lines.append("")

    if report["untriaged_review_only_candidates"]:
        lines.append("Untriaged review-only candidates")
        for item in report["untriaged_review_only_candidates"]:
            lines.append(f"- {item['path']}")
            for suggestion in item["suggestions"]:
                lines.append(f"  - suggest: {suggestion['action']} ({suggestion['reason']})")
            lines.append("")

    if report["triaged"]:
        lines.append("Triaged entries")
        for item in report["triaged"]:
            lines.append(f"- {item['path']}")
            lines.append(f"  - manifest: {item['manifest']}")
            for suggestion in item.get("suggestions", []):
                lines.append(f"  - suggest: {suggestion['action']} ({suggestion['reason']})")
            lines.append("")

    if report["manifest_issues"]:
        lines.append("Manifest issues")
        for issue in report["manifest_issues"]:
            lines.append(f"- {issue['path']}: {issue['issue']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _is_allowed_action(action: object) -> bool:
    return str(action).strip().lower() in {"delete", "modernize"}


def _iter_manifest_issues(manifest_tests: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for rel, entry in manifest_tests.items():
        if not isinstance(rel, str) or not rel.strip():
            continue
        if not isinstance(entry, dict):
            issues.append({"path": str(rel), "issue": "Manifest entry must be a mapping/dict."})
            continue

        status = entry.get("status")
        normalized_status = str(status).strip().lower() if status is not None else ""

        abs_path = PROJECT_ROOT / rel
        if not abs_path.exists():
            if normalized_status not in {"done", "completed"}:
                issues.append({"path": rel, "issue": "Path does not exist on disk."})

        action = entry.get("action")
        if action is not None and not _is_allowed_action(action):
            issues.append(
                {
                    "path": rel,
                    "issue": f"Unknown action: {action!r} (expected 'delete' or 'modernize').",
                }
            )

        if status is not None:
            if normalized_status not in {
                "todo",
                "in_progress",
                "done",
                "completed",
            } and not _is_allowed_action(normalized_status):
                issues.append(
                    {
                        "path": rel,
                        "issue": f"Unknown status: {status!r} (expected todo|in_progress|done).",
                    }
                )

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if there are untriaged actionable candidates or manifest issues.",
    )
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Also fail `--check` when review-only items exist.",
    )
    parser.add_argument("--write", type=str, default="", help="Write output to a file path.")
    args = parser.parse_args(argv)

    manifest = _load_manifest()
    manifest_tests = manifest.get("tests") if isinstance(manifest.get("tests"), dict) else {}

    test_files = _iter_test_files()
    manifest_issues = _iter_manifest_issues(
        manifest_tests if isinstance(manifest_tests, dict) else {}
    )

    actionable_untriaged = 0
    review_only_untriaged = 0

    untriaged_actionable_candidates: list[dict[str, Any]] = []
    untriaged_review_only_candidates: list[dict[str, Any]] = []
    triaged: list[dict[str, Any]] = []

    for path in test_files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        signals = _extract_signals(path)
        suggestions = _suggest_actions(path, signals)

        entry = manifest_tests.get(rel) if isinstance(manifest_tests, dict) else None
        if entry:
            triaged.append(
                {
                    "path": rel,
                    "signals": signals.__dict__,
                    "suggestions": suggestions,
                    "manifest": entry,
                }
            )
        elif suggestions:
            has_actionable = any(s.get("action") in {"delete", "modernize"} for s in suggestions)
            has_review = any(s.get("action") == "review" for s in suggestions)
            if has_actionable:
                actionable_untriaged += 1
                untriaged_actionable_candidates.append(
                    {
                        "path": rel,
                        "signals": signals.__dict__,
                        "suggestions": suggestions,
                    }
                )
            elif has_review:
                review_only_untriaged += 1
                untriaged_review_only_candidates.append(
                    {
                        "path": rel,
                        "signals": signals.__dict__,
                        "suggestions": suggestions,
                    }
                )

    report = {
        "summary": {
            "total_files": len(test_files),
            "triaged_files": len(manifest_tests) if isinstance(manifest_tests, dict) else 0,
            "untriaged_actionable": actionable_untriaged,
            "untriaged_review_only": review_only_untriaged,
            "manifest_issues": len(manifest_issues),
        },
        "untriaged_actionable_candidates": untriaged_actionable_candidates,
        "untriaged_review_only_candidates": untriaged_review_only_candidates,
        "triaged": triaged,
        "manifest_issues": manifest_issues,
    }

    if args.format == "json":
        out = json.dumps(report, indent=2, sort_keys=True)
    else:
        out = _format_text(report)

    if args.write:
        out_path = Path(args.write)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")
    else:
        sys.stdout.write(out)

    if args.check:
        has_actionable = actionable_untriaged > 0
        has_reviews = review_only_untriaged > 0
        if manifest_issues or has_actionable or (args.fail_on_review and has_reviews):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
