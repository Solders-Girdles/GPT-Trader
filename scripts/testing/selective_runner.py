"""Selective test execution based on dependency analysis outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections.abc import Iterable

DEFAULT_DEP_REPORT = Path("dependency_report.json")
DEFAULT_TEST_CATEGORIES = Path("test_categories.json")
DEFAULT_SOURCE_ROOT = Path("src")
DEFAULT_PACKAGE = "gpt_trader"


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Required file missing: {path}. Generate it with dependency/test tools.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_module_to_tests(report: dict[str, object]) -> dict[str, set[str]]:
    mapping = {}
    tests = report.get("tests_to_modules", {})
    for test_path, modules in tests.items():
        for module in modules:
            mapping.setdefault(module, set()).add(test_path)
    return mapping


def build_category_to_tests(categories: dict[str, Iterable[str]]) -> dict[str, set[str]]:
    return {category: set(paths) for category, paths in categories.items()}


def translate_path_to_module(path: Path, source_root: Path, package: str) -> str | None:
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return None
    parts = list(relative.with_suffix("").parts)
    if not parts:
        return None
    if parts[0] == package:
        module_parts = parts
    else:
        module_parts = [package, *parts]
    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    return ".".join(module_parts)


def determine_tests_for_changed_modules(
    changed: Iterable[str],
    module_to_tests: dict[str, set[str]],
    dependency_report: dict[str, object],
) -> set[str]:
    raw_graph = dependency_report.get("graph", {})
    graph: dict[str, list[str]] = {
        module: list(neighbors) for module, neighbors in raw_graph.items()
    }

    # Build reverse graph to find dependents (modules that import changed modules)
    reverse: dict[str, set[str]] = {node: set() for node in graph}
    for src, targets in graph.items():
        for dest in targets:
            reverse.setdefault(dest, set()).add(src)

    impacted_modules: set[str] = set()

    def collect_dependents(module: str) -> None:
        if module in impacted_modules:
            return
        impacted_modules.add(module)
        for parent in reverse.get(module, set()):
            collect_dependents(parent)

    for changed_module in changed:
        if changed_module in module_to_tests:
            collect_dependents(changed_module)

    affected_tests: set[str] = set()
    for module in impacted_modules:
        affected_tests.update(module_to_tests.get(module, set()))
    return affected_tests


def resolve_categories_to_tests(
    categories: Iterable[str], category_to_tests: dict[str, set[str]]
) -> set[str]:
    selected = set()
    for category in categories:
        selected.update(category_to_tests.get(category, set()))
    return selected


def run_pytest(test_paths: Iterable[str]) -> int:
    if not test_paths:
        print("No tests selected; exiting without running pytest.")
        return 0
    cmd = [sys.executable, "-m", "pytest", *sorted(test_paths)]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective pytest runner using dependency data")
    parser.add_argument(
        "--deps", type=Path, default=DEFAULT_DEP_REPORT, help="dependency_report.json path"
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=DEFAULT_TEST_CATEGORIES,
        help="test_categories.json path",
    )
    parser.add_argument(
        "--changed",
        nargs="+",
        help="List of changed modules (dotted names) to determine affected tests",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        help="Changed file paths to translate into modules/tests",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected tests without invoking pytest",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Run full test suite regardless of selective predictions",
    )
    parser.add_argument(
        "--max-selective-ratio",
        type=float,
        default=0.7,
        help="Fallback to full run if selected tests cover more than this fraction of categories",
    )
    parser.add_argument(
        "--category",
        nargs="+",
        help="Explicit test categories to run (e.g., orchestrator execution_helper)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Project source root containing the package (default: src)",
    )
    parser.add_argument(
        "--package",
        type=str,
        default=DEFAULT_PACKAGE,
        help="Root package name for module translation (default: gpt_trader)",
    )
    parser.add_argument(
        "--auto-full-module",
        action="append",
        default=[],
        help="Module prefix that should trigger a full run when changed (repeatable)",
    )
    args = parser.parse_args()

    dep_report = load_json(args.deps)
    categories_report = load_json(args.categories)

    module_to_tests = build_module_to_tests(dep_report)
    category_to_tests = build_category_to_tests(categories_report)

    tests_to_run: set[str] = set()
    changed_modules: set[str] = set(args.changed or [])

    if args.paths:
        for path_str in args.paths:
            normalized = Path(path_str)
            if normalized.suffix != ".py":
                continue
            module = translate_path_to_module(normalized, args.source_root, args.package)
            if module:
                changed_modules.add(module)
            elif normalized.parts and normalized.parts[0] == "tests":
                # Only add test files that actually exist (skip deleted files from git diff)
                if normalized.exists():
                    tests_to_run.add(str(normalized))

    if changed_modules:
        tests_to_run.update(
            determine_tests_for_changed_modules(changed_modules, module_to_tests, dep_report)
        )

    if args.category:
        tests_to_run.update(resolve_categories_to_tests(args.category, category_to_tests))

    if not changed_modules and not args.category and not tests_to_run:
        print("No selection criteria provided; defaulting to entire suite.")
        tests_to_run = resolve_categories_to_tests(category_to_tests.keys(), category_to_tests)

    auto_full_prefixes = tuple(args.auto_full_module)
    if auto_full_prefixes:
        triggered_by = next(
            (
                module
                for module in sorted(changed_modules)
                for prefix in auto_full_prefixes
                if module.startswith(prefix)
            ),
            None,
        )
        if triggered_by:
            print(f"Full run triggered by high-impact module: {triggered_by}")
            args.force_full = True

    if not tests_to_run or args.force_full:
        tests_to_run = resolve_categories_to_tests(category_to_tests.keys(), category_to_tests)

    if not args.force_full:
        unique_tests = set().union(*category_to_tests.values()) if category_to_tests else set()
        denominator = max(1, len(unique_tests))
        ratio = len(tests_to_run) / denominator
        if ratio > args.max_selective_ratio:
            print(
                f"Selected tests cover {ratio:.0%} of suite, exceeding threshold; running full suite."
            )
            tests_to_run = resolve_categories_to_tests(category_to_tests.keys(), category_to_tests)

    if args.dry_run:
        print("Selected tests (dry run):")
        for path in sorted(tests_to_run):
            print(" ", path)
        return

    exit_code = run_pytest(tests_to_run)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
