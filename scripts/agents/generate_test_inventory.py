#!/usr/bin/env python3
"""Generate machine-readable test inventory for AI agent consumption.

This script scans the test suite to create a comprehensive inventory including:
- Test paths and names
- Pytest markers (categories)
- Test file organization
- Marker-based test selection

Usage:
    python scripts/agents/generate_test_inventory.py [--output-dir DIR]
    python scripts/agents/generate_test_inventory.py --by-marker risk
    python scripts/agents/generate_test_inventory.py --by-path tests/unit/gpt_trader/cli

Output:
    var/agents/testing/
    - test_inventory.json (complete test listing)
    - markers.json (marker definitions and counts)
    - index.json (discovery file)
"""

from __future__ import annotations

import argparse
import ast
import configparser
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent


def parse_pytest_ini() -> dict[str, Any]:
    """Parse pytest.ini to extract marker definitions."""
    ini_path = PROJECT_ROOT / "pytest.ini"
    if not ini_path.exists():
        return {"markers": {}, "default_addopts": ""}

    config = configparser.ConfigParser()
    config.read(ini_path)

    markers = {}
    if config.has_option("pytest", "markers"):
        marker_text = config.get("pytest", "markers")
        for line in marker_text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse "marker_name: description"
            if ":" in line:
                name, desc = line.split(":", 1)
                markers[name.strip()] = desc.strip()

    addopts = ""
    if config.has_option("pytest", "addopts"):
        addopts = config.get("pytest", "addopts").strip()

    return {
        "markers": markers,
        "default_addopts": addopts,
    }


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return list with duplicates removed (stable order)."""
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def extract_test_info(file_path: Path) -> list[dict[str, Any]]:
    """Extract test function information from a Python test file."""
    tests = []

    try:
        content = file_path.read_text()
        tree = ast.parse(content)
    except Exception:
        return tests

    def extract_pytestmark_markers(node: ast.expr) -> list[str]:
        """Extract markers from a pytestmark assignment expression."""
        markers: list[str] = []

        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = node.elts
        else:
            values = [node]

        for value in values:
            marker = extract_marker(value)
            if marker:
                markers.append(marker)

        return markers

    # Get file-level markers (module-level pytestmark assignment)
    file_markers: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "pytestmark":
                    file_markers = dedupe_preserve_order(
                        [*file_markers, *extract_pytestmark_markers(node.value)]
                    )
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "pytestmark":
                if node.value is not None:
                    file_markers = dedupe_preserve_order(
                        [*file_markers, *extract_pytestmark_markers(node.value)]
                    )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            test_info = {
                "name": node.name,
                "line": node.lineno,
                "markers": list(file_markers),
                "docstring": ast.get_docstring(node) or "",
            }

            # Extract pytest markers from decorators
            for decorator in node.decorator_list:
                marker = extract_marker(decorator)
                if marker:
                    test_info["markers"].append(marker)

            test_info["markers"] = dedupe_preserve_order(test_info["markers"])
            tests.append(test_info)

        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            # Extract class-level markers
            class_markers = list(file_markers)
            for decorator in node.decorator_list:
                marker = extract_marker(decorator)
                if marker:
                    class_markers.append(marker)
            class_markers = dedupe_preserve_order(class_markers)

            # Extract methods from test class
            for item in node.body:
                if isinstance(
                    item, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and item.name.startswith("test_"):
                    test_info = {
                        "name": f"{node.name}::{item.name}",
                        "line": item.lineno,
                        "markers": list(class_markers),
                        "docstring": ast.get_docstring(item) or "",
                        "class": node.name,
                    }

                    # Extract method-level markers
                    for decorator in item.decorator_list:
                        marker = extract_marker(decorator)
                        if marker:
                            test_info["markers"].append(marker)

                    test_info["markers"] = dedupe_preserve_order(test_info["markers"])
                    tests.append(test_info)

    return sorted(tests, key=lambda test: (test.get("line", 0), test.get("name", "")))


def extract_test_imports(file_path: Path) -> list[str]:
    """Extract gpt_trader imports from a test file."""
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
    except Exception:
        return []

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("gpt_trader"):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("gpt_trader"):
                imports.add(node.module)

    return sorted(imports)


def extract_marker(decorator: ast.expr) -> str | None:
    """Extract pytest marker name from a decorator AST node."""
    # Handle @pytest.mark.marker_name
    if isinstance(decorator, ast.Attribute):
        if isinstance(decorator.value, ast.Attribute):
            if (
                isinstance(decorator.value.value, ast.Name)
                and decorator.value.value.id == "pytest"
                and decorator.value.attr == "mark"
            ):
                return decorator.attr
    # Handle @pytest.mark.marker_name() with call
    elif isinstance(decorator, ast.Call):
        if isinstance(decorator.func, ast.Attribute):
            if isinstance(decorator.func.value, ast.Attribute):
                if (
                    isinstance(decorator.func.value.value, ast.Name)
                    and decorator.func.value.value.id == "pytest"
                    and decorator.func.value.attr == "mark"
                ):
                    return decorator.func.attr
    return None


def infer_suite_markers(rel_path: str) -> list[str]:
    """Infer suite/type markers based on test file path conventions."""
    if rel_path.startswith("tests/unit/"):
        return ["unit"]
    if rel_path.startswith("tests/integration/"):
        return ["integration"]
    if rel_path.startswith("tests/property/"):
        return ["property"]
    if rel_path.startswith("tests/contract/"):
        return ["contract"]
    if rel_path.startswith("tests/real_api/"):
        return ["real_api"]
    return []


def module_to_path(module_name: str) -> str | None:
    """Resolve a module name to a source path (if it exists)."""
    if not module_name.startswith("gpt_trader"):
        return None
    parts = module_name.split(".")
    base_path = PROJECT_ROOT / "src" / Path(*parts)
    py_candidate = base_path.with_suffix(".py")
    if py_candidate.exists():
        return str(py_candidate.relative_to(PROJECT_ROOT))
    init_candidate = base_path / "__init__.py"
    if init_candidate.exists():
        return str(init_candidate.relative_to(PROJECT_ROOT))
    return None


def scan_test_files(test_dir: Path) -> dict[str, Any]:
    """Scan test directory and collect all test information."""
    inventory: dict[str, list[dict[str, Any]]] = {}
    marker_counts: dict[str, int] = defaultdict(int)
    imports_by_file: dict[str, list[str]] = {}
    total_tests = 0

    for test_file in sorted(test_dir.rglob("test_*.py")):
        if "__pycache__" in str(test_file):
            continue

        rel_path = test_file.relative_to(PROJECT_ROOT).as_posix()
        tests = extract_test_info(test_file)
        imports = extract_test_imports(test_file)

        inferred_markers = infer_suite_markers(rel_path)
        if inferred_markers:
            for test in tests:
                test["markers"] = dedupe_preserve_order(
                    [*test.get("markers", []), *inferred_markers]
                )

        if tests:
            inventory[rel_path] = tests
            total_tests += len(tests)

            for test in tests:
                for marker in test.get("markers", []):
                    marker_counts[marker] += 1
        if imports:
            imports_by_file[rel_path] = imports

    return {
        "inventory": dict(sorted(inventory.items())),
        "marker_counts": dict(marker_counts),
        "total_tests": total_tests,
        "total_files": len(inventory),
        "imports_by_file": dict(sorted(imports_by_file.items())),
    }


def categorize_by_path(inventory: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    """Organize tests by path structure."""
    categories: dict[str, list[str]] = defaultdict(list)

    for file_path in inventory.keys():
        parts = Path(file_path).parts

        # Extract category from path (e.g., tests/unit/gpt_trader/cli -> cli)
        if len(parts) >= 4:
            category = parts[3]  # e.g., "cli", "features", "config"
            categories[category].append(file_path)

    return {category: sorted(paths) for category, paths in sorted(categories.items())}


def generate_test_inventory(
    scan_results: dict[str, Any],
    marker_defs: dict[str, str],
) -> dict[str, Any]:
    """Generate the complete test inventory."""
    inventory = scan_results["inventory"]
    path_categories = categorize_by_path(inventory)
    marker_defs = dict(sorted(marker_defs.items()))

    # Group markers by category (from pytest.ini comments)
    marker_categories = {
        "test_type": ["unit", "integration", "property", "behavioral", "contract", "e2e"],
        "component": [
            "api",
            "endpoints",
            "cli",
            "monitoring",
            "risk",
            "execution",
            "backtesting",
            "persistence",
            "security",
            "utilities",
        ],
        "trading": ["perps", "spot", "portfolio", "strategy", "liquidity"],
        "performance": ["perf", "performance", "slow", "load", "stress"],
        "environment": [
            "real_api",
            "uses_mock_broker",
            "requires_db",
            "requires_network",
            "requires_secrets",
        ],
        "async": ["asyncio", "anyio"],
        "special": ["regression", "flaky", "manual", "deprecated"],
    }

    return {
        "version": "1.0",
        "description": "Machine-readable test inventory for GPT-Trader",
        "summary": {
            "total_tests": scan_results["total_tests"],
            "total_files": scan_results["total_files"],
            "markers_used": len(scan_results["marker_counts"]),
        },
        "marker_definitions": marker_defs,
        "marker_categories": marker_categories,
        "marker_counts": dict(
            sorted(scan_results["marker_counts"].items(), key=lambda x: (-x[1], x[0]))
        ),
        "path_categories": path_categories,
        "tests_by_file": inventory,
    }


def filter_by_marker(inventory: dict[str, Any], marker: str) -> list[str]:
    """Return test paths matching a specific marker."""
    matches = []
    for file_path, tests in inventory.get("tests_by_file", {}).items():
        for test in tests:
            if marker in test.get("markers", []):
                test_id = f"{file_path}::{test['name']}"
                matches.append(test_id)
    return matches


def filter_by_path(inventory: dict[str, Any], path_prefix: str) -> list[str]:
    """Return test paths matching a path prefix."""
    matches = []
    for file_path, tests in inventory.get("tests_by_file", {}).items():
        if file_path.startswith(path_prefix):
            for test in tests:
                test_id = f"{file_path}::{test['name']}"
                matches.append(test_id)
    return matches


def module_from_path(path_value: str) -> str | None:
    """Convert a path to a gpt_trader module name."""
    if path_value.startswith("src/"):
        rel_path = path_value[len("src/") :]
    elif path_value.startswith("gpt_trader/"):
        rel_path = path_value
    else:
        return None

    if rel_path.endswith(".py"):
        rel_path = rel_path[:-3]
    module = rel_path.replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    if not module.startswith("gpt_trader"):
        module = f"gpt_trader.{module}"
    return module


def normalize_source_query(query: str) -> str | None:
    """Normalize source query into a gpt_trader module prefix."""
    if not query:
        return None

    if query.startswith("gpt_trader."):
        return query
    if query.startswith("gpt_trader/") or query.startswith("src/"):
        return module_from_path(query)
    if query.startswith("gpt_trader"):
        return query.replace("/", ".")
    return None


def filter_by_source(
    inventory: dict[str, Any],
    source_test_map: dict[str, Any],
    source_query: str,
) -> list[str]:
    """Return test IDs that import the requested module prefix."""
    module_prefix = normalize_source_query(source_query)
    if not module_prefix:
        return []

    source_to_tests = source_test_map.get("source_to_tests", {})
    matched_files: set[str] = set()
    for module_name, test_files in source_to_tests.items():
        if module_name == module_prefix or module_name.startswith(f"{module_prefix}."):
            matched_files.update(test_files)

    matches: list[str] = []
    for file_path in sorted(matched_files):
        for test in inventory.get("tests_by_file", {}).get(file_path, []):
            matches.append(f"{file_path}::{test['name']}")
    return matches


def find_test_files_for_source(
    source_test_map: dict[str, Any],
    source_query: str,
) -> list[str]:
    """Return test file paths that import the requested module prefix."""
    module_prefix = normalize_source_query(source_query)
    if not module_prefix:
        return []

    source_to_tests = source_test_map.get("source_to_tests", {})
    matched_files: set[str] = set()
    for module_name, test_files in source_to_tests.items():
        if module_name == module_prefix or module_name.startswith(f"{module_prefix}."):
            matched_files.update(test_files)

    return sorted(matched_files)


def build_source_test_map(imports_by_file: dict[str, list[str]]) -> dict[str, Any]:
    """Build a source-to-test map from extracted imports."""
    source_to_tests: dict[str, set[str]] = defaultdict(set)
    test_to_sources: dict[str, list[str]] = {}
    source_paths: dict[str, str] = {}
    unresolved: dict[str, list[str]] = defaultdict(list)

    for test_file, modules in imports_by_file.items():
        unique_modules = sorted(set(modules))
        test_to_sources[test_file] = unique_modules
        for module in unique_modules:
            source_to_tests[module].add(test_file)
            if module not in source_paths:
                resolved = module_to_path(module)
                if resolved:
                    source_paths[module] = resolved
                else:
                    unresolved[module].append(test_file)

    source_to_tests_sorted = {
        module: sorted(list(tests)) for module, tests in sorted(source_to_tests.items())
    }

    return {
        "version": "1.0",
        "summary": {
            "tests_scanned": len(imports_by_file),
            "source_modules": len(source_to_tests_sorted),
            "unresolved_modules": len(unresolved),
        },
        "source_to_tests": source_to_tests_sorted,
        "test_to_sources": dict(sorted(test_to_sources.items())),
        "source_paths": dict(sorted(source_paths.items())),
        "unresolved_modules": {
            module: sorted(tests) for module, tests in sorted(unresolved.items())
        },
    }


def dump_json(path: Path, payload: Any) -> None:
    """Write deterministic JSON output."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test inventory for AI agents")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("var/agents/testing"),
        help="Output directory for inventory files",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("tests"),
        help="Test directory to scan",
    )
    parser.add_argument(
        "--by-marker",
        type=str,
        help="Filter tests by marker and output test IDs",
    )
    parser.add_argument(
        "--by-path",
        type=str,
        help="Filter tests by path prefix and output test IDs",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Filter tests by gpt_trader module import and output test IDs",
    )
    parser.add_argument(
        "--source-files",
        action="store_true",
        help="With --source, output test file paths only",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output to stdout instead of files",
    )

    args = parser.parse_args()

    # Parse pytest.ini for marker definitions
    pytest_config = parse_pytest_ini()
    marker_defs = pytest_config["markers"]

    # Scan test files
    print(f"Scanning {args.test_dir} for tests...")
    scan_results = scan_test_files(PROJECT_ROOT / args.test_dir)

    # Generate inventory
    inventory = generate_test_inventory(scan_results, marker_defs)
    source_test_map = build_source_test_map(scan_results["imports_by_file"])

    # Handle filtering options
    if args.by_marker:
        matches = filter_by_marker(inventory, args.by_marker)
        if args.stdout:
            print(json.dumps(matches, indent=2, sort_keys=True))
        else:
            for m in matches:
                print(m)
        print(f"\nFound {len(matches)} tests with marker '{args.by_marker}'", file=sys.stderr)
        return 0

    if args.by_path:
        matches = filter_by_path(inventory, args.by_path)
        if args.stdout:
            print(json.dumps(matches, indent=2, sort_keys=True))
        else:
            for m in matches:
                print(m)
        print(f"\nFound {len(matches)} tests under '{args.by_path}'", file=sys.stderr)
        return 0

    if args.source:
        if args.source_files:
            matches = find_test_files_for_source(source_test_map, args.source)
            if args.stdout:
                print(json.dumps(matches, indent=2, sort_keys=True))
            else:
                for m in matches:
                    print(m)
            print(
                f"\nFound {len(matches)} test files importing '{args.source}'",
                file=sys.stderr,
            )
        else:
            matches = filter_by_source(inventory, source_test_map, args.source)
            if args.stdout:
                print(json.dumps(matches, indent=2, sort_keys=True))
            else:
                for m in matches:
                    print(m)
            print(f"\nFound {len(matches)} tests importing '{args.source}'", file=sys.stderr)
        return 0

    if args.stdout:
        print(json.dumps(inventory, indent=2, sort_keys=True))
        return 0

    # Write output files
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write main inventory
    inventory_path = output_dir / "test_inventory.json"
    dump_json(inventory_path, inventory)
    print(f"Test inventory written to: {inventory_path}")

    # Write markers reference
    markers_ref = {
        "definitions": marker_defs,
        "categories": inventory["marker_categories"],
        "counts": inventory["marker_counts"],
        "usage": {
            "run_by_marker": "pytest -m <marker>",
            "exclude_marker": "pytest -m 'not <marker>'",
            "combine_markers": "pytest -m '<marker1> and <marker2>'",
        },
    }
    markers_path = output_dir / "markers.json"
    dump_json(markers_path, markers_ref)
    print(f"Markers reference written to: {markers_path}")

    # Write index
    index = {
        "version": "1.0",
        "description": "Test inventory for AI agent consumption",
        "files": {
            "test_inventory": "test_inventory.json",
            "markers": "markers.json",
            "source_test_map": "source_test_map.json",
        },
        "summary": inventory["summary"],
        "quick_commands": {
            "run_unit_tests": "pytest -m unit",
            "run_cli_tests": "pytest tests/unit/gpt_trader/cli",
            "run_fast_tests": "pytest -m 'not slow and not performance'",
            "run_risk_tests": "pytest -m risk",
            "list_all_markers": "pytest --markers",
            "find_tests_for_source": "uv run agent-tests --source gpt_trader.cli",
            "find_test_files_for_source": "uv run agent-tests --source gpt_trader.cli --source-files",
        },
    }
    index_path = output_dir / "index.json"
    dump_json(index_path, index)
    print(f"Index written to: {index_path}")

    # Write source-to-test map
    source_test_map_path = output_dir / "source_test_map.json"
    dump_json(source_test_map_path, source_test_map)
    print(f"Source/test map written to: {source_test_map_path}")

    print(
        f"\nFound {inventory['summary']['total_tests']} tests in {inventory['summary']['total_files']} files"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
