#!/usr/bin/env python3
"""
Xfail/Skip Test Inventory - Automated sweep for disabled tests.

Finds:
- @pytest.mark.xfail decorators
- @pytest.mark.skip decorators
- pytest.skip() calls
- pytest.xfail() calls

Extracts reason/context for triage.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"


def find_xfail_tests() -> list[dict[str, Any]]:
    """Find all xfail-marked tests."""
    results = []

    # Find files with xfail markers
    cmd = ["rg", "-l", r"@pytest\.mark\.xfail", str(TESTS_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            files = output.stdout.strip().split("\n")

            for file_path in files:
                if not file_path:
                    continue

                # Get context around each xfail decorator
                cmd_context = ["rg", "-A", "5", "-B", "1", "-n", r"@pytest\.mark\.xfail", file_path]
                context = subprocess.run(cmd_context, capture_output=True, text=True, check=False)

                if context.returncode == 0:
                    # Parse out reason and test name
                    lines = context.stdout.split("\n")
                    current_xfail = None

                    for i, line in enumerate(lines):
                        if "@pytest.mark.xfail" in line:
                            # Extract reason if present
                            reason = None
                            if "reason=" in line:
                                match = re.search(r'reason=["\']([^"\']+)["\']', line)
                                if match:
                                    reason = match.group(1)

                            # Look ahead for test function name
                            test_name = None
                            for j in range(i + 1, min(i + 6, len(lines))):
                                if "def test_" in lines[j]:
                                    match = re.search(r'def (test_\w+)', lines[j])
                                    if match:
                                        test_name = match.group(1)
                                    break

                            # Extract line number
                            line_num = None
                            if ":" in line:
                                parts = line.split(":", 1)
                                try:
                                    line_num = int(parts[0])
                                except ValueError:
                                    pass

                            if test_name:
                                results.append({
                                    "file": file_path,
                                    "line": line_num,
                                    "test_name": test_name,
                                    "reason": reason or "No reason provided",
                                    "marker": "xfail",
                                })
    except Exception:
        pass

    return results


def find_skip_tests() -> list[dict[str, Any]]:
    """Find all skip-marked tests."""
    results = []

    # Find files with skip markers
    cmd = ["rg", "-l", r"@pytest\.mark\.skip", str(TESTS_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            files = output.stdout.strip().split("\n")

            for file_path in files:
                if not file_path:
                    continue

                # Get context around each skip decorator
                cmd_context = ["rg", "-A", "5", "-B", "1", "-n", r"@pytest\.mark\.skip", file_path]
                context = subprocess.run(cmd_context, capture_output=True, text=True, check=False)

                if context.returncode == 0:
                    # Parse out reason and test name
                    lines = context.stdout.split("\n")

                    for i, line in enumerate(lines):
                        if "@pytest.mark.skip" in line:
                            # Extract reason if present
                            reason = None
                            if "reason=" in line:
                                match = re.search(r'reason=["\']([^"\']+)["\']', line)
                                if match:
                                    reason = match.group(1)

                            # Look ahead for test function name
                            test_name = None
                            for j in range(i + 1, min(i + 6, len(lines))):
                                if "def test_" in lines[j]:
                                    match = re.search(r'def (test_\w+)', lines[j])
                                    if match:
                                        test_name = match.group(1)
                                    break

                            # Extract line number
                            line_num = None
                            if ":" in line:
                                parts = line.split(":", 1)
                                try:
                                    line_num = int(parts[0])
                                except ValueError:
                                    pass

                            if test_name:
                                results.append({
                                    "file": file_path,
                                    "line": line_num,
                                    "test_name": test_name,
                                    "reason": reason or "No reason provided",
                                    "marker": "skip",
                                })
    except Exception:
        pass

    return results


def find_inline_skips() -> list[dict[str, Any]]:
    """Find pytest.skip() calls inside tests."""
    results = []

    cmd = ["rg", "-n", r"pytest\.skip\(", str(TESTS_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            for line in output.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split(":", 2)
                if len(parts) >= 3:
                    file_path = parts[0]
                    line_num = parts[1]
                    code = parts[2].strip()

                    # Extract reason from skip call
                    reason = None
                    match = re.search(r'pytest\.skip\(["\']([^"\']+)["\']', code)
                    if match:
                        reason = match.group(1)

                    results.append({
                        "file": file_path,
                        "line": line_num,
                        "code": code[:100],  # Truncate long lines
                        "reason": reason or "No reason provided",
                        "marker": "inline_skip",
                    })
    except Exception:
        pass

    return results


def categorize_by_file_path(tests: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group tests by category based on file path."""
    categories = {
        "integration": [],
        "unit": [],
        "fixtures": [],
        "other": [],
    }

    for test in tests:
        file_path = test["file"]
        if "integration" in file_path:
            categories["integration"].append(test)
        elif "unit" in file_path:
            categories["unit"].append(test)
        elif "fixtures" in file_path or "conftest" in file_path:
            categories["fixtures"].append(test)
        else:
            categories["other"].append(test)

    return categories


def main():
    """Run xfail/skip test inventory."""
    print(f"# Xfail/Skip Test Inventory Report - {Path.cwd().name}\n")

    # Find all disabled tests
    xfail_tests = find_xfail_tests()
    skip_tests = find_skip_tests()
    inline_skips = find_inline_skips()

    all_tests = xfail_tests + skip_tests + inline_skips

    print(f"## Summary")
    print(f"- Total disabled tests: {len(all_tests)}")
    print(f"- @pytest.mark.xfail: {len(xfail_tests)}")
    print(f"- @pytest.mark.skip: {len(skip_tests)}")
    print(f"- pytest.skip() inline: {len(inline_skips)}")
    print()

    # Categorize
    categories = categorize_by_file_path(all_tests)

    print(f"## By Category")
    for category, tests in categories.items():
        if tests:
            print(f"- {category.title()}: {len(tests)}")
    print()

    # Show xfail tests
    if xfail_tests:
        print(f"## Xfail Tests ({len(xfail_tests)} found)\n")
        for test in xfail_tests[:15]:  # Show first 15
            file_name = Path(test["file"]).relative_to(TESTS_DIR) if TESTS_DIR in Path(test["file"]).parents else Path(test["file"]).name
            print(f"- {file_name}:{test['line']} → {test['test_name']}")
            print(f"  Reason: {test['reason']}")

        if len(xfail_tests) > 15:
            print(f"\n  ... and {len(xfail_tests) - 15} more")
        print()

    # Show skip tests
    if skip_tests:
        print(f"## Skip Tests ({len(skip_tests)} found)\n")
        for test in skip_tests[:10]:  # Show first 10
            file_name = Path(test["file"]).relative_to(TESTS_DIR) if TESTS_DIR in Path(test["file"]).parents else Path(test["file"]).name
            print(f"- {file_name}:{test['line']} → {test['test_name']}")
            print(f"  Reason: {test['reason']}")

        if len(skip_tests) > 10:
            print(f"\n  ... and {len(skip_tests) - 10} more")
        print()

    # Show inline skips
    if inline_skips:
        print(f"## Inline Skips ({len(inline_skips)} found)\n")
        for test in inline_skips[:10]:  # Show first 10
            file_name = Path(test["file"]).relative_to(TESTS_DIR) if TESTS_DIR in Path(test["file"]).parents else Path(test["file"]).name
            print(f"- {file_name}:{test['line']}")
            print(f"  Code: {test['code']}")
            print(f"  Reason: {test['reason']}")

        if len(inline_skips) > 10:
            print(f"\n  ... and {len(inline_skips) - 10} more")
        print()

    # Save results
    results = {
        "xfail_tests": xfail_tests,
        "skip_tests": skip_tests,
        "inline_skips": inline_skips,
        "categories": {k: len(v) for k, v in categories.items()},
        "total": len(all_tests),
    }

    output_file = PROJECT_ROOT / "docs/ops/test_xfail_skip_inventory.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
