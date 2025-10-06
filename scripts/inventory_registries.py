#!/usr/bin/env python3
"""
Registry/Factory Inventory Helper - Automated sweep for abstraction usage.

Finds all Factory/Registry/Builder/Manager classes and checks for:
- Direct imports
- Reflective access (getattr, __import__, importlib)
- String-based lookups
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def find_abstractions() -> dict[str, list[Path]]:
    """Find all Factory/Registry/Builder/Manager/Handler classes."""
    patterns = {
        "Factory": r"class \w+Factory",
        "Registry": r"class \w+Registry",
        "Builder": r"class \w+Builder",
        "Manager": r"class \w+Manager",
        "Handler": r"class \w+Handler",
    }

    results = {}
    for pattern_name, pattern in patterns.items():
        cmd = ["rg", "-l", pattern, str(SRC_DIR), "--type", "py"]
        try:
            output = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if output.returncode == 0:
                files = [Path(f) for f in output.stdout.strip().split("\n") if f]
                results[pattern_name] = files
        except Exception:
            results[pattern_name] = []

    return results


def extract_class_names(file_path: Path) -> list[str]:
    """Extract class names from a file."""
    cmd = ["rg", r"^class (\w+)", str(file_path), "-o", "-r", "$1"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            return output.stdout.strip().split("\n")
    except Exception:
        pass
    return []


def check_direct_imports(class_name: str) -> list[str]:
    """Check for direct imports of a class."""
    patterns = [
        f"from .* import .*{class_name}",
        f"import .*{class_name}",
    ]

    files = []
    for pattern in patterns:
        cmd = ["rg", "-l", pattern, str(SRC_DIR), "--type", "py"]
        try:
            output = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if output.returncode == 0:
                files.extend(output.stdout.strip().split("\n"))
        except Exception:
            pass

    return list({f for f in files if f})


def check_reflective_access() -> dict[str, list[str]]:
    """Check for reflective/dynamic access patterns."""
    patterns = {
        "getattr": r"getattr\(",
        "importlib": r"importlib\.import_module",
        "__import__": r"__import__\(",
        "globals_lookup": r"globals\(\)\[",
        "locals_lookup": r"locals\(\)\[",
    }

    results = {}
    for name, pattern in patterns.items():
        cmd = ["rg", "-l", pattern, str(SRC_DIR), "--type", "py"]
        try:
            output = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if output.returncode == 0:
                results[name] = output.stdout.strip().split("\n")
        except Exception:
            results[name] = []

    return {k: v for k, v in results.items() if v and v != [""]}


def check_string_based_lookups(class_name: str) -> list[str]:
    """Check for string-based class lookups."""
    patterns = [
        f'"{class_name}"',
        f"'{class_name}'",
    ]

    files = []
    for pattern in patterns:
        cmd = ["rg", "-l", pattern, str(SRC_DIR), "--type", "py"]
        try:
            output = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if output.returncode == 0:
                files.extend(output.stdout.strip().split("\n"))
        except Exception:
            pass

    return list({f for f in files if f})


def main():
    """Run registry/factory inventory."""
    print(f"# Registry/Factory Inventory Report - {Path.cwd().name}\n")

    abstractions = find_abstractions()
    reflective = check_reflective_access()

    print("## Reflective Access Patterns Found")
    for pattern, files in reflective.items():
        print(f"- {pattern}: {len(files)} files")
    print()

    results = {}

    for pattern_type, files in abstractions.items():
        print(f"## {pattern_type} Classes ({len(files)} files)")

        for file_path in files:
            classes = extract_class_names(file_path)

            for class_name in classes:
                if not any(
                    x in class_name
                    for x in ["Factory", "Registry", "Builder", "Manager", "Handler"]
                ):
                    continue

                print(f"\nAnalyzing: {class_name} ({file_path.name})")

                direct = check_direct_imports(class_name)
                string_based = check_string_based_lookups(class_name)

                # Exclude self-reference
                direct = [f for f in direct if str(file_path) not in f]
                string_based = [f for f in string_based if str(file_path) not in f]

                total_refs = len(direct) + len(string_based)
                status = "UNUSED" if total_refs == 0 else "IN_USE"

                results[class_name] = {
                    "file": str(file_path.relative_to(PROJECT_ROOT)),
                    "type": pattern_type,
                    "direct_imports": direct,
                    "string_lookups": string_based,
                    "total_references": total_refs,
                    "status": status,
                }

                print(f"  → Direct imports: {len(direct)}")
                print(f"  → String lookups: {len(string_based)}")
                print(f"  → Status: {status}")

    # Save results
    output_file = PROJECT_ROOT / "docs/ops/registry_inventory.json"
    with open(output_file, "w") as f:
        json.dump(
            {"reflective_patterns": reflective, "abstractions": results},
            f,
            indent=2,
            sort_keys=True,
        )

    print(f"\n✓ Results saved to: {output_file.relative_to(PROJECT_ROOT)}")

    # Summary table
    print("\n## Summary Table")
    print(f"| Class | Type | Direct Imports | String Lookups | Status |")
    print(f"|-------|------|----------------|----------------|--------|")
    for name, data in sorted(results.items()):
        print(
            f"| {name} | {data['type']} | {len(data['direct_imports'])} | {len(data['string_lookups'])} | {data['status']} |"
        )


if __name__ == "__main__":
    main()
