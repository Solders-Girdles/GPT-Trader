#!/usr/bin/env python3
"""
Import Pattern Inventory - Automated sweep for import patterns.

Finds:
- Dynamic imports (importlib, __import__)
- Try/except ImportError patterns with fallbacks
- TYPE_CHECKING guards and circular dependencies
- Feature flag gated imports
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def find_dynamic_imports() -> list[dict[str, str]]:
    """Find dynamic import patterns."""
    results = []

    # importlib.import_module
    cmd = ["rg", "-n", r"importlib\.import_module", str(SRC_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            for line in output.stdout.strip().split("\n"):
                if line:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        results.append(
                            {
                                "file": parts[0],
                                "line": parts[1],
                                "pattern": "importlib.import_module",
                                "code": parts[2].strip(),
                            }
                        )
    except Exception:
        pass

    # __import__()
    cmd = ["rg", "-n", r"__import__\(", str(SRC_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            for line in output.stdout.strip().split("\n"):
                if line:
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        results.append(
                            {
                                "file": parts[0],
                                "line": parts[1],
                                "pattern": "__import__",
                                "code": parts[2].strip(),
                            }
                        )
    except Exception:
        pass

    return results


def find_try_except_imports() -> list[dict[str, str]]:
    """Find try/except ImportError patterns."""
    results = []

    # Find files with try/except ImportError
    cmd = ["rg", "-l", r"except ImportError", str(SRC_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            files = output.stdout.strip().split("\n")

            for file_path in files:
                if not file_path:
                    continue

                # Get context around ImportError
                cmd_context = ["rg", "-A", "3", "-B", "3", "-n", r"except ImportError", file_path]
                context = subprocess.run(cmd_context, capture_output=True, text=True, check=False)

                if context.returncode == 0:
                    results.append(
                        {
                            "file": file_path,
                            "pattern": "try/except ImportError",
                            "context": context.stdout.strip(),
                        }
                    )
    except Exception:
        pass

    return results


def find_type_checking_guards() -> list[dict[str, str]]:
    """Find TYPE_CHECKING import guards."""
    results = []

    cmd = ["rg", "-l", r"if TYPE_CHECKING:", str(SRC_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            files = output.stdout.strip().split("\n")

            for file_path in files:
                if not file_path:
                    continue

                # Get imports under TYPE_CHECKING
                cmd_context = ["rg", "-A", "5", "-n", r"if TYPE_CHECKING:", file_path]
                context = subprocess.run(cmd_context, capture_output=True, text=True, check=False)

                if context.returncode == 0:
                    # Extract import lines
                    imports = []
                    for line in context.stdout.split("\n"):
                        if "from " in line or "import " in line:
                            imports.append(line.strip())

                    if imports:
                        results.append(
                            {
                                "file": file_path,
                                "imports": imports,
                                "reason": "Likely circular dependency",
                            }
                        )
    except Exception:
        pass

    return results


def find_feature_flags() -> list[dict[str, str]]:
    """Find feature flag patterns in imports or conditionals."""
    results = []

    # Common feature flag patterns
    patterns = [
        r"USE_\w+",
        r"ENABLE_\w+",
        r"FEATURE_\w+",
        r"if.*feature.*:",
    ]

    for pattern in patterns:
        cmd = ["rg", "-n", pattern, str(SRC_DIR), "--type", "py"]
        try:
            output = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if output.returncode == 0:
                for line in output.stdout.strip().split("\n"):
                    if line and ("import" in line.lower() or "if" in line.lower()):
                        parts = line.split(":", 2)
                        if len(parts) >= 3:
                            results.append(
                                {
                                    "file": parts[0],
                                    "line": parts[1],
                                    "code": parts[2].strip(),
                                }
                            )
        except Exception:
            pass

    return results


def main():
    """Run import pattern inventory."""
    print(f"# Import Pattern Inventory Report - {Path.cwd().name}\n")

    # Dynamic imports
    dynamic = find_dynamic_imports()
    print(f"## Dynamic Imports ({len(dynamic)} found)")
    for item in dynamic[:10]:  # Show first 10
        file_name = Path(item["file"]).name
        print(f"- {file_name}:{item['line']} → {item['pattern']}: {item['code'][:80]}")
    if len(dynamic) > 10:
        print(f"  ... and {len(dynamic) - 10} more")
    print()

    # Try/except ImportError
    try_except = find_try_except_imports()
    print(f"## Try/Except ImportError Patterns ({len(try_except)} found)")
    for item in try_except[:5]:  # Show first 5
        file_name = Path(item["file"]).name
        print(f"- {file_name}")
    if len(try_except) > 5:
        print(f"  ... and {len(try_except) - 5} more")
    print()

    # TYPE_CHECKING guards
    type_checking = find_type_checking_guards()
    print(f"## TYPE_CHECKING Guards ({len(type_checking)} found)")
    for item in type_checking[:10]:  # Show first 10
        file_name = Path(item["file"]).name
        imports_count = len(item["imports"])
        print(f"- {file_name} → {imports_count} guarded imports")
    if len(type_checking) > 10:
        print(f"  ... and {len(type_checking) - 10} more")
    print()

    # Feature flags
    flags = find_feature_flags()
    print(f"## Feature Flag Patterns ({len(flags)} found)")
    for item in flags[:10]:  # Show first 10
        file_name = Path(item["file"]).name
        print(f"- {file_name}:{item['line']} → {item['code'][:80]}")
    if len(flags) > 10:
        print(f"  ... and {len(flags) - 10} more")
    print()

    # Save results
    results = {
        "dynamic_imports": dynamic,
        "try_except_imports": try_except,
        "type_checking_guards": type_checking,
        "feature_flags": flags,
    }

    output_file = PROJECT_ROOT / "docs/ops/import_inventory.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file.relative_to(PROJECT_ROOT)}")

    # Summary
    print("\n## Summary")
    print(f"- Dynamic imports: {len(dynamic)}")
    print(f"- Try/except ImportError: {len(try_except)}")
    print(f"- TYPE_CHECKING guards: {len(type_checking)}")
    print(f"- Feature flags: {len(flags)}")


if __name__ == "__main__":
    main()
