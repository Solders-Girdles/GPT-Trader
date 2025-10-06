#!/usr/bin/env python3
"""
Config Inventory Helper - Automated sweep for config file usage.

Finds all config files and traces their usage in the codebase.
Output can be diffed between runs to track changes.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"


def find_config_files() -> list[Path]:
    """Find all config files in the config directory."""
    if not CONFIG_DIR.exists():
        return []
    return sorted(CONFIG_DIR.glob("**/*.yaml")) + sorted(CONFIG_DIR.glob("**/*.json"))


def search_config_usage(config_name: str) -> dict[str, list[str]]:
    """Search for usage of a config file in the codebase."""
    results = {
        "direct_references": [],
        "yaml_loads": [],
        "path_constructions": [],
    }

    # Direct file name references
    cmd = ["rg", "-l", config_name, str(SRC_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            results["direct_references"] = output.stdout.strip().split("\n")
    except Exception:
        pass

    # YAML load patterns near config name
    config_base = config_name.replace(".yaml", "").replace(".json", "")
    yaml_pattern = f"yaml\\.load.*{config_base}|{config_base}.*yaml\\.load"
    cmd = ["rg", "-l", yaml_pattern, str(SRC_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            results["yaml_loads"] = output.stdout.strip().split("\n")
    except Exception:
        pass

    # Path construction patterns
    path_pattern = f"Path.*{config_base}|pathlib.*{config_base}"
    cmd = ["rg", "-l", path_pattern, str(SRC_DIR), "--type", "py"]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if output.returncode == 0:
            results["path_constructions"] = output.stdout.strip().split("\n")
    except Exception:
        pass

    # Remove empty results
    return {k: v for k, v in results.items() if v and v != [""]}


def check_dynamic_loading(config_name: str) -> list[str]:
    """Check for dynamic/string-based config loading."""
    patterns = [
        f'"{config_name}"',
        f"'{config_name}'",
        f'f".*{config_name.replace(".yaml", "")}',
        f"f'.*{config_name.replace('.yaml', '')}",
    ]

    dynamic_refs = []
    for pattern in patterns:
        cmd = ["rg", "-l", pattern, str(SRC_DIR), "--type", "py"]
        try:
            output = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if output.returncode == 0:
                dynamic_refs.extend(output.stdout.strip().split("\n"))
        except Exception:
            pass

    return list({r for r in dynamic_refs if r})


def main():
    """Run config inventory and output results."""
    config_files = find_config_files()

    print(f"# Config Inventory Report - {Path.cwd().name}")
    print(f"Found {len(config_files)} config files\n")

    results = {}

    for config_file in config_files:
        config_name = config_file.name
        print(f"Analyzing: {config_name}")

        usage = search_config_usage(config_name)
        dynamic = check_dynamic_loading(config_name)

        results[config_name] = {
            "path": str(config_file.relative_to(PROJECT_ROOT)),
            "size_bytes": config_file.stat().st_size,
            "usage": usage,
            "dynamic_loading": dynamic,
            "status": "UNUSED" if not usage and not dynamic else "IN_USE",
        }

        # Print summary
        total_refs = sum(len(v) for v in usage.values()) + len(dynamic)
        print(f"  → {total_refs} references found ({results[config_name]['status']})")

    # Output JSON for diffing
    output_file = PROJECT_ROOT / "docs/ops/config_inventory.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"\n✓ Results saved to: {output_file.relative_to(PROJECT_ROOT)}")

    # Summary table
    print("\n## Summary Table")
    print(f"| Config File | Size | References | Status |")
    print(f"|-------------|------|------------|--------|")
    for name, data in sorted(results.items()):
        total_refs = sum(len(v) for v in data["usage"].values()) + len(data["dynamic_loading"])
        print(f"| {name} | {data['size_bytes']} | {total_refs} | {data['status']} |")


if __name__ == "__main__":
    main()
