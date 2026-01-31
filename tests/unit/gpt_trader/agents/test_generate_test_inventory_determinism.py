from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Repository root not found for test inventory script")


@pytest.fixture(scope="module")
def generate_test_inventory_module() -> ModuleType:
    repo_root = _find_repo_root(Path(__file__).resolve())
    script_path = repo_root / "scripts" / "agents" / "generate_test_inventory.py"
    module_spec = importlib.util.spec_from_file_location("generate_test_inventory", script_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _build_scan_results(
    inventory: dict[str, list[dict[str, Any]]],
    marker_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "inventory": inventory,
        "marker_counts": marker_counts,
        "total_tests": sum(len(tests) for tests in inventory.values()),
        "total_files": len(inventory),
        "imports_by_file": {},
    }


def test_generate_test_inventory_deterministic_order(
    generate_test_inventory_module: ModuleType,
) -> None:
    alpha_path = "tests/unit/gpt_trader/alpha/test_alpha.py"
    cli_path = "tests/unit/gpt_trader/cli/test_cli.py"

    inventory_a = {
        alpha_path: [
            {"name": "test_zeta", "line": 12, "markers": ["risk"]},
            {"name": "test_alpha", "line": 5, "markers": ["unit", "risk"]},
        ],
        cli_path: [
            {"name": "test_beta", "line": 7, "markers": ["cli"]},
        ],
    }
    inventory_b = {
        cli_path: [
            {"name": "test_beta", "line": 7, "markers": ["cli"]},
        ],
        alpha_path: [
            {"name": "test_alpha", "line": 5, "markers": ["unit", "risk"]},
            {"name": "test_zeta", "line": 12, "markers": ["risk"]},
        ],
    }

    marker_counts_a = {"unit": 1, "risk": 2, "cli": 1}
    marker_counts_b = {"cli": 1, "risk": 2, "unit": 1}

    marker_defs_a = {"unit": "Unit tests", "risk": "Risk tests", "cli": "CLI tests"}
    marker_defs_b = {"cli": "CLI tests", "unit": "Unit tests", "risk": "Risk tests"}

    scan_results_a = _build_scan_results(inventory_a, marker_counts_a)
    scan_results_b = _build_scan_results(inventory_b, marker_counts_b)

    result_a = generate_test_inventory_module.generate_test_inventory(
        scan_results_a, marker_defs_a
    )
    result_b = generate_test_inventory_module.generate_test_inventory(
        scan_results_b, marker_defs_b
    )

    assert list(result_a["tests_by_file"].items()) == list(result_b["tests_by_file"].items())
    assert list(result_a["marker_definitions"].items()) == list(
        result_b["marker_definitions"].items()
    )
    assert list(result_a["marker_counts"].items()) == list(result_b["marker_counts"].items())
    assert list(result_a["path_categories"].items()) == list(result_b["path_categories"].items())

    assert list(result_a["tests_by_file"].keys()) == [alpha_path, cli_path]
    alpha_tests = result_a["tests_by_file"][alpha_path]
    assert [test["name"] for test in alpha_tests] == ["test_alpha", "test_zeta"]

    assert list(result_a["marker_definitions"].items()) == [
        ("cli", "CLI tests"),
        ("risk", "Risk tests"),
        ("unit", "Unit tests"),
    ]
    assert list(result_a["marker_counts"].items()) == [
        ("risk", 2),
        ("cli", 1),
        ("unit", 1),
    ]
    assert list(result_a["path_categories"].items()) == [
        ("alpha", [alpha_path]),
        ("cli", [cli_path]),
    ]


def test_build_source_test_map_deterministic_order(
    generate_test_inventory_module: ModuleType,
) -> None:
    alpha_path = "tests/unit/gpt_trader/alpha/test_alpha.py"
    cli_path = "tests/unit/gpt_trader/cli/test_cli.py"

    imports_by_file_a = {
        alpha_path: [
            "gpt_trader.cli",
            "gpt_trader.cli",
            "gpt_trader.monitoring",
            "gpt_trader.missing.module",
        ],
        cli_path: [
            "gpt_trader.cli",
            "gpt_trader.monitoring",
        ],
    }
    imports_by_file_b = {
        cli_path: [
            "gpt_trader.monitoring",
            "gpt_trader.cli",
        ],
        alpha_path: [
            "gpt_trader.missing.module",
            "gpt_trader.cli",
            "gpt_trader.monitoring",
        ],
    }

    map_a = generate_test_inventory_module.build_source_test_map(imports_by_file_a)
    map_b = generate_test_inventory_module.build_source_test_map(imports_by_file_b)

    assert list(map_a["source_to_tests"].items()) == list(map_b["source_to_tests"].items())
    assert list(map_a["test_to_sources"].items()) == list(map_b["test_to_sources"].items())
    assert list(map_a["source_paths"].items()) == list(map_b["source_paths"].items())
    assert list(map_a["unresolved_modules"].items()) == list(
        map_b["unresolved_modules"].items()
    )

    assert list(map_a["source_to_tests"].keys()) == sorted(map_a["source_to_tests"].keys())
    assert list(map_a["test_to_sources"].keys()) == sorted(map_a["test_to_sources"].keys())
    assert list(map_a["source_paths"].keys()) == sorted(map_a["source_paths"].keys())
    assert list(map_a["unresolved_modules"].keys()) == sorted(
        map_a["unresolved_modules"].keys()
    )

    assert list(map_a["source_paths"].keys()) == [
        "gpt_trader.cli",
        "gpt_trader.monitoring",
    ]
    assert list(map_a["unresolved_modules"].keys()) == ["gpt_trader.missing.module"]
    assert map_a["source_to_tests"]["gpt_trader.cli"] == [alpha_path, cli_path]
    assert map_a["source_to_tests"]["gpt_trader.monitoring"] == [alpha_path, cli_path]
    assert map_a["source_to_tests"]["gpt_trader.missing.module"] == [alpha_path]
