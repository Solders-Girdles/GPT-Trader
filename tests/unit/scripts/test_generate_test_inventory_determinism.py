from __future__ import annotations

from typing import Any

from scripts.agents import generate_test_inventory


def _build_scan_results(
    file_order: list[str],
    marker_order: list[str],
) -> dict[str, Any]:
    inventory_data = {
        "tests/unit/gpt_trader/cli/test_alpha.py": [
            {
                "name": "test_alpha_last",
                "line": 42,
                "markers": ["risk", "unit"],
                "docstring": "",
            },
            {
                "name": "test_alpha_first",
                "line": 7,
                "markers": ["unit"],
                "docstring": "",
            },
        ],
        "tests/unit/gpt_trader/cli/test_beta.py": [
            {
                "name": "test_beta_first",
                "line": 3,
                "markers": ["unit"],
                "docstring": "",
            },
        ],
    }

    inventory = {path: inventory_data[path] for path in file_order}
    marker_counts = {marker: {"unit": 3, "risk": 1}[marker] for marker in marker_order}

    return {
        "inventory": inventory,
        "marker_counts": marker_counts,
        "total_tests": 3,
        "total_files": 2,
    }


def test_generate_test_inventory_is_deterministic() -> None:
    scan_results_a = _build_scan_results(
        [
            "tests/unit/gpt_trader/cli/test_beta.py",
            "tests/unit/gpt_trader/cli/test_alpha.py",
        ],
        ["unit", "risk"],
    )
    scan_results_b = _build_scan_results(
        [
            "tests/unit/gpt_trader/cli/test_alpha.py",
            "tests/unit/gpt_trader/cli/test_beta.py",
        ],
        ["risk", "unit"],
    )

    marker_defs_a = {"unit": "Unit tests", "risk": "Risk tests"}
    marker_defs_b = {"risk": "Risk tests", "unit": "Unit tests"}

    result_a = generate_test_inventory.generate_test_inventory(scan_results_a, marker_defs_a)
    result_b = generate_test_inventory.generate_test_inventory(scan_results_b, marker_defs_b)

    assert result_a == result_b

    expected_file_order = [
        "tests/unit/gpt_trader/cli/test_alpha.py",
        "tests/unit/gpt_trader/cli/test_beta.py",
    ]
    assert list(result_a["tests_by_file"].keys()) == expected_file_order

    alpha_tests = result_a["tests_by_file"][expected_file_order[0]]
    assert [test["name"] for test in alpha_tests] == [
        "test_alpha_first",
        "test_alpha_last",
    ]

    assert list(result_a["marker_definitions"].keys()) == ["risk", "unit"]
    assert list(result_a["marker_counts"].keys()) == ["unit", "risk"]
    assert list(result_a["path_categories"].keys()) == ["cli"]
    assert result_a["path_categories"]["cli"] == expected_file_order
