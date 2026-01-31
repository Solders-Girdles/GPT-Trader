from __future__ import annotations

from typing import Any

from scripts.agents import generate_test_inventory


def _build_scan_results(
    inventory: dict[str, list[dict[str, Any]]],
    marker_counts: dict[str, int],
) -> dict[str, Any]:
    total_tests = sum(len(tests) for tests in inventory.values())
    return {
        "inventory": inventory,
        "marker_counts": marker_counts,
        "total_tests": total_tests,
        "total_files": len(inventory),
        "imports_by_file": {},
    }


def test_generate_test_inventory_is_deterministic_for_unsorted_inputs() -> None:
    cli_tests = [
        {"name": "test_zebra", "line": 20, "markers": ["beta"], "docstring": ""},
        {"name": "test_alpha", "line": 3, "markers": ["alpha"], "docstring": ""},
    ]
    feature_tests = [
        {
            "name": "test_feature",
            "line": 10,
            "markers": ["alpha", "beta"],
            "docstring": "",
        }
    ]

    inventory_first = {
        "tests/unit/gpt_trader/features/test_feature.py": feature_tests,
        "tests/unit/gpt_trader/cli/test_cli.py": cli_tests,
    }
    inventory_second = dict(
        [
            ("tests/unit/gpt_trader/cli/test_cli.py", list(reversed(cli_tests))),
            ("tests/unit/gpt_trader/features/test_feature.py", list(reversed(feature_tests))),
        ]
    )

    scan_results_first = _build_scan_results(
        inventory_first,
        {"beta": 2, "alpha": 2},
    )
    scan_results_second = _build_scan_results(
        inventory_second,
        {"alpha": 2, "beta": 2},
    )

    marker_definitions_first = {"beta": "beta marker", "alpha": "alpha marker"}
    marker_definitions_second = {"alpha": "alpha marker", "beta": "beta marker"}

    first_inventory = generate_test_inventory.generate_test_inventory(
        scan_results_first,
        marker_definitions_first,
    )
    second_inventory = generate_test_inventory.generate_test_inventory(
        scan_results_second,
        marker_definitions_second,
    )

    assert first_inventory == second_inventory
    assert list(first_inventory["tests_by_file"]) == sorted(first_inventory["tests_by_file"])
    cli_lines = [
        test["line"]
        for test in first_inventory["tests_by_file"]["tests/unit/gpt_trader/cli/test_cli.py"]
    ]
    assert cli_lines == sorted(test["line"] for test in cli_tests)


def test_build_source_test_map_is_deterministic_for_unsorted_inputs() -> None:
    imports_first = {
        "tests/unit/gpt_trader/cli/test_cli.py": [
            "gpt_trader.cli",
            "gpt_trader.agents",
            "gpt_trader.cli",
        ],
        "tests/unit/gpt_trader/features/test_feature.py": [
            "gpt_trader.features.live_trade",
            "gpt_trader.cli",
        ],
    }
    imports_second = dict(
        [
            (
                "tests/unit/gpt_trader/features/test_feature.py",
                ["gpt_trader.cli", "gpt_trader.features.live_trade"],
            ),
            (
                "tests/unit/gpt_trader/cli/test_cli.py",
                ["gpt_trader.agents", "gpt_trader.cli"],
            ),
        ]
    )

    first_map = generate_test_inventory.build_source_test_map(imports_first)
    second_map = generate_test_inventory.build_source_test_map(imports_second)

    assert first_map == second_map
    assert first_map["source_to_tests"]["gpt_trader.cli"] == [
        "tests/unit/gpt_trader/cli/test_cli.py",
        "tests/unit/gpt_trader/features/test_feature.py",
    ]
