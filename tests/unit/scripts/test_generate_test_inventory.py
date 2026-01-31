from __future__ import annotations

from scripts.agents import generate_test_inventory


def test_generate_test_inventory_orders_output() -> None:
    scan_results = {
        "inventory": {
            "tests/unit/gpt_trader/core/test_core.py": [
                {"name": "test_gamma", "line": 5, "markers": ["slow"]},
                {"name": "test_beta", "line": 5, "markers": ["unit"]},
            ],
            "tests/unit/gpt_trader/cli/test_cli.py": [
                {"name": "test_delta", "line": 20, "markers": ["api"]},
                {"name": "test_alpha", "line": 10, "markers": ["api"]},
            ],
        },
        "marker_counts": {"unit": 3, "api": 3, "slow": 1},
        "total_tests": 4,
        "total_files": 2,
        "imports_by_file": {},
    }
    marker_defs = {
        "slow": "Slow tests",
        "unit": "Unit tests",
        "api": "API tests",
    }

    inventory = generate_test_inventory.generate_test_inventory(scan_results, marker_defs)

    assert list(inventory["tests_by_file"].keys()) == [
        "tests/unit/gpt_trader/cli/test_cli.py",
        "tests/unit/gpt_trader/core/test_core.py",
    ]
    assert [
        test["name"] for test in inventory["tests_by_file"]["tests/unit/gpt_trader/cli/test_cli.py"]
    ] == [
        "test_alpha",
        "test_delta",
    ]
    assert [
        test["name"]
        for test in inventory["tests_by_file"]["tests/unit/gpt_trader/core/test_core.py"]
    ] == [
        "test_beta",
        "test_gamma",
    ]
    assert list(inventory["marker_definitions"].keys()) == ["api", "slow", "unit"]
    assert list(inventory["marker_counts"].keys()) == ["api", "unit", "slow"]
    assert list(inventory["path_categories"].keys()) == ["cli", "core"]
    assert inventory["path_categories"]["cli"] == ["tests/unit/gpt_trader/cli/test_cli.py"]
    assert inventory["path_categories"]["core"] == ["tests/unit/gpt_trader/core/test_core.py"]


def test_build_source_test_map_orders_entries() -> None:
    imports_by_file = {
        "tests/unit/gpt_trader/core/test_core.py": [
            "gpt_trader.core",
            "gpt_trader.cli",
            "gpt_trader.core",
        ],
        "tests/unit/gpt_trader/cli/test_cli.py": [
            "gpt_trader.cli",
            "gpt_trader.core",
        ],
    }

    source_map = generate_test_inventory.build_source_test_map(imports_by_file)

    assert list(source_map["test_to_sources"].keys()) == [
        "tests/unit/gpt_trader/cli/test_cli.py",
        "tests/unit/gpt_trader/core/test_core.py",
    ]
    assert source_map["test_to_sources"]["tests/unit/gpt_trader/cli/test_cli.py"] == [
        "gpt_trader.cli",
        "gpt_trader.core",
    ]
    assert source_map["test_to_sources"]["tests/unit/gpt_trader/core/test_core.py"] == [
        "gpt_trader.cli",
        "gpt_trader.core",
    ]
    assert list(source_map["source_to_tests"].keys()) == [
        "gpt_trader.cli",
        "gpt_trader.core",
    ]
    assert source_map["source_to_tests"]["gpt_trader.cli"] == [
        "tests/unit/gpt_trader/cli/test_cli.py",
        "tests/unit/gpt_trader/core/test_core.py",
    ]
    assert list(source_map["source_paths"].keys()) == [
        "gpt_trader.cli",
        "gpt_trader.core",
    ]

    expected_cli_path = generate_test_inventory.module_to_path("gpt_trader.cli")
    expected_core_path = generate_test_inventory.module_to_path("gpt_trader.core")
    assert expected_cli_path is not None
    assert expected_core_path is not None
    assert source_map["source_paths"]["gpt_trader.cli"] == expected_cli_path
    assert source_map["source_paths"]["gpt_trader.core"] == expected_core_path
