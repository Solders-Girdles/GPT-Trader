"""Test categorisation tooling built on the dependency analysis report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set
from collections.abc import Iterable


DEFAULT_REPORT = Path("dependency_report.json")

TARGET_CATEGORIES = {
    "orchestration": {
        "gpt_trader.orchestration",
        "gpt_trader.cli",
    },
    "execution_helper": {
        "gpt_trader.orchestration.live_execution",
        "gpt_trader.features.live_trade.advanced_execution",
        "gpt_trader.features.live_trade.order_policy",
    },
    "strategy_helper": {
        "gpt_trader.features.live_trade.strategies",
        "gpt_trader.features.live_trade.risk",
        "gpt_trader.features.strategy_tools",
    },
    "pipeline_stage": {
        "gpt_trader.orchestration.trading_bot",
    },
    "integration": {
        "gpt_trader.features.brokerages",
        "gpt_trader.persistence",
        "gpt_trader.data_providers",
    },
    "smoke": {
        "gpt_trader.scripts",
        "gpt_trader.monitoring",
    },
}


def load_report(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Dependency report not found at {path}. Run dependency_map.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def categorize_tests(
    tests_to_modules: dict[str, list[str]],
    category_map: dict[str, set[str]],
) -> dict[str, set[str]]:
    categorized: dict[str, set[str]] = defaultdict(set)

    for test_path, modules in tests_to_modules.items():
        assigned = False
        module_set = set(modules)
        for category, prefixes in category_map.items():
            if any(any(mod.startswith(prefix) for prefix in prefixes) for mod in module_set):
                categorized[category].add(test_path)
                assigned = True
        if not assigned:
            categorized["uncategorized"].add(test_path)
    return categorized


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize tests by dependency grouping")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="Path to JSON report produced by dependency_map.py",
    )
    parser.add_argument("--output", type=Path, help="Write mapping to this JSON file")
    parser.add_argument("--show", action="store_true", help="Print summary to stdout")
    args = parser.parse_args()

    report = load_report(args.report)
    tests_to_modules = report.get("tests_to_modules")
    if not tests_to_modules:
        raise SystemExit(
            "tests_to_modules data missing in report. Run dependency_map.py with --tests option."
        )

    categorized = categorize_tests(
        {test: set(mods) for test, mods in tests_to_modules.items()},
        {key: set(value) for key, value in TARGET_CATEGORIES.items()},
    )

    output_payload = {category: sorted(paths) for category, paths in sorted(categorized.items())}

    if args.output:
        args.output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    if args.show or not args.output:
        print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
