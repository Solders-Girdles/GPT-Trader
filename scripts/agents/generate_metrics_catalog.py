#!/usr/bin/env python3
"""Generate a catalog of Prometheus-style metrics used by the runtime.

This script statically scans the codebase for calls to:
- gpt_trader.monitoring.metrics_collector.record_counter
- gpt_trader.monitoring.metrics_collector.record_gauge
- gpt_trader.monitoring.metrics_collector.record_histogram

It extracts metric names (gpt_trader_*) plus any label keys found in literal
labels dicts, and writes a deterministic catalog under var/agents/.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src" / "gpt_trader"

METRIC_FUNCTION_TYPES: dict[str, str] = {
    "record_counter": "counter",
    "record_gauge": "gauge",
    "record_histogram": "histogram",
}


@dataclass(frozen=True)
class MetricLocation:
    path: str
    line: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate metrics catalog for AI agents.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "var" / "agents" / "observability",
        help="Output directory for catalog files (default: var/agents/observability).",
    )
    return parser.parse_args()


def _extract_module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            if not isinstance(node.value.value, str):
                continue
            if isinstance(node.target, ast.Name):
                constants[node.target.id] = node.value.value
    return constants


def _extract_metrics_import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "gpt_trader.monitoring.metrics_collector":
            continue
        for item in node.names:
            if item.name not in METRIC_FUNCTION_TYPES:
                continue
            local_name = item.asname or item.name
            aliases[local_name] = item.name
    return aliases


def _extract_label_keys(node: ast.Call) -> set[str]:
    for keyword in node.keywords:
        if keyword.arg != "labels":
            continue
        value = keyword.value
        if not isinstance(value, ast.Dict):
            return set()
        keys: set[str] = set()
        for key in value.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                return set()
            keys.add(key.value)
        return keys
    return set()


def _resolve_metric_name(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def collect_metrics() -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}

    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel_path = path.relative_to(PROJECT_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
        except SyntaxError:
            continue

        constants = _extract_module_string_constants(tree)
        import_aliases = _extract_metrics_import_aliases(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Name):
                continue

            canonical = import_aliases.get(func.id, func.id)
            if canonical not in METRIC_FUNCTION_TYPES:
                continue
            if not node.args:
                continue

            metric_name = _resolve_metric_name(node.args[0], constants)
            if not metric_name or not metric_name.startswith("gpt_trader_"):
                continue

            metric_type = METRIC_FUNCTION_TYPES[canonical]
            label_keys = _extract_label_keys(node)
            location = MetricLocation(path=rel_path, line=getattr(node, "lineno", 0) or 0)

            entry = metrics.setdefault(
                metric_name,
                {"types": set(), "label_keys": set(), "locations": []},
            )
            entry["types"].add(metric_type)
            entry["label_keys"].update(label_keys)
            entry["locations"].append(location)

    return metrics


def render_markdown(metrics: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Metrics Catalog")
    lines.append("")
    lines.append("Generated by `scripts/agents/generate_metrics_catalog.py`.")
    lines.append("")
    lines.append("| Metric | Type | Label keys | Defined in |")
    lines.append("|--------|------|------------|------------|")

    for metric in metrics:
        types = ", ".join(metric["types"]) if metric["types"] else "unknown"
        label_keys = ", ".join(metric["label_keys"]) if metric["label_keys"] else ""
        locations = ", ".join(f"{loc['path']}:{loc['line']}" for loc in metric["locations"])
        lines.append(f"| `{metric['name']}` | {types} | {label_keys} | {locations} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = collect_metrics()

    metrics: list[dict[str, Any]] = []
    for name in sorted(raw):
        entry = raw[name]
        locations = sorted(entry["locations"], key=lambda loc: (loc.path, loc.line))
        metrics.append(
            {
                "name": name,
                "types": sorted(entry["types"]),
                "label_keys": sorted(entry["label_keys"]),
                "locations": [{"path": loc.path, "line": loc.line} for loc in locations],
            }
        )

    payload = {
        "version": "1.0",
        "metrics": metrics,
    }

    (output_dir / "metrics_catalog.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics_catalog.md").write_text(render_markdown(metrics), encoding="utf-8")
    (output_dir / "index.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "description": "Prometheus-style metric catalog (static scan).",
                "files": {
                    "metrics_catalog_json": "metrics_catalog.json",
                    "metrics_catalog_markdown": "metrics_catalog.md",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
