#!/usr/bin/env python3
"""Generate reasoning artifacts for CLI flow and config linkage.

Outputs JSON, Markdown, and DOT artifacts under var/agents/reasoning/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "var" / "agents" / "reasoning"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))


CLI_FLOW_NODES = [
    {
        "id": "cli_entrypoint",
        "label": "CLI entrypoint (gpt_trader.cli:main)",
        "type": "entrypoint",
        "path": "src/gpt_trader/cli/__init__.py",
    },
    {
        "id": "cli_run_command",
        "label": "CLI run command",
        "type": "command",
        "path": "src/gpt_trader/cli/commands/run.py",
    },
    {
        "id": "cli_services",
        "label": "CLI config/services",
        "type": "services",
        "path": "src/gpt_trader/cli/services.py",
    },
    {
        "id": "profile_loader",
        "label": "ProfileLoader",
        "type": "config",
        "path": "src/gpt_trader/app/config/profile_loader.py",
    },
    {
        "id": "bot_config",
        "label": "BotConfig",
        "type": "config",
        "path": "src/gpt_trader/app/config/bot_config.py",
    },
    {
        "id": "bootstrap",
        "label": "build_bot / bot_from_profile",
        "type": "bootstrap",
        "path": "src/gpt_trader/app/bootstrap.py",
    },
    {
        "id": "container",
        "label": "ApplicationContainer",
        "type": "container",
        "path": "src/gpt_trader/app/container.py",
    },
    {
        "id": "trading_bot",
        "label": "TradingBot",
        "type": "runtime",
        "path": "src/gpt_trader/features/live_trade/bot.py",
    },
    {
        "id": "trading_engine",
        "label": "TradingEngine",
        "type": "runtime",
        "path": "src/gpt_trader/features/live_trade/engines/strategy.py",
    },
    {
        "id": "strategy_factory",
        "label": "create_strategy",
        "type": "strategy",
        "path": "src/gpt_trader/features/live_trade/factory.py",
    },
]

CLI_FLOW_EDGES = [
    {
        "from": "cli_entrypoint",
        "to": "cli_run_command",
        "label": "dispatch to run command",
    },
    {
        "from": "cli_run_command",
        "to": "cli_services",
        "label": "build config + instantiate bot",
    },
    {
        "from": "cli_services",
        "to": "profile_loader",
        "label": "load profile schema",
    },
    {
        "from": "profile_loader",
        "to": "bot_config",
        "label": "construct BotConfig",
    },
    {
        "from": "cli_services",
        "to": "container",
        "label": "create ApplicationContainer",
    },
    {
        "from": "bootstrap",
        "to": "container",
        "label": "optional bootstrap path",
    },
    {
        "from": "container",
        "to": "trading_bot",
        "label": "create bot",
    },
    {
        "from": "trading_bot",
        "to": "trading_engine",
        "label": "instantiate engine",
    },
    {
        "from": "trading_engine",
        "to": "strategy_factory",
        "label": "select strategy",
    },
]

GUARD_STACK_CLUSTERS = [
    {"id": "preflight", "label": "Preflight"},
    {"id": "runtime", "label": "Runtime Guards + Monitoring"},
]

GUARD_STACK_NODES = [
    {
        "id": "preflight_entry",
        "label": "Preflight entrypoint",
        "type": "entrypoint",
        "path": "scripts/production_preflight.py",
        "cluster": "preflight",
    },
    {
        "id": "preflight_cli",
        "label": "Preflight CLI",
        "type": "cli",
        "path": "src/gpt_trader/preflight/cli.py",
        "cluster": "preflight",
    },
    {
        "id": "preflight_core",
        "label": "PreflightCheck",
        "type": "core",
        "path": "src/gpt_trader/preflight/core.py",
        "cluster": "preflight",
    },
    {
        "id": "preflight_checks",
        "label": "Preflight checks",
        "type": "checks",
        "path": "src/gpt_trader/preflight/checks/",
        "cluster": "preflight",
    },
    {
        "id": "preflight_report",
        "label": "Preflight report",
        "type": "report",
        "path": "src/gpt_trader/preflight/report.py",
        "cluster": "preflight",
    },
    {
        "id": "trading_engine",
        "label": "TradingEngine",
        "type": "runtime",
        "path": "src/gpt_trader/features/live_trade/engines/strategy.py",
        "cluster": "runtime",
    },
    {
        "id": "execution_guard_manager",
        "label": "GuardManager (execution)",
        "type": "runtime_guard",
        "path": "src/gpt_trader/features/live_trade/execution/guard_manager.py",
        "cluster": "runtime",
    },
    {
        "id": "execution_guards",
        "label": "Execution guards",
        "type": "guards",
        "path": "src/gpt_trader/features/live_trade/execution/guards/",
        "cluster": "runtime",
    },
    {
        "id": "monitoring_guard_manager",
        "label": "RuntimeGuardManager",
        "type": "monitoring",
        "path": "src/gpt_trader/monitoring/guards/manager.py",
        "cluster": "runtime",
    },
    {
        "id": "monitoring_guards",
        "label": "Monitoring guards",
        "type": "monitoring",
        "path": "src/gpt_trader/monitoring/guards/builtins.py",
        "cluster": "runtime",
    },
    {
        "id": "health_signals",
        "label": "Health signals",
        "type": "monitoring",
        "path": "src/gpt_trader/monitoring/health_signals.py",
        "cluster": "runtime",
    },
    {
        "id": "health_checks",
        "label": "Health checks",
        "type": "monitoring",
        "path": "src/gpt_trader/monitoring/health_checks.py",
        "cluster": "runtime",
    },
    {
        "id": "status_reporter",
        "label": "Status reporter",
        "type": "monitoring",
        "path": "src/gpt_trader/monitoring/status_reporter.py",
        "cluster": "runtime",
    },
]

GUARD_STACK_EDGES = [
    {
        "from": "preflight_entry",
        "to": "preflight_cli",
        "label": "delegate CLI",
    },
    {
        "from": "preflight_cli",
        "to": "preflight_core",
        "label": "create PreflightCheck",
    },
    {
        "from": "preflight_core",
        "to": "preflight_checks",
        "label": "run checks",
    },
    {
        "from": "preflight_core",
        "to": "preflight_report",
        "label": "generate report",
    },
    {
        "from": "trading_engine",
        "to": "execution_guard_manager",
        "label": "runtime guard sweep",
    },
    {
        "from": "execution_guard_manager",
        "to": "execution_guards",
        "label": "execute runtime guards",
    },
    {
        "from": "trading_engine",
        "to": "execution_guards",
        "label": "pre-trade guard stack",
    },
    {
        "from": "trading_engine",
        "to": "monitoring_guard_manager",
        "label": "emit guard events",
    },
    {
        "from": "monitoring_guard_manager",
        "to": "monitoring_guards",
        "label": "evaluate guards",
    },
    {
        "from": "monitoring_guards",
        "to": "health_signals",
        "label": "emit health signals",
    },
    {
        "from": "health_signals",
        "to": "health_checks",
        "label": "evaluate thresholds",
    },
    {
        "from": "health_checks",
        "to": "status_reporter",
        "label": "report status",
    },
]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_markdown(path: Path, content: str) -> None:
    path.write_text(content)


def _write_dot(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines))


def build_cli_flow_map() -> dict[str, Any]:
    return {
        "artifact": "cli_flow_map",
        "generated_at": _timestamp(),
        "description": "CLI → config → container → engine flow map.",
        "entrypoints": [
            "uv run gpt-trader run --profile dev --dev-fast",
            "uv run coinbase-trader run --profile dev --dev-fast",
        ],
        "nodes": CLI_FLOW_NODES,
        "edges": CLI_FLOW_EDGES,
    }


def build_cli_flow_markdown(flow: dict[str, Any]) -> str:
    lines = [
        "# CLI → Config → Container → Engine Flow",
        "",
        f"Generated: {flow['generated_at']}",
        "",
        "## Entrypoints",
        *[f"- `{entry}`" for entry in flow.get("entrypoints", [])],
        "",
        "## Nodes",
        "| ID | Label | Path |",
        "|----|-------|------|",
    ]
    for node in flow["nodes"]:
        lines.append(f"| {node['id']} | {node['label']} | `{node['path']}` |")

    lines.extend(["", "## Edges", "| From | To | Description |", "|------|----|-------------|"])
    for edge in flow["edges"]:
        lines.append(f"| {edge['from']} | {edge['to']} | {edge['label']} |")

    return "\n".join(lines) + "\n"


def build_cli_flow_dot(flow: dict[str, Any]) -> list[str]:
    lines = ["digraph CliFlow {", "  rankdir=LR;"]
    for node in flow["nodes"]:
        label = f"{node['label']}\\n{node['path']}"
        lines.append(f'  "{node["id"]}" [shape=box, label="{label}"]; ')
    for edge in flow["edges"]:
        lines.append(f'  "{edge["from"]}" -> "{edge["to"]}" [label="{edge["label"]}"];')
    lines.append("}")
    return lines


def build_guard_stack_map() -> dict[str, Any]:
    return {
        "artifact": "guard_stack_map",
        "generated_at": _timestamp(),
        "description": "Guard stack map (preflight checks vs runtime guards + monitoring).",
        "clusters": GUARD_STACK_CLUSTERS,
        "nodes": GUARD_STACK_NODES,
        "edges": GUARD_STACK_EDGES,
        "notes": [
            "Preflight checks run via the preflight CLI and PreflightCheck facade.",
            "Runtime guard sweep is owned by TradingEngine and GuardManager.",
        ],
    }


def build_guard_stack_markdown(guard_map: dict[str, Any]) -> str:
    lines = [
        "# Guard Stack Map",
        "",
        f"Generated: {guard_map['generated_at']}",
        "",
    ]

    cluster_index = {cluster["id"]: cluster for cluster in guard_map["clusters"]}
    for cluster_id in [cluster["id"] for cluster in guard_map["clusters"]]:
        cluster = cluster_index[cluster_id]
        lines.append(f"## {cluster['label']}")
        lines.append("| ID | Label | Path |")
        lines.append("|----|-------|------|")
        for node in guard_map["nodes"]:
            if node["cluster"] == cluster_id:
                lines.append(f"| {node['id']} | {node['label']} | `{node['path']}` |")
        lines.append("")

    lines.extend(["## Edges", "| From | To | Description |", "|------|----|-------------|"])
    for edge in guard_map["edges"]:
        lines.append(f"| {edge['from']} | {edge['to']} | {edge['label']} |")

    lines.append("")
    lines.append("## Notes")
    for note in guard_map.get("notes", []):
        lines.append(f"- {note}")

    return "\n".join(lines) + "\n"


def build_guard_stack_dot(guard_map: dict[str, Any]) -> list[str]:
    lines = ["digraph GuardStack {", "  rankdir=LR;"]
    for cluster in guard_map["clusters"]:
        lines.append(f"  subgraph cluster_{cluster['id']} {{")
        lines.append(f'    label="{cluster["label"]}";')
        lines.append("    style=rounded;")
        for node in guard_map["nodes"]:
            if node["cluster"] != cluster["id"]:
                continue
            label = f"{node['label']}\\n{node['path']}"
            lines.append(f'    "{node["id"]}" [shape=box, label="{label}"]; ')
        lines.append("  }")

    for edge in guard_map["edges"]:
        lines.append(f'  "{edge["from"]}" -> "{edge["to"]}" [label="{edge["label"]}"];')
    lines.append("}")
    return lines


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if path.parts and path.parts[0] == "tests":
            continue
        yield path


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _find_usage(patterns: list[re.Pattern[str]], files: list[Path]) -> list[str]:
    hits: list[str] = []
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        if any(pattern.search(content) for pattern in patterns):
            hits.append(_relative_path(file_path))
    return sorted(hits)


def _compile_patterns(prefix: str, field_name: str) -> list[re.Pattern[str]]:
    escaped = re.escape(field_name)
    prefix_escaped = re.escape(prefix)
    base_patterns = [
        rf"\\b{prefix_escaped}\\.{escaped}\\b",
        rf"\\bcontext\\.{prefix_escaped}\\.{escaped}\\b",
        rf"\\bself\\.config\\.{escaped}\\b",
        rf"\\bbot\\.config\\.{escaped}\\b",
        rf"\\bbot_config\\.{escaped}\\b",
    ]
    return [re.compile(pattern) for pattern in base_patterns]


def build_config_code_map() -> dict[str, Any]:
    from gpt_trader.app.config.bot_config import (
        BotConfig,
        BotRiskConfig,
        HealthThresholdsConfig,
        MeanReversionConfig,
    )
    from gpt_trader.features.live_trade.strategies.perps_baseline import PerpsStrategyConfig

    files = list(_iter_python_files(SRC_ROOT))

    skip_fields = {
        "strategy",
        "risk",
        "mean_reversion",
        "health_thresholds",
        "regime_config",
        "ensemble_config",
    }

    sections = [
        {
            "name": "bot_config",
            "label": "BotConfig (top-level)",
            "prefix": "config",
            "fields": [f.name for f in fields(BotConfig) if f.name not in skip_fields],
        },
        {
            "name": "risk",
            "label": "BotRiskConfig",
            "prefix": "config.risk",
            "fields": [f.name for f in fields(BotRiskConfig)],
        },
        {
            "name": "strategy",
            "label": "PerpsStrategyConfig",
            "prefix": "config.strategy",
            "fields": [f.name for f in fields(PerpsStrategyConfig)],
        },
        {
            "name": "mean_reversion",
            "label": "MeanReversionConfig",
            "prefix": "config.mean_reversion",
            "fields": [f.name for f in fields(MeanReversionConfig)],
        },
        {
            "name": "health_thresholds",
            "label": "HealthThresholdsConfig",
            "prefix": "config.health_thresholds",
            "fields": [f.name for f in fields(HealthThresholdsConfig)],
        },
    ]

    alias_fields = {
        "short_ma": "strategy.short_ma_period",
        "long_ma": "strategy.long_ma_period",
        "target_leverage": "risk.target_leverage",
        "max_leverage": "risk.max_leverage",
        "trailing_stop_pct": "strategy.trailing_stop_pct or risk.trailing_stop_pct",
        "active_enable_shorts": "strategy.enable_shorts / mean_reversion.enable_shorts",
        "is_hybrid_mode": "trading_modes contains spot+cfm",
        "is_cfm_only": "trading_modes contains only cfm",
        "is_spot_only": "trading_modes contains only spot",
    }

    section_entries: list[dict[str, Any]] = []
    for section in sections:
        entries = []
        for field_name in section["fields"]:
            patterns = _compile_patterns(section["prefix"], field_name)
            usage = _find_usage(patterns, files)
            entries.append(
                {
                    "field": field_name,
                    "usage_count": len(usage),
                    "files": usage,
                    "notes": "No direct usage found" if not usage else "",
                }
            )
        entries.sort(key=lambda item: item["field"])
        section_entries.append(
            {
                "section": section["name"],
                "label": section["label"],
                "prefix": section["prefix"],
                "fields": entries,
            }
        )

    alias_entries = []
    for alias, target in alias_fields.items():
        patterns = _compile_patterns("config", alias)
        usage = _find_usage(patterns, files)
        alias_entries.append(
            {
                "alias": alias,
                "target": target,
                "usage_count": len(usage),
                "files": usage,
                "notes": "No direct usage found" if not usage else "",
            }
        )
    alias_entries.sort(key=lambda item: item["alias"])

    return {
        "artifact": "config_code_map",
        "generated_at": _timestamp(),
        "description": "Config field → code linkage map based on static scan.",
        "scan_root": "src/gpt_trader",
        "sections": section_entries,
        "aliases": alias_entries,
        "notes": [
            "Scan uses simple regex matching (config.<field>) across src/gpt_trader.",
            "Dynamic config access or indirect usage may not appear in results.",
        ],
    }


def build_config_code_markdown(config_map: dict[str, Any]) -> str:
    lines = [
        "# Config → Code Linkage Map",
        "",
        f"Generated: {config_map['generated_at']}",
        "",
        f"Scan root: `{config_map['scan_root']}`",
        "",
    ]

    for section in config_map["sections"]:
        lines.extend(
            [
                f"## {section['label']}",
                "| Field | Usage count | Example files |",
                "|-------|-------------|---------------|",
            ]
        )
        for field_entry in section["fields"]:
            examples = field_entry["files"][:3]
            example_text = ", ".join(f"`{item}`" for item in examples) if examples else "—"
            lines.append(
                f"| {field_entry['field']} | {field_entry['usage_count']} | {example_text} |"
            )
        lines.append("")

    lines.append("## Alias Fields")
    lines.append("| Alias | Canonical target | Usage count | Example files |")
    lines.append("|-------|------------------|-------------|---------------|")
    for alias_entry in config_map["aliases"]:
        examples = alias_entry["files"][:3]
        example_text = ", ".join(f"`{item}`" for item in examples) if examples else "—"
        lines.append(
            "| {alias} | {target} | {count} | {examples} |".format(
                alias=alias_entry["alias"],
                target=alias_entry["target"],
                count=alias_entry["usage_count"],
                examples=example_text,
            )
        )

    lines.append("")
    lines.append("## Notes")
    for note in config_map.get("notes", []):
        lines.append(f"- {note}")

    return "\n".join(lines) + "\n"


def build_config_code_dot(config_map: dict[str, Any]) -> list[str]:
    lines = ["digraph ConfigCode {", "  rankdir=LR;"]
    seen_nodes: set[str] = set()

    for section in config_map["sections"]:
        section_label = section["label"]
        if section_label not in seen_nodes:
            lines.append(f'  "{section_label}" [shape=box];')
            seen_nodes.add(section_label)
        file_counts: dict[str, int] = {}
        for field_entry in section["fields"]:
            for file_path in field_entry["files"]:
                file_counts[file_path] = file_counts.get(file_path, 0) + 1
        top_files = sorted(file_counts.items(), key=lambda item: item[1], reverse=True)[:12]
        for file_path, count in top_files:
            if file_path not in seen_nodes:
                lines.append(f'  "{file_path}" [shape=ellipse];')
                seen_nodes.add(file_path)
            lines.append(f'  "{section_label}" -> "{file_path}" [label="{count}"];')

    lines.append("}")
    return lines


def generate(output_dir: Path) -> dict[str, Path]:
    _ensure_output_dir()

    cli_flow = build_cli_flow_map()
    cli_flow_json = output_dir / "cli_flow_map.json"
    cli_flow_md = output_dir / "cli_flow_map.md"
    cli_flow_dot = output_dir / "cli_flow_map.dot"

    _write_json(cli_flow_json, cli_flow)
    _write_markdown(cli_flow_md, build_cli_flow_markdown(cli_flow))
    _write_dot(cli_flow_dot, build_cli_flow_dot(cli_flow))

    guard_map = build_guard_stack_map()
    guard_json = output_dir / "guard_stack_map.json"
    guard_md = output_dir / "guard_stack_map.md"
    guard_dot = output_dir / "guard_stack_map.dot"

    _write_json(guard_json, guard_map)
    _write_markdown(guard_md, build_guard_stack_markdown(guard_map))
    _write_dot(guard_dot, build_guard_stack_dot(guard_map))

    config_map = build_config_code_map()
    config_json = output_dir / "config_code_map.json"
    config_md = output_dir / "config_code_map.md"
    config_dot = output_dir / "config_code_map.dot"

    _write_json(config_json, config_map)
    _write_markdown(config_md, build_config_code_markdown(config_map))
    _write_dot(config_dot, build_config_code_dot(config_map))

    return {
        "cli_flow_map.json": cli_flow_json,
        "cli_flow_map.md": cli_flow_md,
        "cli_flow_map.dot": cli_flow_dot,
        "guard_stack_map.json": guard_json,
        "guard_stack_map.md": guard_md,
        "guard_stack_map.dot": guard_dot,
        "config_code_map.json": config_json,
        "config_code_map.md": config_md,
        "config_code_map.dot": config_dot,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reasoning artifacts (CLI flow + config map)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: var/agents/reasoning)",
    )
    args = parser.parse_args()

    outputs = generate(args.output_dir)
    print("Generated reasoning artifacts:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
