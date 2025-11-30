#!/usr/bin/env python3
"""Query risk configuration for AI agent consumption.

This script provides a machine-readable interface to query risk parameters,
including:
- Current risk limits
- Per-symbol overrides
- Time-based configurations
- Kill switch status

Usage:
    python scripts/agents/query_risk_config.py [--profile PROFILE] [--format json|text]
    python scripts/agents/query_risk_config.py --field max_leverage
    python scripts/agents/query_risk_config.py --generate-schema

Output:
    JSON representation of risk configuration with field documentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


FIELD_DOCS = {
    "max_leverage": {
        "description": "Maximum allowed leverage multiplier",
        "type": "integer",
        "min": 1,
        "max": 10,
        "applies_to": "all",
    },
    "daily_loss_limit": {
        "description": "Maximum daily loss in quote currency",
        "type": "decimal",
        "unit": "USD",
        "applies_to": "all",
    },
    "max_exposure_pct": {
        "description": "Maximum portfolio exposure as decimal",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "applies_to": "all",
    },
    "max_position_pct_per_symbol": {
        "description": "Maximum position size per symbol as decimal",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "applies_to": "per_symbol",
    },
    "min_liquidation_buffer_pct": {
        "description": "Minimum buffer before liquidation",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "applies_to": "perps",
    },
    "leverage_max_per_symbol": {
        "description": "Per-symbol maximum leverage overrides",
        "type": "dict[str, int]",
        "applies_to": "per_symbol",
    },
    "max_notional_per_symbol": {
        "description": "Per-symbol maximum notional value",
        "type": "dict[str, decimal]",
        "unit": "USD",
        "applies_to": "per_symbol",
    },
    "slippage_guard_bps": {
        "description": "Slippage guard in basis points",
        "type": "integer",
        "min": 0,
        "max": 1000,
        "applies_to": "all",
    },
    "kill_switch_enabled": {
        "description": "Emergency kill switch status",
        "type": "boolean",
        "applies_to": "all",
    },
    "reduce_only_mode": {
        "description": "Force reduce-only mode for all orders",
        "type": "boolean",
        "applies_to": "all",
    },
    "daytime_start_utc": {
        "description": "Daytime period start (HH:MM UTC)",
        "type": "string",
        "format": "HH:MM",
        "applies_to": "time_based",
    },
    "daytime_end_utc": {
        "description": "Daytime period end (HH:MM UTC)",
        "type": "string",
        "format": "HH:MM",
        "applies_to": "time_based",
    },
    "day_leverage_max_per_symbol": {
        "description": "Per-symbol leverage limits during daytime",
        "type": "dict[str, int]",
        "applies_to": "time_based",
    },
    "night_leverage_max_per_symbol": {
        "description": "Per-symbol leverage limits during nighttime",
        "type": "dict[str, int]",
        "applies_to": "time_based",
    },
    "enable_pre_trade_liq_projection": {
        "description": "Enable pre-trade liquidation projection",
        "type": "boolean",
        "applies_to": "perps",
    },
}


def load_risk_config(profile: str | None = None) -> dict[str, Any]:
    """Load risk configuration from file or defaults."""
    from gpt_trader.orchestration.configuration.risk.model import RiskConfig

    config_path = None
    if profile:
        # Look for profile-specific risk config
        possible_paths = [
            Path(f"config/risk/{profile}.json"),
            Path(f"var/profiles/{profile}/risk.json"),
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path:
        config = RiskConfig.from_json(config_path)
    else:
        config = RiskConfig.from_env()

    return config.to_dict()


def enrich_with_docs(config: dict[str, Any]) -> dict[str, Any]:
    """Add documentation to each configuration field."""
    enriched = {}
    for key, value in config.items():
        field_info: dict[str, Any] = {"value": value}
        if key in FIELD_DOCS:
            field_info.update(FIELD_DOCS[key])
        enriched[key] = field_info
    return enriched


def generate_schema() -> dict[str, Any]:
    """Generate JSON schema for risk configuration."""
    from gpt_trader.orchestration.configuration.risk.model import RiskConfig

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "RiskConfig",
        "type": "object",
        "description": "Risk management configuration for GPT-Trader",
        "properties": {},
    }

    for field in fields(RiskConfig):
        field_name = field.name
        field_schema: dict[str, Any] = {}

        if field_name in FIELD_DOCS:
            docs = FIELD_DOCS[field_name]
            field_schema["description"] = docs["description"]

            if docs["type"] == "integer":
                field_schema["type"] = "integer"
                if "min" in docs:
                    field_schema["minimum"] = docs["min"]
                if "max" in docs:
                    field_schema["maximum"] = docs["max"]
            elif docs["type"] == "float":
                field_schema["type"] = "number"
                if "min" in docs:
                    field_schema["minimum"] = docs["min"]
                if "max" in docs:
                    field_schema["maximum"] = docs["max"]
            elif docs["type"] == "decimal":
                field_schema["type"] = "string"
                field_schema["format"] = "decimal"
            elif docs["type"] == "boolean":
                field_schema["type"] = "boolean"
            elif docs["type"] == "string":
                field_schema["type"] = "string"
                if "format" in docs:
                    field_schema["pattern"] = r"^\d{2}:\d{2}$"
            elif docs["type"].startswith("dict"):
                field_schema["type"] = "object"
                field_schema["additionalProperties"] = True

        schema["properties"][field_name] = field_schema

    return schema


def format_text(config: dict[str, Any]) -> str:
    """Format configuration as human-readable text."""
    lines = ["Risk Configuration", "=" * 40]
    for key, info in config.items():
        value = info.get("value", info)
        desc = info.get("description", "")
        if isinstance(value, dict) and not value:
            value = "{}"
        lines.append(f"{key}: {value}")
        if desc:
            lines.append(f"  ({desc})")
    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Query risk configuration")
    parser.add_argument(
        "--profile",
        type=str,
        help="Profile name to load config for",
    )
    parser.add_argument(
        "--field",
        type=str,
        help="Query a specific field",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--generate-schema",
        action="store_true",
        help="Generate JSON schema instead of config",
    )
    parser.add_argument(
        "--with-docs",
        action="store_true",
        help="Include field documentation in output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file (defaults to stdout)",
    )

    args = parser.parse_args()

    if args.generate_schema:
        schema = generate_schema()
        output = json.dumps(schema, indent=2)
        if args.output:
            args.output.write_text(output)
            print(f"Schema written to: {args.output}")
        else:
            print(output)
        return 0

    config = load_risk_config(args.profile)

    if args.field:
        if args.field not in config:
            print(f"Error: Unknown field '{args.field}'", file=sys.stderr)
            print(f"Available fields: {', '.join(config.keys())}", file=sys.stderr)
            return 1
        value = config[args.field]
        if args.with_docs and args.field in FIELD_DOCS:
            result = {"value": value, **FIELD_DOCS[args.field]}
        else:
            result = value
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.with_docs:
        config = enrich_with_docs(config)

    if args.format == "text":
        output = format_text(config)
    else:
        output = json.dumps(config, indent=2, default=str)

    if args.output:
        args.output.write_text(output)
        print(f"Config written to: {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
