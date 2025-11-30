#!/usr/bin/env python3
"""Generate JSON schemas from configuration dataclasses for AI agent consumption.

Usage:
    python scripts/agents/generate_config_schemas.py [--output-dir DIR] [--format json|yaml]

Output:
    Generates JSON Schema v7 documents for:
    - BotConfig (trading parameters)
    - RiskConfig (risk management)

Example:
    python scripts/agents/generate_config_schemas.py --output-dir var/agents/schemas
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import MISSING, fields
from decimal import Decimal
from pathlib import Path
from typing import Any, get_args, get_origin

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def python_type_to_json_schema(python_type: type, field_name: str = "") -> dict[str, Any]:
    """Convert a Python type annotation to JSON Schema."""
    origin = get_origin(python_type)
    args = get_args(python_type)

    # Handle None/NoneType
    if python_type is type(None):
        return {"type": "null"}

    # Handle Union types (including Optional)
    if origin is type(None) or (hasattr(python_type, "__origin__") and origin is None):
        return {"type": "null"}

    # Handle Optional[X] which is Union[X, None]
    if origin is type(None):
        return {"type": "null"}

    # Check for Union/Optional
    try:
        from types import UnionType

        if origin is UnionType or str(origin) == "typing.Union":
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                schema = python_type_to_json_schema(non_none_types[0], field_name)
                schema["nullable"] = True
                return schema
            else:
                return {"oneOf": [python_type_to_json_schema(t, field_name) for t in args]}
    except ImportError:
        pass

    # Handle | union syntax (Python 3.10+)
    if str(origin) == "types.UnionType":
        non_none_types = [t for t in args if t is not type(None)]
        if len(non_none_types) == 1:
            schema = python_type_to_json_schema(non_none_types[0], field_name)
            schema["nullable"] = True
            return schema
        return {"oneOf": [python_type_to_json_schema(t, field_name) for t in args]}

    # Handle list
    if origin is list:
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": python_type_to_json_schema(item_type, field_name),
        }

    # Handle dict
    if origin is dict:
        key_type = args[0] if args else str
        value_type = args[1] if len(args) > 1 else Any
        return {
            "type": "object",
            "additionalProperties": python_type_to_json_schema(value_type, field_name),
        }

    # Handle basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        Decimal: {"type": "string", "format": "decimal", "description": "Decimal number as string"},
        Path: {"type": "string", "format": "path", "description": "File system path"},
        Any: {},
        object: {},
    }

    if python_type in type_mapping:
        return type_mapping[python_type].copy()

    # Handle Enum types
    if hasattr(python_type, "__mro__") and any(
        c.__name__ == "Enum" for c in python_type.__mro__ if hasattr(c, "__name__")
    ):
        return {
            "type": "string",
            "enum": [e.name for e in python_type],
        }

    # Default fallback
    return {"type": "string", "description": f"Type: {python_type}"}


def get_default_value(field_info: dataclasses.Field) -> Any:
    """Extract default value from a dataclass field."""
    if field_info.default is not MISSING:
        default = field_info.default
        if isinstance(default, Decimal):
            return str(default)
        if isinstance(default, Path):
            return str(default)
        return default
    if field_info.default_factory is not MISSING:
        try:
            default = field_info.default_factory()
            if isinstance(default, Decimal):
                return str(default)
            if isinstance(default, Path):
                return str(default)
            return default
        except Exception:
            return None
    return None


def dataclass_to_json_schema(cls: type, title: str | None = None) -> dict[str, Any]:
    """Convert a dataclass to JSON Schema v7."""
    if not dataclasses.is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": title or cls.__name__,
        "type": "object",
        "properties": {},
        "required": [],
    }

    if cls.__doc__:
        schema["description"] = cls.__doc__.strip()

    for field_info in fields(cls):
        field_schema = python_type_to_json_schema(field_info.type, field_info.name)

        # Add default if present
        default = get_default_value(field_info)
        if default is not None:
            field_schema["default"] = default

        # Check if field is required (no default)
        if field_info.default is MISSING and field_info.default_factory is MISSING:
            schema["required"].append(field_info.name)

        schema["properties"][field_info.name] = field_schema

    # Remove empty required list
    if not schema["required"]:
        del schema["required"]

    return schema


def add_field_descriptions(schema: dict[str, Any], descriptions: dict[str, str]) -> None:
    """Add descriptions to schema fields."""
    for field_name, description in descriptions.items():
        if field_name in schema.get("properties", {}):
            schema["properties"][field_name]["description"] = description


def generate_bot_config_schema() -> dict[str, Any]:
    """Generate schema for BotConfig."""
    from gpt_trader.orchestration.configuration.bot_config.bot_config import BotConfig

    schema = dataclass_to_json_schema(BotConfig, "BotConfig")

    # Add field descriptions
    descriptions = {
        "max_position_size": "Maximum position size in quote currency (e.g., USD)",
        "max_leverage": "Maximum leverage multiplier (1-10 typical)",
        "stop_loss_pct": "Stop loss percentage as decimal (0.02 = 2%)",
        "take_profit_pct": "Take profit percentage as decimal (0.04 = 4%)",
        "short_ma": "Short moving average period in candles",
        "long_ma": "Long moving average period in candles",
        "interval": "Trading interval in seconds",
        "symbols": "List of trading symbols (e.g., ['BTC-USD', 'ETH-USD'])",
        "derivatives_enabled": "Enable perpetual futures trading",
        "trailing_stop_pct": "Trailing stop percentage as decimal",
        "perps_position_fraction": "Fraction of balance to use for perps positions",
        "target_leverage": "Target leverage for derivatives positions",
        "enable_shorts": "Allow short positions",
        "reduce_only_mode": "Only allow position-reducing orders",
        "time_in_force": "Order time-in-force: GTC, IOC, FOK",
        "enable_order_preview": "Enable order preview before submission",
        "account_telemetry_interval": "Account snapshot interval in seconds (null to disable)",
        "log_level": "Logging level: DEBUG, INFO, WARNING, ERROR",
        "dry_run": "Simulate trades without execution",
        "mock_broker": "Use mock broker for testing",
        "profile": "Trading profile name or enum",
        "metadata": "Additional metadata key-value pairs",
    }
    add_field_descriptions(schema, descriptions)

    return schema


def generate_risk_config_schema() -> dict[str, Any]:
    """Generate schema for RiskConfig."""
    from gpt_trader.orchestration.configuration.risk.model import RiskConfig

    schema = dataclass_to_json_schema(RiskConfig, "RiskConfig")

    descriptions = {
        "max_leverage": "Maximum allowed leverage multiplier",
        "daily_loss_limit": "Maximum daily loss in quote currency",
        "max_exposure_pct": "Maximum portfolio exposure as decimal (0.8 = 80%)",
        "max_position_pct_per_symbol": "Maximum position size per symbol as decimal",
        "min_liquidation_buffer_pct": "Minimum buffer before liquidation as decimal",
        "leverage_max_per_symbol": "Per-symbol maximum leverage overrides",
        "max_notional_per_symbol": "Per-symbol maximum notional value",
        "slippage_guard_bps": "Slippage guard in basis points (100 = 1%)",
        "kill_switch_enabled": "Enable emergency kill switch",
        "reduce_only_mode": "Force reduce-only mode for all orders",
        "daytime_start_utc": "Daytime period start (HH:MM UTC)",
        "daytime_end_utc": "Daytime period end (HH:MM UTC)",
        "day_leverage_max_per_symbol": "Per-symbol leverage limits during daytime",
        "night_leverage_max_per_symbol": "Per-symbol leverage limits during nighttime",
        "day_mmr_per_symbol": "Maintenance margin requirement overrides (day)",
        "night_mmr_per_symbol": "Maintenance margin requirement overrides (night)",
        "enable_pre_trade_liq_projection": "Enable pre-trade liquidation projection",
    }
    add_field_descriptions(schema, descriptions)

    return schema


def generate_all_schemas() -> dict[str, dict[str, Any]]:
    """Generate all configuration schemas."""
    return {
        "bot_config": generate_bot_config_schema(),
        "risk_config": generate_risk_config_schema(),
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate JSON schemas for configuration dataclasses"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("var/agents/schemas"),
        help="Output directory for schema files (default: var/agents/schemas)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "combined"],
        default="json",
        help="Output format: json (separate files) or combined (single file)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output to stdout instead of files",
    )

    args = parser.parse_args()

    schemas = generate_all_schemas()

    if args.stdout:
        print(json.dumps(schemas, indent=2, default=str))
        return 0

    # Create output directory
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "combined":
        # Write single combined file
        combined_path = output_dir / "config_schemas.json"
        with open(combined_path, "w") as f:
            json.dump(schemas, f, indent=2, default=str)
        print(f"Combined schema written to: {combined_path}")
    else:
        # Write individual files
        for name, schema in schemas.items():
            schema_path = output_dir / f"{name}_schema.json"
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2, default=str)
            print(f"Schema written to: {schema_path}")

    # Also generate an index file
    index = {
        "schemas": list(schemas.keys()),
        "version": "1.0",
        "description": "Configuration schemas for GPT-Trader AI agent consumption",
        "files": {name: f"{name}_schema.json" for name in schemas.keys()},
    }
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index written to: {index_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
