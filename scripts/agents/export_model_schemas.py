#!/usr/bin/env python3
"""Export JSON schemas and documentation for core domain models.

This script generates machine-readable schemas for AI agent consumption,
covering:
- Core trading interfaces (Order, Position, Quote, etc.)
- Enumeration types (OrderSide, OrderType, OrderStatus, etc.)
- Error hierarchy (TradingError and subclasses)
- CLI error codes

Usage:
    python scripts/agents/export_model_schemas.py [--output-dir DIR]

Output:
    Generates schema files in var/agents/models/ including:
    - interfaces_schema.json (Order, Position, Balance, etc.)
    - enums_schema.json (OrderSide, OrderType, etc.)
    - errors_schema.json (error codes and recovery info)
    - index.json (discovery file)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import MISSING, fields
from datetime import datetime
from decimal import Decimal
from enum import Enum
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

    # Handle Union/Optional (X | None)
    if str(origin) == "types.UnionType" or str(origin) == "typing.Union":
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
        value_type = args[1] if len(args) > 1 else Any
        return {
            "type": "object",
            "additionalProperties": python_type_to_json_schema(value_type, field_name),
        }

    # Handle Enum types
    if isinstance(python_type, type) and issubclass(python_type, Enum):
        return {
            "type": "string",
            "enum": [e.value for e in python_type],
        }

    # Handle basic types
    type_mapping: dict[type, dict[str, Any]] = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        Decimal: {"type": "string", "format": "decimal"},
        datetime: {"type": "string", "format": "date-time"},
        Path: {"type": "string", "format": "path"},
        Any: {},
        object: {},
    }

    if python_type in type_mapping:
        return type_mapping[python_type].copy()

    # Default fallback
    type_name = getattr(python_type, "__name__", str(python_type))
    return {"type": "string", "description": f"Type: {type_name}"}


def get_default_value(field_info: dataclasses.Field) -> Any:
    """Extract default value from a dataclass field."""
    if field_info.default is not MISSING:
        default = field_info.default
        if isinstance(default, Decimal):
            return str(default)
        if isinstance(default, Enum):
            return default.value
        if isinstance(default, datetime):
            return default.isoformat()
        return default
    if field_info.default_factory is not MISSING:
        try:
            default = field_info.default_factory()
            if isinstance(default, Decimal):
                return str(default)
            return default
        except Exception:
            return None
    return None


def dataclass_to_schema(cls: type, descriptions: dict[str, str] | None = None) -> dict[str, Any]:
    """Convert a dataclass to JSON Schema."""
    schema: dict[str, Any] = {
        "title": cls.__name__,
        "type": "object",
        "properties": {},
        "required": [],
    }

    if cls.__doc__:
        schema["description"] = cls.__doc__.strip()

    for field_info in fields(cls):
        field_schema = python_type_to_json_schema(field_info.type, field_info.name)

        default = get_default_value(field_info)
        if default is not None:
            field_schema["default"] = default

        if descriptions and field_info.name in descriptions:
            field_schema["description"] = descriptions[field_info.name]

        if field_info.default is MISSING and field_info.default_factory is MISSING:
            schema["required"].append(field_info.name)

        schema["properties"][field_info.name] = field_schema

    if not schema["required"]:
        del schema["required"]

    return schema


def enum_to_schema(enum_cls: type) -> dict[str, Any]:
    """Convert an Enum to JSON Schema with documentation."""
    values = []
    descriptions = {}

    for member in enum_cls:
        values.append(member.value)
        # Check for docstring on enum member
        if hasattr(member, "__doc__") and member.__doc__:
            descriptions[member.value] = member.__doc__

    schema: dict[str, Any] = {
        "title": enum_cls.__name__,
        "type": "string",
        "enum": values,
    }

    if enum_cls.__doc__:
        schema["description"] = enum_cls.__doc__.strip()

    if descriptions:
        schema["x-enum-descriptions"] = descriptions

    return schema


def generate_interface_schemas() -> dict[str, Any]:
    """Generate schemas for core trading interfaces."""
    from gpt_trader.core import (
        Balance,
        Candle,
        Order,
        Position,
        Product,
        Quote,
    )

    schemas = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GPT-Trader Core Interfaces",
        "description": "Domain model schemas for trading operations",
        "definitions": {},
    }

    # Define field descriptions for each model
    candle_descriptions = {
        "ts": "Candle timestamp (start of period)",
        "open": "Opening price",
        "high": "Highest price in period",
        "low": "Lowest price in period",
        "close": "Closing price",
        "volume": "Trading volume in base asset",
    }

    quote_descriptions = {
        "symbol": "Trading symbol (e.g., BTC-USD)",
        "bid": "Best bid price",
        "ask": "Best ask price",
        "last": "Last traded price",
        "ts": "Quote timestamp",
    }

    product_descriptions = {
        "symbol": "Trading symbol (e.g., BTC-USD)",
        "base_asset": "Base asset (e.g., BTC)",
        "quote_asset": "Quote asset (e.g., USD)",
        "market_type": "Market type: SPOT, PERPETUAL, FUTURE",
        "min_size": "Minimum order size in base asset",
        "step_size": "Order size increment",
        "min_notional": "Minimum order value in quote asset",
        "price_increment": "Minimum price increment (tick size)",
        "leverage_max": "Maximum leverage (null for spot)",
        "expiry": "Contract expiry date (futures only)",
        "contract_size": "Contract multiplier",
        "funding_rate": "Current funding rate (perpetuals)",
        "next_funding_time": "Next funding timestamp",
    }

    order_descriptions = {
        "id": "Unique order identifier from broker",
        "symbol": "Trading symbol",
        "side": "Order side: BUY or SELL",
        "type": "Order type: MARKET, LIMIT, STOP, STOP_LIMIT",
        "quantity": "Order quantity in base asset",
        "status": "Order status: PENDING, SUBMITTED, FILLED, etc.",
        "filled_quantity": "Quantity already filled",
        "price": "Limit price (null for market orders)",
        "stop_price": "Stop trigger price",
        "tif": "Time in force: GTC, IOC, FOK",
        "client_id": "Client-provided order identifier",
        "avg_fill_price": "Average fill price",
        "submitted_at": "Order submission timestamp",
        "updated_at": "Last update timestamp",
        "created_at": "Creation timestamp",
    }

    position_descriptions = {
        "symbol": "Trading symbol",
        "quantity": "Position size (positive for long, negative for short)",
        "entry_price": "Average entry price",
        "mark_price": "Current mark price",
        "unrealized_pnl": "Unrealized profit/loss",
        "realized_pnl": "Realized profit/loss",
        "side": "Position side: long or short",
        "leverage": "Current leverage",
    }

    balance_descriptions = {
        "asset": "Asset symbol (e.g., USD, BTC)",
        "total": "Total balance",
        "available": "Available balance (total - hold)",
        "hold": "Amount on hold for open orders",
    }

    models = [
        (Candle, candle_descriptions),
        (Quote, quote_descriptions),
        (Product, product_descriptions),
        (Order, order_descriptions),
        (Position, position_descriptions),
        (Balance, balance_descriptions),
    ]

    for model_cls, descriptions in models:
        schemas["definitions"][model_cls.__name__] = dataclass_to_schema(model_cls, descriptions)

    return schemas


def generate_enum_schemas() -> dict[str, Any]:
    """Generate schemas for all enumeration types."""
    from gpt_trader.core import (
        MarketType,
        OrderSide,
        OrderStatus,
        OrderType,
        TimeInForce,
    )

    schemas = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GPT-Trader Enumerations",
        "description": "Enumeration types for trading operations",
        "definitions": {},
    }

    enums = [OrderSide, OrderType, TimeInForce, MarketType, OrderStatus]

    # Add descriptions for enum values
    enum_docs = {
        "OrderSide": {
            "BUY": "Buy/long order",
            "SELL": "Sell/short order",
        },
        "OrderType": {
            "MARKET": "Execute immediately at current market price",
            "LIMIT": "Execute at specified price or better",
            "STOP": "Market order triggered when stop price is reached",
            "STOP_LIMIT": "Limit order triggered when stop price is reached",
        },
        "TimeInForce": {
            "GTC": "Good Till Cancelled - remains active until filled or cancelled",
            "IOC": "Immediate Or Cancel - fill immediately or cancel unfilled portion",
            "FOK": "Fill Or Kill - fill entire order immediately or cancel",
        },
        "MarketType": {
            "SPOT": "Spot market (immediate delivery)",
            "PERPETUAL": "Perpetual futures contract (no expiry)",
            "FUTURE": "Futures contract with expiry",
            "FUTURES": "Alias for FUTURE",
        },
        "OrderStatus": {
            "PENDING": "Order created but not yet submitted",
            "SUBMITTED": "Order submitted to broker",
            "PARTIALLY_FILLED": "Order partially executed",
            "FILLED": "Order fully executed",
            "CANCELLED": "Order cancelled by user or system",
            "REJECTED": "Order rejected by broker",
            "EXPIRED": "Order expired (e.g., IOC unfilled portion)",
            "FAILED": "Order failed due to error",
        },
    }

    for enum_cls in enums:
        schema = enum_to_schema(enum_cls)
        if enum_cls.__name__ in enum_docs:
            schema["x-enum-descriptions"] = enum_docs[enum_cls.__name__]
        schemas["definitions"][enum_cls.__name__] = schema

    return schemas


def generate_error_schemas() -> dict[str, Any]:
    """Generate schemas for error types with recovery information."""
    from gpt_trader.errors import (
        AggregateError,
        BacktestError,
        ConfigurationError,
        DataError,
        ExecutionError,
        InsufficientFundsError,
        NetworkError,
        OptimizationError,
        RiskLimitExceeded,
        SliceIsolationError,
        StrategyError,
        TimeoutError,
        TradingError,
        ValidationError,
    )

    schemas = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GPT-Trader Error Types",
        "description": "Error hierarchy for trading operations",
        "error_codes": {},
        "recovery_guide": {},
    }

    # Base error schema
    schemas["base_error_schema"] = {
        "type": "object",
        "properties": {
            "error_code": {"type": "string", "description": "Machine-readable error code"},
            "message": {"type": "string", "description": "Human-readable error message"},
            "context": {"type": "object", "description": "Additional error context"},
            "recoverable": {"type": "boolean", "description": "Whether error is recoverable"},
            "timestamp": {"type": "string", "format": "date-time"},
            "traceback": {"type": "string", "description": "Stack trace if available"},
        },
        "required": ["error_code", "message"],
    }

    error_classes = [
        TradingError,
        DataError,
        ConfigurationError,
        ValidationError,
        ExecutionError,
        NetworkError,
        InsufficientFundsError,
        StrategyError,
        BacktestError,
        OptimizationError,
        RiskLimitExceeded,
        TimeoutError,
        SliceIsolationError,
        AggregateError,
    ]

    # Error codes with recovery information
    error_info = {
        "TradingError": {
            "code": "TRADING_ERROR",
            "description": "Base class for all trading errors",
            "recoverable": True,
            "recovery_actions": ["Check error context", "Retry operation if transient"],
        },
        "DataError": {
            "code": "DATA_ERROR",
            "description": "Issues with market data",
            "recoverable": True,
            "recovery_actions": [
                "Verify symbol is valid",
                "Check data source availability",
                "Retry with backoff",
            ],
        },
        "ConfigurationError": {
            "code": "CONFIG_ERROR",
            "description": "Configuration issues",
            "recoverable": False,
            "recovery_actions": [
                "Check config file syntax",
                "Verify required fields",
                "Compare with schema",
            ],
        },
        "ValidationError": {
            "code": "VALIDATION_ERROR",
            "description": "Input validation failures",
            "recoverable": False,
            "recovery_actions": [
                "Check field value against schema",
                "Verify data types",
                "Check constraints",
            ],
        },
        "ExecutionError": {
            "code": "EXECUTION_ERROR",
            "description": "Trade execution failures",
            "recoverable": True,
            "recovery_actions": [
                "Check order status",
                "Verify account balance",
                "Retry if transient",
            ],
        },
        "NetworkError": {
            "code": "NETWORK_ERROR",
            "description": "Network/API failures",
            "recoverable": True,
            "recovery_actions": [
                "Retry with exponential backoff",
                "Check API status",
                "Verify credentials",
            ],
        },
        "InsufficientFundsError": {
            "code": "INSUFFICIENT_FUNDS",
            "description": "Not enough balance for operation",
            "recoverable": False,
            "recovery_actions": [
                "Reduce order size",
                "Deposit funds",
                "Close existing positions",
            ],
        },
        "StrategyError": {
            "code": "STRATEGY_ERROR",
            "description": "Strategy execution failures",
            "recoverable": True,
            "recovery_actions": [
                "Check strategy parameters",
                "Verify data inputs",
                "Review strategy logic",
            ],
        },
        "BacktestError": {
            "code": "BACKTEST_ERROR",
            "description": "Backtesting failures",
            "recoverable": True,
            "recovery_actions": [
                "Check date range",
                "Verify data availability",
                "Reduce backtest scope",
            ],
        },
        "OptimizationError": {
            "code": "OPTIMIZATION_ERROR",
            "description": "Optimization failures",
            "recoverable": True,
            "recovery_actions": [
                "Check parameter bounds",
                "Reduce trial count",
                "Verify objective function",
            ],
        },
        "RiskLimitExceeded": {
            "code": "RISK_LIMIT_EXCEEDED",
            "description": "Risk limits breached",
            "recoverable": False,
            "recovery_actions": [
                "Reduce position size",
                "Close positions to reduce exposure",
                "Adjust risk parameters",
            ],
        },
        "TimeoutError": {
            "code": "TIMEOUT_ERROR",
            "description": "Operation timed out",
            "recoverable": True,
            "recovery_actions": [
                "Increase timeout",
                "Retry operation",
                "Check system resources",
            ],
        },
        "SliceIsolationError": {
            "code": "SLICE_ISOLATION_ERROR",
            "description": "Feature slice isolation violated",
            "recoverable": False,
            "recovery_actions": [
                "Review import dependencies",
                "Check slice boundaries",
                "Refactor to use proper interfaces",
            ],
        },
        "AggregateError": {
            "code": "AGGREGATE_ERROR",
            "description": "Multiple errors occurred",
            "recoverable": True,
            "recovery_actions": [
                "Process each contained error",
                "Fix errors in order of severity",
            ],
        },
    }

    for error_cls in error_classes:
        name = error_cls.__name__
        if name in error_info:
            schemas["error_codes"][name] = {
                "code": error_info[name]["code"],
                "description": error_info[name]["description"],
                "recoverable_default": error_info[name]["recoverable"],
                "class_doc": error_cls.__doc__ or "",
            }
            schemas["recovery_guide"][error_info[name]["code"]] = error_info[name][
                "recovery_actions"
            ]

    # Add CLI error codes
    from gpt_trader.cli.response import CliErrorCode

    schemas["cli_error_codes"] = {code.name: code.value for code in CliErrorCode}

    return schemas


def generate_all_schemas() -> dict[str, dict[str, Any]]:
    """Generate all model schemas."""
    return {
        "interfaces": generate_interface_schemas(),
        "enums": generate_enum_schemas(),
        "errors": generate_error_schemas(),
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Export JSON schemas for domain models")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("var/agents/models"),
        help="Output directory for schema files",
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

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, schema in schemas.items():
        schema_path = output_dir / f"{name}_schema.json"
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2, default=str)
        print(f"Schema written to: {schema_path}")

    # Generate index
    index = {
        "schemas": list(schemas.keys()),
        "version": "1.0",
        "description": "Domain model schemas for GPT-Trader AI agent consumption",
        "files": {name: f"{name}_schema.json" for name in schemas.keys()},
        "usage": {
            "interfaces": "Core trading data structures (Order, Position, etc.)",
            "enums": "Enumeration types with valid values",
            "errors": "Error codes with recovery guidance",
        },
    }
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index written to: {index_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
