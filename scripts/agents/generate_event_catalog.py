#!/usr/bin/env python3
"""Generate a catalog of structured log events for AI agent consumption.

This script scans the codebase to discover logging patterns and generates
a machine-readable event catalog documenting:
- Event operation types and their frequency
- Common log fields and their meanings
- Component/module categorization
- Log level distribution

Usage:
    python scripts/agents/generate_event_catalog.py [--output-dir DIR]

Output:
    var/agents/logging/
    - event_catalog.json (event types with fields)
    - log_schema.json (JSON log format specification)
    - index.json (discovery file)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def scan_logging_calls(source_dir: Path) -> dict[str, Any]:
    """Scan source files for logging calls and extract patterns."""
    operations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    components: dict[str, list[str]] = defaultdict(list)
    fields_seen: dict[str, int] = defaultdict(int)
    log_levels: dict[str, int] = defaultdict(int)

    for py_file in source_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
        except Exception:
            continue

        # Extract operation= patterns
        operation_pattern = r'operation\s*=\s*["\']([^"\']+)["\']'
        for match in re.finditer(operation_pattern, content):
            operation = match.group(1)

            # Find the surrounding context
            start = max(0, match.start() - 200)
            end = min(len(content), match.end() + 100)
            context = content[start:end]

            # Extract component if present
            component_match = re.search(r'component\s*=\s*["\']([^"\']+)["\']', context)
            component = component_match.group(1) if component_match else None

            # Extract status if present
            status_match = re.search(r'status\s*=\s*["\']([^"\']+)["\']', context)
            status = status_match.group(1) if status_match else None

            # Determine log level from context
            level = "INFO"
            for lvl in ["debug", "info", "warning", "error", "critical"]:
                if f".{lvl}(" in context.lower():
                    level = lvl.upper()
                    break

            # Find other keyword arguments
            kwarg_pattern = r"(\w+)\s*="
            kwargs = set(re.findall(kwarg_pattern, context))
            # Filter out common non-field kwargs
            kwargs -= {"operation", "component", "status", "extra", "exc_info"}

            operations[operation].append(
                {
                    "file": str(py_file.relative_to(source_dir)),
                    "component": component,
                    "status": status,
                    "level": level,
                    "fields": list(kwargs)[:10],  # Limit fields
                }
            )

            if component:
                components[component].append(operation)

            for field in kwargs:
                fields_seen[field] += 1

            log_levels[level] += 1

    return {
        "operations": dict(operations),
        "components": {k: list(set(v)) for k, v in components.items()},
        "fields": dict(fields_seen),
        "levels": dict(log_levels),
    }


def categorize_operations(operations: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    """Categorize operations by domain."""
    categories: dict[str, list[str]] = {
        "trading": [],
        "data": [],
        "monitoring": [],
        "security": [],
        "configuration": [],
        "system": [],
        "strategy": [],
        "risk": [],
        "other": [],
    }

    category_keywords = {
        "trading": ["order", "trade", "position", "balance", "fill", "cancel", "submit"],
        "data": ["candle", "cache", "fetch", "load", "validate", "coverage", "quote"],
        "monitoring": ["telemetry", "health", "monitor", "metrics", "stream"],
        "security": ["vault", "secret", "suspicious", "ip", "allowlist", "encrypt"],
        "configuration": ["config", "setting", "runtime", "profile"],
        "system": ["server", "shutdown", "startup", "schedule", "coroutine"],
        "strategy": ["strategy", "signal", "indicator", "prepare", "execute"],
        "risk": ["risk", "guard", "liquidation", "margin", "limit", "alert"],
    }

    for op_name in operations.keys():
        categorized = False
        op_lower = op_name.lower()
        for category, keywords in category_keywords.items():
            if any(kw in op_lower for kw in keywords):
                categories[category].append(op_name)
                categorized = True
                break
        if not categorized:
            categories["other"].append(op_name)

    return {k: sorted(v) for k, v in categories.items() if v}


def generate_log_schema() -> dict[str, Any]:
    """Generate JSON schema for the structured log format."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "GPT-Trader Structured Log Entry",
        "description": "Schema for JSON log entries produced by StructuredJSONFormatter",
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "string",
                "format": "date-time",
                "description": "ISO 8601 timestamp in UTC",
            },
            "level": {
                "type": "string",
                "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "description": "Log severity level",
            },
            "logger": {
                "type": "string",
                "description": "Logger name (usually module path)",
            },
            "message": {
                "type": "string",
                "description": "Human-readable log message",
            },
            "module": {
                "type": "string",
                "description": "Python module name",
            },
            "function": {
                "type": "string",
                "description": "Function name where log was called",
            },
            "line": {
                "type": "integer",
                "description": "Line number in source file",
            },
            "thread": {
                "type": "integer",
                "description": "Thread ID",
            },
            "process": {
                "type": "integer",
                "description": "Process ID",
            },
            "correlation_id": {
                "type": "string",
                "description": "Request correlation ID for tracing",
            },
            "operation": {
                "type": "string",
                "description": "Operation type identifier for event categorization",
            },
            "component": {
                "type": "string",
                "description": "System component name",
            },
            "status": {
                "type": "string",
                "description": "Operation status (success, error, pending, etc.)",
            },
            "exception": {
                "type": "object",
                "description": "Exception information if error",
                "properties": {
                    "type": {"type": "string"},
                    "message": {"type": "string"},
                    "module": {"type": "string"},
                },
            },
        },
        "required": ["timestamp", "level", "logger", "message"],
        "additionalProperties": True,
    }


def generate_event_catalog(scan_results: dict[str, Any]) -> dict[str, Any]:
    """Generate the event catalog from scan results."""
    operations = scan_results["operations"]
    categories = categorize_operations(operations)

    events = {}
    for op_name, occurrences in operations.items():
        # Aggregate info from all occurrences
        components = set()
        statuses = set()
        levels = set()
        all_fields: set[str] = set()
        files = set()

        for occ in occurrences:
            if occ.get("component"):
                components.add(occ["component"])
            if occ.get("status"):
                statuses.add(occ["status"])
            levels.add(occ["level"])
            all_fields.update(occ.get("fields", []))
            files.add(occ["file"])

        events[op_name] = {
            "occurrence_count": len(occurrences),
            "components": sorted(components) if components else None,
            "statuses": sorted(statuses) if statuses else None,
            "levels": sorted(levels),
            "common_fields": sorted(all_fields)[:15],
            "source_files": sorted(files)[:5],  # Limit to 5 example files
        }

    return {
        "version": "1.0",
        "description": "Catalog of structured log events in GPT-Trader",
        "total_event_types": len(events),
        "categories": categories,
        "events": events,
        "field_frequency": dict(sorted(scan_results["fields"].items(), key=lambda x: -x[1])[:30]),
        "level_distribution": scan_results["levels"],
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate event catalog from logging patterns")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("var/agents/logging"),
        help="Output directory for catalog files",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("src/gpt_trader"),
        help="Source directory to scan",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output to stdout instead of files",
    )

    args = parser.parse_args()

    print(f"Scanning {args.source_dir} for logging patterns...")
    scan_results = scan_logging_calls(args.source_dir)

    event_catalog = generate_event_catalog(scan_results)
    log_schema = generate_log_schema()

    if args.stdout:
        output = {
            "event_catalog": event_catalog,
            "log_schema": log_schema,
        }
        print(json.dumps(output, indent=2))
        return 0

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write event catalog
    catalog_path = output_dir / "event_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(event_catalog, f, indent=2)
    print(f"Event catalog written to: {catalog_path}")

    # Write log schema
    schema_path = output_dir / "log_schema.json"
    with open(schema_path, "w") as f:
        json.dump(log_schema, f, indent=2)
    print(f"Log schema written to: {schema_path}")

    # Write index
    index = {
        "version": "1.0",
        "description": "Logging event catalog for AI agent consumption",
        "files": {
            "event_catalog": "event_catalog.json",
            "log_schema": "log_schema.json",
        },
        "summary": {
            "total_events": event_catalog["total_event_types"],
            "categories": list(event_catalog["categories"].keys()),
            "top_operations": list(event_catalog["events"].keys())[:10],
        },
        "usage": {
            "parsing": "Use log_schema.json to validate/parse JSON log entries",
            "filtering": "Use event_catalog.json to filter logs by operation type",
            "correlation": "Use correlation_id field to trace requests across logs",
        },
    }
    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index written to: {index_path}")

    print(f"\nFound {event_catalog['total_event_types']} unique event types")

    return 0


if __name__ == "__main__":
    sys.exit(main())
