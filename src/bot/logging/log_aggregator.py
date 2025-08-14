"""Log aggregation and analysis for GPT-Trader.

This module provides tools for:
- Log collection and parsing
- Pattern detection
- Performance analysis
- Error correlation
- Trade event tracking
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: datetime
    level: str
    logger: str
    message: str
    module: str = ""
    function: str = ""
    line: int = 0
    extra_fields: dict[str, Any] = field(default_factory=dict)
    exception: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, json_str: str) -> LogEntry:
        """Create LogEntry from JSON string.

        Args:
            json_str: JSON log string.

        Returns:
            LogEntry instance.
        """
        data = json.loads(json_str)

        # Parse timestamp
        timestamp = datetime.fromisoformat(data["timestamp"])

        # Extract exception if present
        exception = data.get("exception")

        # Extract extra fields
        extra_fields = {
            k: v
            for k, v in data.items()
            if k
            not in [
                "timestamp",
                "level",
                "logger",
                "message",
                "module",
                "function",
                "line",
                "exception",
            ]
        }

        return cls(
            timestamp=timestamp,
            level=data.get("level", "INFO"),
            logger=data.get("logger", ""),
            message=data.get("message", ""),
            module=data.get("module", ""),
            function=data.get("function", ""),
            line=data.get("line", 0),
            extra_fields=extra_fields,
            exception=exception,
        )

    @classmethod
    def from_text(cls, text_str: str) -> LogEntry:
        """Create LogEntry from text log string.

        Args:
            text_str: Text log string.

        Returns:
            LogEntry instance.
        """
        # Parse standard format: "timestamp | logger | level | message"
        pattern = r"^(\S+ \S+) \| (\S+) \| (\S+) \| (.*)$"
        match = re.match(pattern, text_str)

        if match:
            timestamp_str, logger, level, message = match.groups()
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

            return cls(timestamp=timestamp, level=level, logger=logger, message=message)

        # Fallback for non-standard format
        return cls(timestamp=datetime.now(), level="INFO", logger="unknown", message=text_str)


class LogAggregator:
    """Aggregates and analyzes log entries."""

    def __init__(self, max_entries: int = 10000) -> None:
        """Initialize log aggregator.

        Args:
            max_entries: Maximum entries to keep in memory.
        """
        self.max_entries = max_entries
        self.entries: deque[LogEntry] = deque(maxlen=max_entries)

        # Categorized storage
        self.errors: list[LogEntry] = []
        self.warnings: list[LogEntry] = []
        self.trades: list[LogEntry] = []
        self.metrics: list[LogEntry] = []

        # Statistics
        self.stats = {
            "total_entries": 0,
            "errors": 0,
            "warnings": 0,
            "trades": 0,
            "metrics": 0,
        }

    def add_entry(self, entry: LogEntry) -> None:
        """Add a log entry.

        Args:
            entry: Log entry to add.
        """
        self.entries.append(entry)
        self.stats["total_entries"] += 1

        # Categorize entry
        if entry.level == "ERROR" or entry.level == "CRITICAL":
            self.errors.append(entry)
            self.stats["errors"] += 1
        elif entry.level == "WARNING":
            self.warnings.append(entry)
            self.stats["warnings"] += 1

        # Check for trade events
        if "event_type" in entry.extra_fields:
            event_type = entry.extra_fields["event_type"]
            if event_type in ["order_placed", "order_filled", "position_opened", "position_closed"]:
                self.trades.append(entry)
                self.stats["trades"] += 1
            elif event_type in ["metric", "latency", "throughput"]:
                self.metrics.append(entry)
                self.stats["metrics"] += 1

    def parse_file(self, file_path: Path, format_type: str = "json") -> None:
        """Parse log file and add entries.

        Args:
            file_path: Path to log file.
            format_type: Log format ('json' or 'text').
        """
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    if format_type == "json":
                        entry = LogEntry.from_json(line)
                    else:
                        entry = LogEntry.from_text(line)

                    self.add_entry(entry)
                except Exception:
                    # Skip malformed entries
                    continue

    def get_entries_by_time(
        self, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[LogEntry]:
        """Get entries within time range.

        Args:
            start_time: Start of time range.
            end_time: End of time range.

        Returns:
            List of matching entries.
        """
        result = []

        for entry in self.entries:
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            result.append(entry)

        return result

    def get_entries_by_level(self, level: str) -> list[LogEntry]:
        """Get entries by log level.

        Args:
            level: Log level to filter by.

        Returns:
            List of matching entries.
        """
        return [e for e in self.entries if e.level == level]

    def get_error_summary(self) -> dict[str, Any]:
        """Get error summary statistics.

        Returns:
            Dictionary with error statistics.
        """
        if not self.errors:
            return {"total_errors": 0}

        # Group errors by type
        error_types = defaultdict(int)
        error_modules = defaultdict(int)

        for entry in self.errors:
            if entry.exception:
                error_type = entry.exception.get("type", "Unknown")
                error_types[error_type] += 1

            error_modules[entry.module] += 1

        # Find most recent errors
        recent_errors = sorted(self.errors, key=lambda x: x.timestamp, reverse=True)[:10]

        return {
            "total_errors": len(self.errors),
            "error_types": dict(error_types),
            "error_modules": dict(error_modules),
            "error_rate": len(self.errors) / len(self.entries) if self.entries else 0,
            "recent_errors": [
                {"timestamp": e.timestamp.isoformat(), "message": e.message, "module": e.module}
                for e in recent_errors
            ],
        }

    def get_trade_summary(self) -> dict[str, Any]:
        """Get trade summary statistics.

        Returns:
            Dictionary with trade statistics.
        """
        if not self.trades:
            return {"total_trades": 0}

        # Analyze trade events
        orders_placed = 0
        orders_filled = 0
        positions_opened = 0
        positions_closed = 0
        total_pnl = 0.0
        symbols_traded = set()

        for entry in self.trades:
            event_type = entry.extra_fields.get("event_type")

            if event_type == "order_placed":
                orders_placed += 1
            elif event_type == "order_filled":
                orders_filled += 1
            elif event_type == "position_opened":
                positions_opened += 1
            elif event_type == "position_closed":
                positions_closed += 1
                total_pnl += entry.extra_fields.get("pnl", 0)

            if "symbol" in entry.extra_fields:
                symbols_traded.add(entry.extra_fields["symbol"])

        return {
            "total_trades": len(self.trades),
            "orders_placed": orders_placed,
            "orders_filled": orders_filled,
            "positions_opened": positions_opened,
            "positions_closed": positions_closed,
            "fill_rate": orders_filled / orders_placed if orders_placed > 0 else 0,
            "total_pnl": total_pnl,
            "symbols_traded": list(symbols_traded),
            "unique_symbols": len(symbols_traded),
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary statistics.

        Returns:
            Dictionary with performance statistics.
        """
        if not self.metrics:
            return {"total_metrics": 0}

        # Analyze performance metrics
        latencies = []
        throughputs = []
        metrics_by_name = defaultdict(list)

        for entry in self.metrics:
            event_type = entry.extra_fields.get("event_type")

            if event_type == "latency":
                latencies.append(entry.extra_fields.get("latency_ms", 0))
            elif event_type == "throughput":
                throughputs.append(entry.extra_fields.get("throughput_per_second", 0))
            elif event_type == "metric":
                metric_name = entry.extra_fields.get("metric_name")
                metric_value = entry.extra_fields.get("value")
                if metric_name and metric_value is not None:
                    metrics_by_name[metric_name].append(metric_value)

        # Calculate statistics
        summary = {
            "total_metrics": len(self.metrics),
            "latency_stats": {},
            "throughput_stats": {},
            "custom_metrics": {},
        }

        if latencies:
            summary["latency_stats"] = {
                "count": len(latencies),
                "mean": sum(latencies) / len(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "p50": sorted(latencies)[len(latencies) // 2],
                "p95": (
                    sorted(latencies)[int(len(latencies) * 0.95)]
                    if len(latencies) > 20
                    else max(latencies)
                ),
            }

        if throughputs:
            summary["throughput_stats"] = {
                "count": len(throughputs),
                "mean": sum(throughputs) / len(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
            }

        for metric_name, values in metrics_by_name.items():
            if values:
                summary["custom_metrics"][metric_name] = {
                    "count": len(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        return summary

    def detect_patterns(self) -> dict[str, Any]:
        """Detect patterns in logs.

        Returns:
            Dictionary with detected patterns.
        """
        patterns = {
            "error_bursts": [],
            "performance_degradation": [],
            "repeated_errors": {},
            "correlated_events": [],
        }

        # Detect error bursts (>5 errors in 1 minute)
        if self.errors:
            error_times = [e.timestamp for e in self.errors]
            for i in range(len(error_times)):
                burst_end = error_times[i] + timedelta(minutes=1)
                burst_count = sum(1 for t in error_times[i:] if t <= burst_end)
                if burst_count >= 5:
                    patterns["error_bursts"].append(
                        {"timestamp": error_times[i].isoformat(), "count": burst_count}
                    )

        # Detect repeated errors
        error_messages = defaultdict(int)
        for error in self.errors:
            # Normalize message by removing numbers
            normalized = re.sub(r"\d+", "N", error.message)
            error_messages[normalized] += 1

        patterns["repeated_errors"] = {
            msg: count for msg, count in error_messages.items() if count >= 3
        }

        return patterns

    def to_dataframe(self) -> pd.DataFrame:
        """Convert entries to pandas DataFrame.

        Returns:
            DataFrame with log entries.
        """
        data = []

        for entry in self.entries:
            row = {
                "timestamp": entry.timestamp,
                "level": entry.level,
                "logger": entry.logger,
                "message": entry.message,
                "module": entry.module,
                "function": entry.function,
                "line": entry.line,
            }

            # Add extra fields with prefix
            for key, value in entry.extra_fields.items():
                row[f"extra_{key}"] = value

            # Add exception info
            if entry.exception:
                row["exception_type"] = entry.exception.get("type")
                row["exception_message"] = entry.exception.get("message")

            data.append(row)

        return pd.DataFrame(data)

    def generate_report(self) -> str:
        """Generate comprehensive log analysis report.

        Returns:
            Formatted report string.
        """
        report = []
        report.append("=" * 60)
        report.append("LOG ANALYSIS REPORT")
        report.append("=" * 60)

        # Overall statistics
        report.append("\n## Overall Statistics")
        report.append(f"Total Entries: {self.stats['total_entries']}")
        report.append(f"Errors: {self.stats['errors']}")
        report.append(f"Warnings: {self.stats['warnings']}")
        report.append(f"Trade Events: {self.stats['trades']}")
        report.append(f"Metrics: {self.stats['metrics']}")

        # Error summary
        error_summary = self.get_error_summary()
        if error_summary["total_errors"] > 0:
            report.append("\n## Error Summary")
            report.append(f"Total Errors: {error_summary['total_errors']}")
            report.append(f"Error Rate: {error_summary['error_rate']:.2%}")

            if error_summary["error_types"]:
                report.append("\nError Types:")
                for error_type, count in error_summary["error_types"].items():
                    report.append(f"  - {error_type}: {count}")

        # Trade summary
        trade_summary = self.get_trade_summary()
        if trade_summary["total_trades"] > 0:
            report.append("\n## Trade Summary")
            report.append(f"Total Trade Events: {trade_summary['total_trades']}")
            report.append(f"Orders Placed: {trade_summary['orders_placed']}")
            report.append(f"Orders Filled: {trade_summary['orders_filled']}")
            report.append(f"Fill Rate: {trade_summary['fill_rate']:.2%}")
            report.append(f"Total P&L: ${trade_summary['total_pnl']:.2f}")
            report.append(f"Unique Symbols: {trade_summary['unique_symbols']}")

        # Performance summary
        perf_summary = self.get_performance_summary()
        if perf_summary["total_metrics"] > 0:
            report.append("\n## Performance Summary")
            report.append(f"Total Metrics: {perf_summary['total_metrics']}")

            if perf_summary["latency_stats"]:
                report.append("\nLatency Statistics:")
                stats = perf_summary["latency_stats"]
                report.append(f"  - Mean: {stats['mean']:.2f}ms")
                report.append(f"  - P50: {stats['p50']:.2f}ms")
                report.append(f"  - P95: {stats['p95']:.2f}ms")
                report.append(f"  - Max: {stats['max']:.2f}ms")

        # Pattern detection
        patterns = self.detect_patterns()
        if patterns["error_bursts"] or patterns["repeated_errors"]:
            report.append("\n## Detected Patterns")

            if patterns["error_bursts"]:
                report.append(f"\nError Bursts: {len(patterns['error_bursts'])}")

            if patterns["repeated_errors"]:
                report.append("\nRepeated Errors:")
                for msg, count in list(patterns["repeated_errors"].items())[:5]:
                    report.append(f"  - ({count}x) {msg[:80]}...")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
