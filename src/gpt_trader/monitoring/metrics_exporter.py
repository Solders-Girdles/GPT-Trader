"""
Prometheus metrics exporter.

Converts internal metrics summary to Prometheus text exposition format.
See: https://prometheus.io/docs/instrumenting/exposition_formats/
"""

from __future__ import annotations

import re
from typing import Any

# Regex to parse internal label format: name{key=val,key2=val2}
_METRIC_PATTERN = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\}$")
_LABEL_PATTERN = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^,}]+)")


def _parse_metric_key(key: str) -> tuple[str, dict[str, str]]:
    """Parse internal metric key into name and labels.

    Args:
        key: Metric key like "gpt_trader_order_total{result=success,side=buy}"
             or just "gpt_trader_equity_dollars"

    Returns:
        Tuple of (metric_name, labels_dict)
    """
    match = _METRIC_PATTERN.match(key)
    if not match:
        # No labels
        return key, {}

    name = match.group(1)
    labels_str = match.group(2)

    labels = {}
    for label_match in _LABEL_PATTERN.finditer(labels_str):
        labels[label_match.group(1)] = label_match.group(2)

    return name, labels


def _format_labels(labels: dict[str, str]) -> str:
    """Format labels for Prometheus output.

    Args:
        labels: Label dict (e.g., {"result": "success", "side": "buy"})

    Returns:
        Prometheus label string like '{result="success",side="buy"}'
        or empty string if no labels.
    """
    if not labels:
        return ""

    # Sort for consistent output, quote values
    sorted_pairs = sorted(labels.items())
    label_parts = [f'{k}="{v}"' for k, v in sorted_pairs]
    return "{" + ",".join(label_parts) + "}"


def _escape_help(text: str) -> str:
    """Escape text for HELP lines."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def format_prometheus(metrics_summary: dict[str, Any]) -> str:
    """Format metrics summary as Prometheus text exposition format.

    Converts the internal metrics format to standard Prometheus format:
    - Counters: metric_name{labels} value
    - Gauges: metric_name{labels} value
    - Histograms: metric_name_bucket{le="bound",labels}, _sum, _count

    Args:
        metrics_summary: Dict from get_metrics_summary() with keys:
            - counters: dict[str, int]
            - gauges: dict[str, float]
            - histograms: dict[str, HistogramData.to_dict()]

    Returns:
        Prometheus text format string.
    """
    lines: list[str] = []

    # Track which metric names we've seen to emit TYPE only once
    seen_names: set[str] = set()

    # Counters
    counters = metrics_summary.get("counters", {})
    for key, value in sorted(counters.items()):
        name, labels = _parse_metric_key(key)
        if name not in seen_names:
            lines.append(f"# TYPE {name} counter")
            seen_names.add(name)
        lines.append(f"{name}{_format_labels(labels)} {value}")

    # Gauges
    gauges = metrics_summary.get("gauges", {})
    for key, value in sorted(gauges.items()):
        name, labels = _parse_metric_key(key)
        if name not in seen_names:
            lines.append(f"# TYPE {name} gauge")
            seen_names.add(name)
        lines.append(f"{name}{_format_labels(labels)} {value}")

    # Histograms
    histograms = metrics_summary.get("histograms", {})
    for key, data in sorted(histograms.items()):
        name, labels = _parse_metric_key(key)

        if name not in seen_names:
            lines.append(f"# TYPE {name} histogram")
            seen_names.add(name)

        buckets = data.get("buckets", {})
        count = data.get("count", 0)
        total = data.get("sum", 0.0)

        # Sort buckets by bound value and emit cumulative counts
        sorted_buckets = sorted(buckets.items(), key=lambda x: float(x[0]))
        cumulative = 0
        for bound, bucket_count in sorted_buckets:
            cumulative += bucket_count
            # Add le label for bucket
            bucket_labels = {**labels, "le": bound}
            lines.append(f"{name}_bucket{_format_labels(bucket_labels)} {cumulative}")

        # Add +Inf bucket (equals total count)
        inf_labels = {**labels, "le": "+Inf"}
        lines.append(f"{name}_bucket{_format_labels(inf_labels)} {count}")

        # Add _sum and _count
        lines.append(f"{name}_sum{_format_labels(labels)} {total}")
        lines.append(f"{name}_count{_format_labels(labels)} {count}")

    # Add trailing newline
    if lines:
        lines.append("")

    return "\n".join(lines)


__all__ = ["format_prometheus"]
