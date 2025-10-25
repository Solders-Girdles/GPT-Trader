"""Shared logging utilities for standardization across the codebase.

This module provides common logging patterns, formatters, and utilities
that were previously scattered across multiple modules, creating inconsistency.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="shared_logging_utils")


class StructuredLogger:
    """Enhanced logger with structured output and correlation support."""

    def __init__(self, component: str, logger_name: str | None = None):
        """Initialize structured logger."""
        self.component = component
        self.base_logger = logger_name or logger
        self._log_buffer: list[dict[str, Any]] = []

    def _log_structured(
        self,
        level: str,
        message: str,
        *,
        operation: str | None = None,
        symbol: str | None = None,
        details: dict[str, Any] | None = None,
        correlation_id: str | None = None,
        exception: Exception | None = None,
    ) -> None:
        """Internal structured logging method."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "component": self.component,
            "message": message,
            "operation": operation,
            "symbol": symbol,
            "details": details or {},
            "correlation_id": correlation_id,
        }

        if exception:
            log_entry["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": str(exception.__traceback__),
            }

        # Add to buffer for batch processing
        self._log_buffer.append(log_entry)

        # Determine target logger and level
        target_logger = self.base_logger if hasattr(self.base_logger, level) else logger
        log_method = getattr(target_logger, level.lower(), target_logger.info)

        try:
            log_method(message, extra=log_entry)
        except Exception:
            # Fallback to basic logging if structured logging fails
            logger.error(f"Logging failed for {self.component}: {message}")

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log_structured("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_structured("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log_structured("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_structured("DEBUG", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log_structured("CRITICAL", message, **kwargs)

    def flush_buffered_logs(self, target_logger: Any = None) -> None:
        """Flush all buffered logs to target logger."""
        if not self._log_buffer:
            return

        target = target_logger or self.base_logger

        for log_entry in self._log_buffer:
            try:
                # Try to use structured logging if available
                if hasattr(target, "info") and callable(target.info):
                    target.info(json.dumps(log_entry), extra=log_entry)
                else:
                    # Fallback to basic logging
                    level_map = {
                        "INFO": target.info,
                        "WARNING": target.warning,
                        "ERROR": target.error,
                        "CRITICAL": target.critical,
                    }
                    log_method = level_map.get(log_entry["level"], target.info)
                    log_method(log_entry["message"], extra=log_entry)
            except Exception as e:
                logger.error(f"Failed to log structured entry: {e}")

        self._log_buffer.clear()

    def get_buffer_size(self) -> int:
        """Get current buffer size."""
        return len(self._log_buffer)

    def set_buffer_size(self, max_size: int) -> None:
        """Set maximum buffer size."""
        if self.get_buffer_size() > max_size:
            # Drop oldest entries to maintain buffer size
            self._log_buffer = self._log_buffer[-max_size:]
            logger.warning(
                f"Log buffer truncated to {max_size} entries",
                component=self.component,
            )


class AuditLogger:
    """Logger for audit trails and compliance tracking."""

    def __init__(self, component: str):
        """Initialize audit logger."""
        self.component = component
        self._audit_trail: list[dict[str, Any]] = []

    def log_action(
        self,
        action: str,
        actor: str | None = None,
        *,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        result: str | None = None,
    ) -> None:
        """Log an auditable action."""
        audit_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "component": self.component,
            "action": action,
            "actor": actor,
            "resource_id": resource_id,
            "details": details or {},
            "result": result,
        }

        self._audit_trail.append(audit_entry)

        # Write to audit trail
        try:
            # This would write to persistent audit storage
            # For now, just log to file
            logger.debug(
                f"Audit action logged: {action}",
                operation="audit_action",
                action=action,
                actor=actor,
                audit_entry=audit_entry,
            )
        except Exception as e:
            logger.error(f"Failed to log audit action: {e}")

    def get_audit_trail(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit trail entries."""
        return self._audit_trail[-limit:]

    def export_audit_trail(self, file_path: str) -> None:
        """Export audit trail to file."""
        try:
            with open(file_path, "w") as f:
                json.dump(self._audit_trail, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to export audit trail to {file_path}: {e}")


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""

    def __init__(self, component: str):
        """Initialize performance logger."""
        self.component = component
        self._metrics: dict[str, float] = {}

    def start_timer(self, operation_id: str) -> None:
        """Start timing an operation."""
        self._metrics[f"{operation_id}_start"] = datetime.now(UTC).timestamp()

    def end_timer(self, operation_id: str) -> None:
        """End timing an operation."""
        if f"{operation_id}_start" in self._metrics:
            start_time = float(self._metrics[f"{operation_id}_start"])
            duration = datetime.now(UTC).timestamp() - start_time
            self._metrics[f"{operation_id}_duration"] = max(duration, 0.0)
            self._metrics[f"{operation_id}_last"] = datetime.now(UTC).isoformat()

    def log_metric(self, metric_name: str, value: float, unit: str = "ms") -> None:
        """Log a performance metric."""
        self._metrics[metric_name] = {
            "value": value,
            "unit": unit,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all performance metrics."""
        duration_keys = [
            key
            for key, value in self._metrics.items()
            if key.endswith("_duration") and isinstance(value, (int, float))
        ]
        durations = [float(self._metrics[key]) for key in duration_keys]
        average_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total_operations": len([key for key in self._metrics if key.endswith("_start")]),
            "average_duration": average_duration,
            "metrics": {
                key: value
                for key, value in self._metrics.items()
                if not key.endswith(("_start", "_duration", "_last"))
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }


__all__ = [
    "StructuredLogger",
    "AuditLogger",
    "PerformanceLogger",
]
