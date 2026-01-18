"""
System maintenance service for TradingEngine.

Extracted from TradingEngine to separate concerns:
- Report system health metrics (CPU, memory, latency)
- Periodically prune EventStore to prevent unbounded growth
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

from gpt_trader.monitoring.metrics_collector import record_gauge
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.app.protocols import EventStoreProtocol
    from gpt_trader.monitoring.status_reporter import StatusReporter

logger = get_logger(__name__, component="system_maintenance")

# Default pruning configuration
DEFAULT_PRUNE_INTERVAL_SECONDS = 3600  # 1 hour
DEFAULT_PRUNE_MAX_ROWS = 1_000_000  # Keep 1M events max


class SystemMaintenanceService:
    """
    Manages system health reporting and database maintenance.

    Responsibilities:
    - Collect and report system health metrics (CPU, memory, latency)
    - Periodically prune EventStore to prevent unbounded growth
    """

    def __init__(
        self,
        status_reporter: StatusReporter,
        event_store: EventStoreProtocol | None = None,
        prune_interval_seconds: int = DEFAULT_PRUNE_INTERVAL_SECONDS,
        prune_max_rows: int = DEFAULT_PRUNE_MAX_ROWS,
    ) -> None:
        """
        Initialize the system maintenance service.

        Args:
            status_reporter: StatusReporter for health metric updates
            event_store: EventStore for pruning (can be None)
            prune_interval_seconds: Interval between prune operations
            prune_max_rows: Maximum rows to retain after pruning
        """
        self._status_reporter = status_reporter
        self._event_store = event_store
        self._prune_interval_seconds = prune_interval_seconds
        self._prune_max_rows = prune_max_rows
        self._running = False

    def report_system_status(
        self,
        latency_seconds: float,
        connection_status: str,
    ) -> None:
        """
        Collect and report system health metrics.

        Also emits Prometheus gauges for process memory and event store cache.

        Args:
            latency_seconds: Last measured API latency in seconds
            connection_status: Current connection status string
        """
        memory_usage, cpu_usage = self._collect_process_metrics()

        # Emit event store cache gauges if available
        self._collect_event_store_metrics()

        # Convert latency to milliseconds
        latency_ms = latency_seconds * 1000

        # Rate limit usage placeholder (would need broker headers)
        rate_limit = "OK"

        self._status_reporter.update_system(
            latency=latency_ms,
            connection=connection_status,
            rate_limit=rate_limit,
            memory=memory_usage,
            cpu=cpu_usage,
        )

    def _collect_event_store_metrics(self) -> None:
        """Collect and emit event store cache metrics as Prometheus gauges."""
        if self._event_store is None:
            return

        try:
            # Get cache size (number of events)
            if hasattr(self._event_store, "get_cache_size"):
                cache_size = self._event_store.get_cache_size()
                record_gauge("gpt_trader_event_store_cache_size", float(cache_size))

            # Get cache fill ratio
            if hasattr(self._event_store, "get_cache_fill_ratio"):
                fill_ratio = self._event_store.get_cache_fill_ratio()
                record_gauge("gpt_trader_deque_cache_fill_ratio", fill_ratio)
        except Exception as e:
            logger.debug("Failed to collect event store metrics: %s", e)

    def _collect_process_metrics(self) -> tuple[str, str]:
        """
        Collect CPU and memory metrics for current process.

        Also emits a Prometheus gauge for process memory in MB.

        Returns:
            Tuple of (memory_usage, cpu_usage) as formatted strings
        """

        def is_valid_metric(value: Any) -> bool:
            return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0

        try:
            import psutil

            process = psutil.Process()
        except ImportError:
            return "N/A", "N/A"
        except Exception:
            return "Unknown", "Unknown"

        memory_usage = "Unknown"
        cpu_usage = "Unknown"

        try:
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            if is_valid_metric(memory_mb):
                memory_usage = f"{memory_mb:.1f}MB"
                record_gauge("gpt_trader_process_memory_mb", memory_mb)
        except Exception:
            memory_usage = "Unknown"

        try:
            cpu_value = process.cpu_percent()
            if is_valid_metric(cpu_value):
                cpu_usage = f"{cpu_value}%"
        except Exception:
            cpu_usage = "Unknown"

        return memory_usage, cpu_usage

    async def start_prune_loop(self) -> asyncio.Task[Any]:
        """
        Start the database pruning background task.

        Returns:
            The asyncio Task running the prune loop
        """
        self._running = True
        return asyncio.create_task(self._prune_loop())

    async def stop(self) -> None:
        """Signal the prune loop to stop."""
        self._running = False

    async def _prune_loop(self) -> None:
        """Periodically prune the event store to prevent unbounded growth."""
        logger.info(
            "Starting database prune task",
            interval_seconds=self._prune_interval_seconds,
            max_rows=self._prune_max_rows,
        )

        while self._running:
            await asyncio.sleep(self._prune_interval_seconds)

            if not self._running:
                break

            if self._event_store is None:
                continue

            try:
                # Check if the event store supports pruning
                if hasattr(self._event_store, "prune"):
                    # Run pruning in a separate thread to avoid blocking the event loop
                    pruned = await asyncio.to_thread(
                        self._event_store.prune, max_rows=self._prune_max_rows
                    )
                    if pruned > 0:
                        logger.info("Pruned old events from database", count=pruned)
            except Exception as e:
                logger.error("Database pruning failed", error=str(e))

        logger.debug("Prune loop stopped")
