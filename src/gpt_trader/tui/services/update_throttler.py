"""
Update Throttler for TUI.

Batches high-frequency state updates to reduce UI redraws and flicker.
Especially useful during rapid market data updates or reconnections.

The throttler collects updates within a configurable interval and
applies them all at once, reducing the number of UI refresh cycles.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.monitoring.status_reporter import BotStatus

logger = get_logger(__name__, component="tui")


@dataclass
class ThrottlerStats:
    """Statistics about throttler performance."""

    total_updates_received: int = 0
    total_batches_flushed: int = 0
    updates_in_current_batch: int = 0
    last_flush_time: float = 0.0

    @property
    def average_batch_size(self) -> float:
        """Average number of updates per batch."""
        if self.total_batches_flushed == 0:
            return 0.0
        return self.total_updates_received / self.total_batches_flushed


@dataclass
class UpdateThrottler:
    """
    Batches high-frequency updates to reduce UI redraws.

    Usage:
        throttler = UpdateThrottler(min_interval=0.1)
        throttler.set_flush_callback(apply_updates)

        # Queue updates as they come in
        throttler.queue_update("market", market_data)
        throttler.queue_update("positions", position_data)

        # Updates will be batched and flushed together
    """

    min_interval: float = 0.1  # Minimum time between flushes (seconds)
    enabled: bool = True

    # Internal state
    _pending_updates: dict[str, Any] = field(default_factory=dict)
    _flush_task: asyncio.Task | None = field(default=None, repr=False)
    _flush_callback: Callable[[dict[str, Any]], None] | None = field(default=None, repr=False)
    _stats: ThrottlerStats = field(default_factory=ThrottlerStats)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def set_flush_callback(
        self,
        callback: Callable[[dict[str, Any]], None],
    ) -> None:
        """
        Set the callback to invoke when flushing updates.

        Args:
            callback: Function that receives dict of component->data updates
        """
        self._flush_callback = callback

    def queue_update(self, component: str, data: Any) -> None:
        """
        Queue an update for batching.

        If multiple updates for the same component arrive before flush,
        only the latest one is kept (last-write-wins).

        Args:
            component: Component name (e.g., "market", "positions", "orders")
            data: Update data for that component
        """
        if not self.enabled:
            # If disabled, apply immediately via callback
            if self._flush_callback:
                self._flush_callback({component: data})
            return

        self._pending_updates[component] = data
        self._stats.total_updates_received += 1
        self._stats.updates_in_current_batch += 1

        # Schedule flush if not already scheduled
        self._schedule_flush()

    def queue_full_status(self, status: BotStatus) -> None:
        """
        Queue a full BotStatus update.

        Extracts individual components and queues them separately.

        Args:
            status: Complete BotStatus snapshot
        """
        if not self.enabled:
            if self._flush_callback:
                self._flush_callback({"full_status": status})
            return

        # Store full status for bulk processing
        self._pending_updates["full_status"] = status
        self._stats.total_updates_received += 1
        self._stats.updates_in_current_batch += 1

        self._schedule_flush()

    def _schedule_flush(self) -> None:
        """Schedule a flush after min_interval if not already scheduled."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_after_delay())

    async def _flush_after_delay(self) -> None:
        """Wait for min_interval then flush all pending updates."""
        await asyncio.sleep(self.min_interval)
        await self.flush()

    async def flush(self) -> None:
        """Flush all pending updates immediately."""
        async with self._lock:
            if not self._pending_updates:
                return

            updates = self._pending_updates.copy()
            self._pending_updates.clear()

            # Update stats
            self._stats.total_batches_flushed += 1
            self._stats.updates_in_current_batch = 0
            import time

            self._stats.last_flush_time = time.time()

            # Invoke callback with batched updates
            if self._flush_callback:
                try:
                    self._flush_callback(updates)
                    logger.debug(
                        f"Flushed {len(updates)} component updates: {list(updates.keys())}"
                    )
                except Exception as e:
                    logger.error(f"Error in flush callback: {e}", exc_info=True)

    def flush_sync(self) -> None:
        """
        Synchronous flush for use in non-async contexts.

        Creates a task to flush but doesn't wait for it.
        """
        if self._pending_updates:
            asyncio.create_task(self.flush())

    def cancel_pending(self) -> int:
        """
        Cancel any pending flush and clear updates.

        Returns:
            Number of updates that were discarded
        """
        discarded = len(self._pending_updates)

        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            self._flush_task = None

        self._pending_updates.clear()
        self._stats.updates_in_current_batch = 0

        if discarded > 0:
            logger.debug(f"Cancelled {discarded} pending updates")

        return discarded

    def get_stats(self) -> ThrottlerStats:
        """Get throttler statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = ThrottlerStats()

    @property
    def pending_count(self) -> int:
        """Number of updates waiting to be flushed."""
        return len(self._pending_updates)

    @property
    def is_flush_scheduled(self) -> bool:
        """Check if a flush is scheduled."""
        return self._flush_task is not None and not self._flush_task.done()

    def enable(self) -> None:
        """Enable throttling."""
        self.enabled = True
        logger.debug("Update throttling enabled")

    def disable(self) -> None:
        """
        Disable throttling and flush pending updates.

        When disabled, updates are applied immediately.
        """
        self.enabled = False
        self.flush_sync()
        logger.debug("Update throttling disabled")

    def set_interval(self, interval: float) -> None:
        """
        Set the minimum interval between flushes.

        Args:
            interval: Minimum seconds between flushes (0.05 to 1.0 recommended)
        """
        if interval < 0.01:
            logger.warning(f"Throttle interval {interval} too low, using 0.01")
            interval = 0.01
        elif interval > 2.0:
            logger.warning(f"Throttle interval {interval} too high, using 2.0")
            interval = 2.0

        self.min_interval = interval
        logger.debug(f"Throttle interval set to {interval}s")
