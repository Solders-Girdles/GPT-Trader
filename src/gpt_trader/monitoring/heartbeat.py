"""
Heartbeat service for dead man's switch functionality.

This module provides a HeartbeatService that:
- Periodically writes heartbeat events to EventStore for audit/debugging
- Optionally pings an external dead man's switch URL (e.g., Healthchecks.io)
- Enables external alerting if the trading bot goes silent
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_trader.orchestration.protocols import EventStoreProtocol

logger = logging.getLogger(__name__)

# Event type for heartbeats
EVENT_HEARTBEAT = "heartbeat"


@dataclass
class HeartbeatService:
    """
    Background service that sends periodic heartbeat signals.

    Features:
    - Records heartbeat events to EventStore for debugging/audit
    - Pings external dead man's switch URL for monitoring
    - Configurable interval and metadata

    Usage:
        service = HeartbeatService(
            event_store=event_store,
            ping_url="https://hc-ping.com/your-uuid",
            interval_seconds=60,
        )
        task = await service.start()
        # ... later ...
        await service.stop()
    """

    event_store: EventStoreProtocol | None = None
    ping_url: str | None = None
    interval_seconds: int = 60
    bot_id: str = ""
    enabled: bool = True

    # Internal state
    _running: bool = field(default=False, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _last_heartbeat: float = field(default=0.0, repr=False)
    _heartbeat_count: int = field(default=0, repr=False)

    async def start(self) -> asyncio.Task[None] | None:
        """Start the heartbeat background task.

        Returns:
            The background task, or None if disabled/already running.
        """
        if not self.enabled:
            logger.info("Heartbeat service disabled")
            return None

        if self._running:
            logger.warning("Heartbeat service already running")
            return self._task

        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            f"Heartbeat service started (interval={self.interval_seconds}s, "
            f"ping_url={'configured' if self.ping_url else 'none'})"
        )
        return self._task

    async def stop(self) -> None:
        """Stop the heartbeat service."""
        if not self._running:
            return

        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info(f"Heartbeat service stopped (sent {self._heartbeat_count} heartbeats)")

    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await self._send_heartbeat()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            await asyncio.sleep(self.interval_seconds)

    async def _send_heartbeat(self) -> None:
        """Send a single heartbeat."""
        now = time.time()
        self._heartbeat_count += 1
        self._last_heartbeat = now

        # Record to EventStore
        if self.event_store is not None:
            self.event_store.store(
                {
                    "type": EVENT_HEARTBEAT,
                    "data": {
                        "timestamp": now,
                        "bot_id": self.bot_id,
                        "count": self._heartbeat_count,
                    },
                }
            )

        # Ping external URL
        if self.ping_url:
            await self._ping_external()

        logger.debug(f"Heartbeat #{self._heartbeat_count} sent")

    async def _ping_external(self) -> bool:
        """Ping external dead man's switch URL.

        Returns:
            True if ping succeeded, False otherwise.
        """
        if not self.ping_url:
            return False

        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.ping_url) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.warning(f"Heartbeat ping returned {response.status}")
                        return False
        except TimeoutError:
            logger.warning("Heartbeat ping timed out")
            return False
        except Exception as e:
            logger.warning(f"Heartbeat ping failed: {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get heartbeat service status."""
        return {
            "enabled": self.enabled,
            "running": self._running,
            "interval_seconds": self.interval_seconds,
            "ping_url_configured": bool(self.ping_url),
            "heartbeat_count": self._heartbeat_count,
            "last_heartbeat": self._last_heartbeat,
            "seconds_since_last": (
                time.time() - self._last_heartbeat if self._last_heartbeat > 0 else None
            ),
        }

    @property
    def is_healthy(self) -> bool:
        """Check if heartbeat is healthy (recent heartbeat sent)."""
        if not self._running:
            return False
        if self._last_heartbeat == 0:
            return True  # Just started, no heartbeat yet
        # Healthy if last heartbeat within 2x interval
        return (time.time() - self._last_heartbeat) < (self.interval_seconds * 2)


__all__ = ["HeartbeatService", "EVENT_HEARTBEAT"]
