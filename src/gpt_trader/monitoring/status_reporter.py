"""
File-based status reporter for operational monitoring.

This module provides a StatusReporter that periodically writes the bot's
internal state to a JSON file for external monitoring tools or human inspection.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.monitoring.heartbeat import HeartbeatService

logger = get_logger(__name__, component="status_reporter")


@dataclass
class EngineStatus:
    """Status snapshot of the trading engine."""

    running: bool = False
    uptime_seconds: float = 0.0
    cycle_count: int = 0
    last_cycle_time: float | None = None
    errors_count: int = 0
    last_error: str | None = None
    last_error_time: float | None = None


@dataclass
class MarketStatus:
    """Status snapshot of market data."""

    symbols: list[str] = field(default_factory=list)
    last_prices: dict[str, str] = field(default_factory=dict)
    last_price_update: float | None = None


@dataclass
class PositionStatus:
    """Status snapshot of positions."""

    count: int = 0
    symbols: list[str] = field(default_factory=list)
    total_unrealized_pnl: str = "0"


@dataclass
class HeartbeatStatus:
    """Status snapshot of heartbeat service."""

    enabled: bool = False
    running: bool = False
    heartbeat_count: int = 0
    last_heartbeat: float | None = None
    is_healthy: bool = False


@dataclass
class BotStatus:
    """Complete status snapshot of the trading bot."""

    bot_id: str = ""
    timestamp: float = field(default_factory=time.time)
    timestamp_iso: str = ""
    version: str = "1.0.0"

    engine: EngineStatus = field(default_factory=EngineStatus)
    market: MarketStatus = field(default_factory=MarketStatus)
    positions: PositionStatus = field(default_factory=PositionStatus)
    heartbeat: HeartbeatStatus = field(default_factory=HeartbeatStatus)

    # Overall health
    healthy: bool = True
    health_issues: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp_iso:
            self.timestamp_iso = datetime.utcfromtimestamp(self.timestamp).isoformat() + "Z"


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


@dataclass
class StatusReporter:
    """
    Periodically writes bot status to a JSON file.

    Features:
    - Atomic file writes (write to temp, then rename)
    - Configurable update interval
    - Includes engine, market, position, and heartbeat status
    - Health summary with issue detection

    Usage:
        reporter = StatusReporter(
            status_file="/var/run/gpt-trader/status.json",
            update_interval=10,
        )
        await reporter.start()
        # ... later ...
        reporter.update_engine_status(running=True, cycle_count=42)
        # ... shutdown ...
        await reporter.stop()
    """

    status_file: str = "status.json"
    update_interval: int = 10  # seconds
    bot_id: str = ""
    enabled: bool = True

    # Internal state
    _running: bool = field(default=False, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _status: BotStatus = field(default_factory=BotStatus, repr=False)

    # Mutable status tracking
    _cycle_count: int = field(default=0, repr=False)
    _errors_count: int = field(default=0, repr=False)
    _last_error: str | None = field(default=None, repr=False)
    _last_error_time: float | None = field(default=None, repr=False)
    _last_prices: dict[str, Decimal] = field(default_factory=dict, repr=False)
    _last_price_update: float | None = field(default=None, repr=False)
    _positions: dict[str, Any] = field(default_factory=dict, repr=False)
    _heartbeat_service: HeartbeatService | None = field(default=None, repr=False)

    async def start(self) -> asyncio.Task[None] | None:
        """Start the status reporter background task."""
        if not self.enabled:
            logger.info("Status reporter disabled")
            return None

        if self._running:
            logger.warning("Status reporter already running")
            return self._task

        self._running = True
        self._start_time = time.time()
        self._status = BotStatus(bot_id=self.bot_id)

        # Ensure directory exists
        status_path = Path(self.status_file)
        status_path.parent.mkdir(parents=True, exist_ok=True)

        # Write initial status
        await self._write_status()

        self._task = asyncio.create_task(self._report_loop())
        logger.info(
            f"Status reporter started (file={self.status_file}, interval={self.update_interval}s)"
        )
        return self._task

    async def stop(self) -> None:
        """Stop the status reporter."""
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

        # Write final status
        await self._write_status()
        logger.info("Status reporter stopped")

    async def _report_loop(self) -> None:
        """Main reporting loop."""
        while self._running:
            try:
                await self._write_status()
            except Exception as e:
                logger.error(f"Status report error: {e}")

            await asyncio.sleep(self.update_interval)

    async def _write_status(self) -> None:
        """Write current status to file atomically."""
        self._update_status()
        status_dict = asdict(self._status)

        # Atomic write: write to temp file, then rename
        status_path = Path(self.status_file)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=status_path.parent,
            prefix=".status_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(temp_fd, "w") as f:
                json.dump(status_dict, f, indent=2, cls=DecimalEncoder)
            os.rename(temp_path, self.status_file)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _update_status(self) -> None:
        """Update the status object with current values."""
        now = time.time()
        self._status.timestamp = now
        self._status.timestamp_iso = datetime.utcfromtimestamp(now).isoformat() + "Z"
        self._status.bot_id = self.bot_id

        # Engine status
        self._status.engine.running = self._running
        self._status.engine.uptime_seconds = now - self._start_time if self._start_time else 0
        self._status.engine.cycle_count = self._cycle_count
        self._status.engine.errors_count = self._errors_count
        self._status.engine.last_error = self._last_error
        self._status.engine.last_error_time = self._last_error_time

        # Market status
        self._status.market.last_prices = {k: str(v) for k, v in self._last_prices.items()}
        self._status.market.last_price_update = self._last_price_update
        self._status.market.symbols = list(self._last_prices.keys())

        # Position status
        self._status.positions.count = len(self._positions)
        self._status.positions.symbols = list(self._positions.keys())
        total_pnl = sum(
            (p.get("unrealized_pnl", Decimal("0")) for p in self._positions.values()),
            Decimal("0"),
        )
        self._status.positions.total_unrealized_pnl = str(total_pnl)

        # Heartbeat status
        if self._heartbeat_service:
            hb_status = self._heartbeat_service.get_status()
            self._status.heartbeat.enabled = hb_status.get("enabled", False)
            self._status.heartbeat.running = hb_status.get("running", False)
            self._status.heartbeat.heartbeat_count = hb_status.get("heartbeat_count", 0)
            self._status.heartbeat.last_heartbeat = hb_status.get("last_heartbeat")
            self._status.heartbeat.is_healthy = self._heartbeat_service.is_healthy

        # Health assessment
        self._assess_health()

    def _assess_health(self) -> None:
        """Assess overall health and populate issues list."""
        issues: list[str] = []

        # Check engine running
        if not self._status.engine.running:
            issues.append("Engine not running")

        # Check for recent errors
        if self._errors_count > 0 and self._last_error_time:
            time_since_error = time.time() - self._last_error_time
            if time_since_error < 300:  # Error in last 5 minutes
                issues.append(f"Recent error: {self._last_error}")

        # Check price staleness
        if self._last_price_update:
            time_since_price = time.time() - self._last_price_update
            if time_since_price > 120:  # No price update in 2 minutes
                issues.append(f"Stale prices ({int(time_since_price)}s old)")

        # Check heartbeat health
        if self._heartbeat_service and self._status.heartbeat.enabled:
            if not self._status.heartbeat.is_healthy:
                issues.append("Heartbeat unhealthy")

        self._status.healthy = len(issues) == 0
        self._status.health_issues = issues

    # --- Update methods called by TradingEngine ---

    def set_heartbeat_service(self, service: HeartbeatService) -> None:
        """Set the heartbeat service reference for status reporting."""
        self._heartbeat_service = service

    def record_cycle(self) -> None:
        """Record a completed trading cycle."""
        self._cycle_count += 1
        self._status.engine.last_cycle_time = time.time()

    def record_error(self, error: str) -> None:
        """Record an error occurrence."""
        self._errors_count += 1
        self._last_error = error
        self._last_error_time = time.time()

    def update_price(self, symbol: str, price: Decimal) -> None:
        """Update the last known price for a symbol."""
        self._last_prices[symbol] = price
        self._last_price_update = time.time()

    def update_positions(self, positions: dict[str, Any]) -> None:
        """Update the current positions."""
        self._positions = positions

    def get_status(self) -> dict[str, Any]:
        """Get current status as a dictionary."""
        self._update_status()
        return asdict(self._status)


__all__ = ["StatusReporter", "BotStatus", "EngineStatus", "MarketStatus", "PositionStatus"]
