"""
Null StatusReporter adapter for graceful degradation.

Provides a no-op implementation of the StatusReporter interface
when the real StatusReporter is not available. This allows the TUI
to launch and operate in degraded mode with limited functionality.
"""

import asyncio
import time
from collections.abc import Callable
from decimal import Decimal

from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BotStatus,
    EngineStatus,
    HeartbeatStatus,
    MarketStatus,
    PositionStatus,
    RiskStatus,
    StrategyStatus,
    SystemStatus,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui.adapters")


class NullStatusReporter:
    """
    No-op StatusReporter for degraded mode.

    Implements the same interface as StatusReporter and DemoStatusReporter
    but returns static unavailable data instead of real updates. Observers
    are stored but never notified since there's no data source.

    Attributes:
        is_null_reporter: Always True, used to identify this adapter.
    """

    is_null_reporter: bool = True

    def __init__(self) -> None:
        """Initialize the null reporter."""
        self._observers: list[Callable[[BotStatus], None]] = []
        self._running = False
        logger.warning("NullStatusReporter initialized (degraded mode)")

    def add_observer(self, callback: Callable[[BotStatus], None]) -> None:
        """
        Register an observer (stored but never notified).

        Args:
            callback: The observer callback function.
        """
        if callback not in self._observers:
            self._observers.append(callback)
            logger.debug(f"NullStatusReporter: Observer registered (total: {len(self._observers)})")

    def remove_observer(self, callback: Callable[[BotStatus], None]) -> None:
        """
        Unregister an observer.

        Args:
            callback: The observer callback function to remove.
        """
        if callback in self._observers:
            self._observers.remove(callback)
            logger.debug(f"NullStatusReporter: Observer removed (total: {len(self._observers)})")

    def get_status(self) -> BotStatus:
        """
        Get current status snapshot.

        Returns:
            A BotStatus indicating unavailable/degraded state.
        """
        return _create_unavailable_status()

    async def start(self) -> asyncio.Task | None:
        """
        Start the reporter (no-op).

        Returns:
            None since there's no background task to run.
        """
        self._running = True
        logger.debug("NullStatusReporter: start() called (no-op)")
        return None

    async def stop(self) -> None:
        """Stop the reporter (no-op)."""
        self._running = False
        logger.debug("NullStatusReporter: stop() called (no-op)")


def _create_unavailable_status() -> BotStatus:
    """
    Create a BotStatus indicating data is unavailable.

    Returns:
        A minimal BotStatus with healthy=False and appropriate health issues.
    """
    return BotStatus(
        bot_id="unavailable",
        timestamp=time.time(),
        timestamp_iso="",  # Will be set by __post_init__
        version="--",
        engine=EngineStatus(
            running=False,
            uptime_seconds=0.0,
            cycle_count=0,
            last_cycle_time=None,
            errors_count=0,
        ),
        market=MarketStatus(
            symbols=[],
            last_prices={},
            last_price_update=None,
            price_history={},
        ),
        positions=PositionStatus(
            count=0,
            symbols=[],
            total_unrealized_pnl=Decimal("0"),
            equity=Decimal("0"),
            positions={},
        ),
        orders=[],
        trades=[],
        account=AccountStatus(
            volume_30d=Decimal("0"),
            fees_30d=Decimal("0"),
            fee_tier="",
            balances=[],
        ),
        strategy=StrategyStatus(
            active_strategies=[],
            last_decisions=[],
        ),
        risk=RiskStatus(
            max_leverage=0.0,
            daily_loss_limit_pct=0.0,
            current_daily_loss_pct=0.0,
            reduce_only_mode=False,
            reduce_only_reason="",
            guards=[],
        ),
        system=SystemStatus(
            api_latency=0.0,
            connection_status="UNAVAILABLE",
            rate_limit_usage="--",
            memory_usage="--",
            cpu_usage="--",
        ),
        heartbeat=HeartbeatStatus(
            enabled=False,
            running=False,
            heartbeat_count=0,
            last_heartbeat=None,
            is_healthy=False,
        ),
        healthy=False,
        health_issues=["StatusReporter not available - limited functionality"],
    )
