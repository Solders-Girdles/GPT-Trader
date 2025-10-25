"""Bot lifecycle management separated from perps_bot.py.

This module contains lifecycle management logic that was previously
embedded in the large perps_bot.py file. It provides:

- Bot startup and shutdown coordination
- Component lifecycle orchestration
- Streaming management for real-time data
- Account telemetry and monitoring
- Clean error handling and logging
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.config_controller import ConfigController
    from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
    from bot_v2.orchestration.session_guard import TradingSessionGuard

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="perps_lifecycle_management")


class PerpsBotLifecycleManager:
    """Service responsible for PerpsBot lifecycle management.

    This service consolidates lifecycle-related logic that was previously
    distributed throughout the large PerpsBot class, providing focused
    responsibility for bot startup, operation, and shutdown procedures.
    """

    def __init__(
        self,
        config_controller: ConfigController,
        bot_state: PerpsBotRuntimeState,
        session_guard: TradingSessionGuard,
    ) -> None:
        """Initialize bot lifecycle manager.

        Args:
            config_controller: Configuration management controller
            bot_state: Runtime state instance for the bot
            session_guard: Trading session guard for time windows
        """
        self.config_controller = config_controller
        self.bot_state = bot_state
        self.session_guard = session_guard
        self._streaming_task: asyncio.Task[Any] | None = None
        self._telemetry_task: asyncio.Task[Any] | None = None
        self._running = False

    async def start_lifecycle(self) -> None:
        """Start all bot lifecycle components."""
        if self._running:
            logger.warning(
                "Bot lifecycle already running",
                operation="lifecycle_start",
                status="already_running",
            )
            return

        self._running = True
        logger.info(
            "Starting PerpsBot lifecycle management",
            operation="lifecycle_start",
            config=self.config_controller.current.profile.value,
        )

        # Initialize bot state components
        await self._initialize_symbols_state()

        # Start background services
        await self._start_account_telemetry()
        await self._start_streaming_if_enabled()

        logger.info(
            "PerpsBot lifecycle management started successfully",
            operation="lifecycle_ready",
        )

    async def stop_lifecycle(self) -> None:
        """Stop all bot lifecycle components."""
        if not self._running:
            logger.warning(
                "Bot lifecycle not running",
                operation="lifecycle_stop",
                status="not_running",
            )
            return

        self._running = False
        logger.info(
            "Stopping PerpsBot lifecycle management",
            operation="lifecycle_stop",
        )

        # Stop background services
        await self._stop_streaming()
        await self._stop_account_telemetry()

        logger.info(
            "PerpsBot lifecycle management stopped",
            operation="lifecycle_stopped",
        )

    async def _initialize_symbols_state(self) -> None:
        """Initialize symbol processing state."""
        # This would handle symbol-specific initialization
        # For now, placeholder implementation
        logger.debug(
            "Symbol state initialization completed",
            operation="symbols_init",
            symbol_count=len(self.bot_state.symbols or []),
        )

    async def _start_account_telemetry(self) -> None:
        """Start account telemetry collection."""
        if not self._running:
            return

        logger.info(
            "Starting account telemetry",
            operation="telemetry_start",
            interval_seconds=300,
        )

        # Account telemetry logic would go here
        # This is a placeholder for the actual implementation

    async def _stop_account_telemetry(self) -> None:
        """Stop account telemetry collection."""
        if self._telemetry_task:
            self._telemetry_task.cancel()
            try:
                await self._telemetry_task
            except asyncio.CancelledError:
                pass
            self._telemetry_task = None

        logger.info(
            "Account telemetry stopped",
            operation="telemetry_stop",
        )

    async def _start_streaming_if_enabled(self) -> None:
        """Start streaming if enabled in configuration."""
        if not self._running:
            return

        config = self.config_controller.current
        if not config.perps_enable_streaming:
            logger.info(
                "Streaming disabled in configuration",
                operation="streaming_disabled",
                reason="configuration",
            )
            return

        await self._start_streaming_background()

    async def _start_streaming_background(self) -> None:
        """Start background streaming task."""
        if self._streaming_task:
            logger.warning(
                "Streaming task already running",
                operation="streaming_start",
                status="already_running",
            )
            return

        config = self.config_controller.current
        symbols = self.bot_state.symbols or []

        # Start streaming loop for each symbol
        self._streaming_task = asyncio.create_task(
            self._run_stream_loop(symbols, config.perps_stream_level)
        )

        logger.info(
            "Background streaming started",
            operation="streaming_started",
            symbol_count=len(symbols),
            stream_level=config.perps_stream_level,
        )

    async def _stop_streaming(self) -> None:
        """Stop streaming tasks."""
        if not self._streaming_task:
            return

        logger.info(
            "Stopping background streaming",
            operation="streaming_stop",
        )

        self._streaming_task.cancel()
        try:
            await self._streaming_task
        except asyncio.CancelledError:
            pass
        self._streaming_task = None

        logger.info(
            "Background streaming stopped",
            operation="streaming_stopped",
        )

    async def _run_stream_loop(self, symbols: list[str], level: int) -> None:
        """Run the main streaming loop."""
        try:
            # Placeholder streaming implementation
            # Real implementation would handle WebSocket connections,
            # market data processing, order book management, etc.
            while self._running:
                # Process symbols at specified level
                for symbol in symbols:
                    logger.debug(
                        f"Processing symbol {symbol} at level {level}",
                        operation="streaming_process",
                        symbol=symbol,
                        level=level,
                    )
                    # Simulate some work
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(
                "Streaming loop cancelled",
                operation="streaming_cancelled",
            )
        except Exception as exc:
            logger.error(
                "Streaming loop error",
                operation="streaming_error",
                error=str(exc),
                exc_info=True,
            )

    def is_running(self) -> bool:
        """Check if lifecycle management is running."""
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get current status of lifecycle management."""
        return {
            "running": self._running,
            "streaming_active": self._streaming_task is not None,
            "telemetry_active": self._telemetry_task is not None,
            "session_guard_active": self.session_guard.is_trading_allowed(),
        }


__all__ = [
    "PerpsBotLifecycleManager",
]
