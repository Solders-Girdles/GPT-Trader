"""
Demo bot for TUI testing.

A lightweight mock bot that simulates trading activity without
connecting to real exchanges or executing real trades.
"""

import asyncio
import time
from typing import Any

from gpt_trader.tui.demo.mock_data import MockDataGenerator
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="demo")


class DemoEngine:
    """Mock trading engine for demo mode."""

    def __init__(self, data_generator: MockDataGenerator | None = None) -> None:
        self.status_reporter = DemoStatusReporter(data_generator=data_generator)
        self.context = DemoContext()


class DemoContext:
    """Mock coordinator context."""

    def __init__(self) -> None:
        self.runtime_state = DemoRuntimeState()


class DemoRuntimeState:
    """Mock runtime state."""

    def __init__(self) -> None:
        self.start_time = time.time()

    @property
    def uptime(self) -> float:
        """Calculate current uptime."""
        return time.time() - self.start_time


class DemoStatusReporter:
    """Mock status reporter that generates realistic data."""

    def __init__(
        self, update_interval: float = 2.0, data_generator: MockDataGenerator | None = None
    ) -> None:
        self.update_interval = update_interval
        self.data_generator = data_generator or MockDataGenerator()
        self._observers: list = []
        self._running = False
        self._task: asyncio.Task | None = None

    def add_observer(self, callback: Any) -> None:
        """Register an observer for status updates."""
        if callback not in self._observers:
            self._observers.append(callback)
            logger.info(f"Observer added. Total observers: {len(self._observers)}")

    def remove_observer(self, callback: Any) -> None:
        """Unregister an observer."""
        if callback in self._observers:
            self._observers.remove(callback)
            logger.info(f"Observer removed. Total observers: {len(self._observers)}")

    def get_status(self) -> dict[str, Any]:
        """Get current status snapshot."""
        return self.data_generator.generate_full_status()

    async def start(self) -> asyncio.Task:
        """Start the status reporter loop."""
        self._running = True
        self._task = asyncio.create_task(self._report_loop())
        logger.info("Demo status reporter started")
        return self._task

    async def stop(self) -> None:
        """Stop the status reporter loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Demo status reporter stopped")

    async def _report_loop(self) -> None:
        """Main reporting loop that notifies observers."""
        while self._running:
            try:
                # Generate new status
                status = self.data_generator.generate_full_status()

                # Notify all observers
                for observer in self._observers:
                    try:
                        if asyncio.iscoroutinefunction(observer):
                            await observer(status)
                        else:
                            observer(status)
                    except Exception as e:
                        logger.error(f"Error notifying observer: {e}", exc_info=True)

                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in demo report loop: {e}", exc_info=True)
                await asyncio.sleep(self.update_interval)


class DemoBot:
    """
    Mock bot for TUI demo mode.

    Simulates a trading bot without connecting to real exchanges.
    Perfect for UI development and testing.
    """

    def __init__(
        self, config: Any | None = None, data_generator: MockDataGenerator | None = None
    ) -> None:
        self.config = config or DemoConfig()
        self.engine = DemoEngine(data_generator=data_generator)
        self.running = False
        self._task: asyncio.Task | None = None

    async def run(self, single_cycle: bool = False) -> None:
        """Start the demo bot."""
        try:
            logger.info("Starting demo bot")
            self.running = True

            # Start status reporter
            await self.engine.status_reporter.start()

            # Keep running until stopped
            if not single_cycle:
                while self.running:
                    await asyncio.sleep(1)

            logger.info("Demo bot run loop exited")

        except asyncio.CancelledError:
            logger.info("Demo bot run cancelled")
            raise
        except Exception as e:
            logger.error(f"Demo bot error: {e}", exc_info=True)
            raise
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop the demo bot."""
        logger.info("Stopping demo bot")
        self.running = False
        await self.engine.status_reporter.stop()
        logger.info("Demo bot stopped")

    async def shutdown(self) -> None:
        """Alias for stop()."""
        await self.stop()

    async def flatten_and_stop(self) -> list[str]:
        """Mock panic sequence."""
        logger.warning("Demo bot: flatten_and_stop called (simulated)")
        messages = [
            "DEMO MODE: Would close all positions",
            "DEMO MODE: Would cancel all orders",
            "DEMO MODE: Bot stopped",
        ]
        await self.stop()
        return messages


class DemoConfig:
    """Mock configuration for demo bot."""

    def __init__(self) -> None:
        self.symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        self.update_interval = 2.0
        self.exchange = "DEMO"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "symbols": self.symbols,
            "update_interval": self.update_interval,
            "exchange": self.exchange,
            "mode": "DEMO",
        }
