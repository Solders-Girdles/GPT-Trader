"""
Simplified TradingBot.
Acts as the main entry point runner.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.app.container import ApplicationContainer
    from gpt_trader.core import Product
    from gpt_trader.features.brokerages.core.protocols import BrokerProtocol
    from gpt_trader.features.live_trade.risk.protocols import RiskManagerProtocol
    from gpt_trader.monitoring.notifications.service import NotificationService
    from gpt_trader.orchestration.protocols import (
        AccountManagerProtocol,
        EventStoreProtocol,
        RuntimeStateProtocol,
    )

logger = get_logger(__name__, component="trading_bot")


class TradingBot:
    def __init__(
        self,
        config: BotConfig,
        container: ApplicationContainer,
        event_store: EventStoreProtocol | None = None,
        orders_store: Any = None,
        notification_service: NotificationService | None = None,
    ) -> None:
        self.config = config
        self.container = container
        self.running = False

        # Get services directly from container (no longer using ServiceRegistry)
        self.broker: BrokerProtocol | None = container.broker
        self.risk_manager: RiskManagerProtocol | None = container.risk_manager
        self.account_manager: AccountManagerProtocol | None = getattr(
            container, "account_manager", None
        )
        self.account_telemetry: Any = getattr(container, "account_telemetry", None)
        self.runtime_state: RuntimeStateProtocol | None = getattr(container, "runtime_state", None)

        # Get event_store from parameter or container
        self._event_store = event_store or container.event_store

        # Get notification_service from parameter or container
        self._notification_service = notification_service or container.notification_service

        # Setup context
        self.context = CoordinatorContext(
            config=config,
            container=container,
            broker=self.broker,
            symbols=tuple(config.symbols),
            risk_manager=self.risk_manager,
            event_store=self._event_store,
            notification_service=self._notification_service,
        )

        self.engine = TradingEngine(self.context)

    async def run(self, single_cycle: bool = False) -> None:
        self.running = True
        logger.info("=" * 60)
        logger.info(f"TradingBot.run() called - Starting with symbols: {self.config.symbols}")
        logger.info(f"Interval: {self.config.interval}s")
        logger.info(f"Read-only mode: {getattr(self.config, 'read_only', False)}")
        logger.info("=" * 60)

        tasks = await self.engine.start_background_tasks()
        logger.info(f"Started {len(tasks)} background tasks")

        try:
            if single_cycle:
                # For testing/dev-fast
                await asyncio.sleep(self.config.interval + 0.1)
                await self.engine.shutdown()
            else:
                # Keep alive until tasks complete (which is never, unless error or shutdown)
                logger.info("Bot entering main loop (gathering background tasks)")
                await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Bot stopped (CancelledError caught).")
        finally:
            logger.info("Bot shutting down...")
            await self.engine.shutdown()
            self.running = False
            logger.info("Bot shutdown complete.")

    async def stop(self) -> None:
        self.running = False
        await self.engine.shutdown()

    async def flatten_and_stop(self) -> list[str]:
        """
        Emergency shutdown: Stop bot and close all open positions.
        Returns a list of messages describing actions taken.

        Note: This path intentionally bypasses the canonical guard stack
        (TradingEngine.submit_order) because emergency closures must succeed
        even when guards would block normal trading.
        """
        self.running = False
        logger.warning(
            "EMERGENCY: Initiating Flatten & Stop",
            bypass_reason="emergency_shutdown",
            operation="flatten_and_stop",
        )
        messages = ["Bot stopped."]

        if not self.broker:
            messages.append("Error: No broker connection available.")
            await self.engine.shutdown()
            return messages

        try:
            # 1. Fetch positions
            positions = await asyncio.to_thread(self.broker.list_positions)
            if not positions:
                messages.append("No open positions found.")
            else:
                # 2. Close each position
                from gpt_trader.core import OrderSide, OrderType

                for pos in positions:
                    try:
                        # Determine closing side
                        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY
                        # Use absolute quantity for order
                        quantity = abs(pos.quantity)

                        # Direct broker call bypasses guard stack intentionally
                        await asyncio.to_thread(
                            self.broker.place_order, pos.symbol, side, OrderType.MARKET, quantity
                        )
                        logger.info(
                            "Emergency close submitted",
                            symbol=pos.symbol,
                            quantity=str(quantity),
                            bypass_reason="emergency_shutdown",
                            operation="flatten_and_stop",
                        )
                        messages.append(f"Submitted CLOSE for {pos.symbol} ({quantity})")
                    except Exception as e:
                        logger.error(f"Failed to close {pos.symbol}: {e}")
                        messages.append(f"Failed to close {pos.symbol}: {e}")

        except Exception as e:
            logger.error(f"Flatten failed: {e}")
            messages.append(f"Critical Error during flatten: {e}")

        await self.engine.shutdown()
        return messages

    async def shutdown(self) -> None:
        """Alias for stop() to match CLI interface."""
        await self.stop()

    def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Any = None,
        product: Any = None,
        position_state: Any = None,
    ) -> Any:
        """Execute a trading decision. Stub for interface compatibility."""
        if hasattr(self.engine, "execute_decision"):
            return self.engine.execute_decision(symbol, decision, mark, product, position_state)
        return None

    def get_product(self, symbol: str) -> Product | None:
        """Get product metadata for a symbol."""
        if self.broker and hasattr(self.broker, "get_product"):
            return self.broker.get_product(symbol)
        return None


__all__ = ["TradingBot"]
