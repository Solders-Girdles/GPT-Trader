"""
Simplified TradingBot.
Acts as the main entry point runner.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from gpt_trader.app.config import BotConfig
from gpt_trader.app.runtime_ui_adapter import NullUIAdapter, RuntimeUIAdapter
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.features.live_trade.lifecycle import (
    TRADING_BOT_TRANSITIONS,
    LifecycleStateMachine,
    TradingBotState,
)
from gpt_trader.utilities.async_tools import BoundedToThread
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.app.container import ApplicationContainer
    from gpt_trader.app.protocols import (
        AccountManagerProtocol,
        EventStoreProtocol,
        RuntimeStateProtocol,
    )
    from gpt_trader.core import Product
    from gpt_trader.features.brokerages.core.protocols import BrokerProtocol
    from gpt_trader.features.live_trade.risk.protocols import RiskManagerProtocol
    from gpt_trader.monitoring.notifications.service import NotificationService

logger = get_logger(__name__, component="trading_bot")


class TradingBot:
    def __init__(
        self,
        config: BotConfig,
        container: ApplicationContainer,
        event_store: EventStoreProtocol | None = None,
        orders_store: Any = None,
        notification_service: NotificationService | None = None,
        ui_adapter: RuntimeUIAdapter | None = None,
    ) -> None:
        self.config = config
        self.container = container
        self._lifecycle: LifecycleStateMachine[TradingBotState] = LifecycleStateMachine(
            initial_state=TradingBotState.INIT,
            entity="trading_bot",
            transitions=TRADING_BOT_TRANSITIONS,
            logger=logger,
        )

        # Get services directly from container (legacy registry removed)
        self.broker: BrokerProtocol | None = container.broker
        self.risk_manager: RiskManagerProtocol | None = container.risk_manager
        self.account_manager: AccountManagerProtocol | None = getattr(
            container, "account_manager", None
        )
        self.account_telemetry: Any = getattr(container, "account_telemetry", None)
        self.runtime_state: RuntimeStateProtocol | None = getattr(container, "runtime_state", None)

        # Get event_store from parameter or container
        self._event_store = event_store or container.event_store

        # Get orders_store from parameter or container
        self._orders_store = orders_store or container.orders_store

        # Get notification_service from parameter or container
        self._notification_service = notification_service or container.notification_service

        # Concurrency-limited entrypoint for sync broker calls from async code.
        broker_call_limit = getattr(config, "max_concurrent_broker_calls", None)
        if broker_call_limit is None:
            broker_call_limit = getattr(config, "max_concurrent_rest_calls", 5)
        try:
            broker_call_limit = int(broker_call_limit)
        except (TypeError, ValueError):
            broker_call_limit = 5
        broker_call_limit = max(1, broker_call_limit)
        self._broker_calls = BoundedToThread(max_concurrency=broker_call_limit)

        # Setup context
        self.context = CoordinatorContext(
            config=config,
            container=container,
            broker=self.broker,
            broker_calls=self._broker_calls,
            symbols=tuple(config.symbols),
            risk_manager=self.risk_manager,
            event_store=self._event_store,
            orders_store=self._orders_store,
            notification_service=self._notification_service,
        )

        self.engine = TradingEngine(self.context)
        self.ui_adapter: RuntimeUIAdapter = ui_adapter or NullUIAdapter()
        self.ui_adapter.attach(self)

    @property
    def state(self) -> TradingBotState:
        return self._lifecycle.state

    @property
    def running(self) -> bool:
        return self.state in (TradingBotState.STARTING, TradingBotState.RUNNING)

    @running.setter
    def running(self, value: bool) -> None:
        target = TradingBotState.RUNNING if value else TradingBotState.STOPPED
        self._lifecycle.transition(
            target,
            reason="running_override",
            details={"via": "running_set"},
            force=True,
        )

    def _transition_state(
        self,
        target: TradingBotState,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        return self._lifecycle.transition(
            target,
            reason=reason,
            details=details,
            force=force,
        )

    def set_ui_adapter(self, adapter: RuntimeUIAdapter | None) -> None:
        """Attach a runtime UI adapter (no-op when None)."""
        if adapter is None:
            adapter = NullUIAdapter()

        if adapter is self.ui_adapter:
            return

        self.ui_adapter.detach()
        self.ui_adapter = adapter
        self.ui_adapter.attach(self)

    async def run(self, single_cycle: bool = False) -> None:
        self._transition_state(
            TradingBotState.STARTING,
            reason="run_called",
            details={"single_cycle": single_cycle},
        )
        logger.info("=" * 60)
        logger.info(f"TradingBot.run() called - Starting with symbols: {self.config.symbols}")
        logger.info(f"Interval: {self.config.interval}s")
        logger.info(f"Read-only mode: {getattr(self.config, 'read_only', False)}")
        logger.info("=" * 60)

        tasks = await self.engine.start_background_tasks()
        self._transition_state(
            TradingBotState.RUNNING,
            reason="background_tasks_started",
            details={"task_count": len(tasks)},
        )
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
        except Exception as exc:
            self._transition_state(
                TradingBotState.ERROR,
                reason="run_failed",
                details={"error": str(exc)},
            )
            raise
        finally:
            logger.info("Bot shutting down...")
            self._transition_state(TradingBotState.STOPPING, reason="shutdown_start")
            await self.engine.shutdown()
            self._transition_state(TradingBotState.STOPPED, reason="shutdown_complete")
            logger.info("Bot shutdown complete.")

    async def stop(self) -> None:
        self._transition_state(TradingBotState.STOPPING, reason="stop_called")
        await self.engine.shutdown()
        self._transition_state(TradingBotState.STOPPED, reason="stop_complete")

    async def flatten_and_stop(self) -> list[str]:
        """
        Emergency shutdown: Stop bot and close all open positions.
        Returns a list of messages describing actions taken.

        Note: This path intentionally bypasses the canonical guard stack
        (TradingEngine.submit_order) because emergency closures must succeed
        even when guards would block normal trading.
        """
        self._transition_state(
            TradingBotState.STOPPING,
            reason="flatten_and_stop",
            details={"bypass_reason": "emergency_shutdown"},
        )
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
            broker_calls = getattr(self, "_broker_calls", None)
            if broker_calls is not None and asyncio.iscoroutinefunction(
                getattr(broker_calls, "__call__", None)
            ):
                positions = await broker_calls(self.broker.list_positions)
            else:
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
                        if broker_calls is not None and asyncio.iscoroutinefunction(
                            getattr(broker_calls, "__call__", None)
                        ):
                            await broker_calls(
                                self.broker.place_order,
                                pos.symbol,
                                side,
                                OrderType.MARKET,
                                quantity,
                            )
                        else:
                            await asyncio.to_thread(
                                self.broker.place_order,
                                pos.symbol,
                                side,
                                OrderType.MARKET,
                                quantity,
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
        self._transition_state(TradingBotState.STOPPED, reason="flatten_and_stop_complete")
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
