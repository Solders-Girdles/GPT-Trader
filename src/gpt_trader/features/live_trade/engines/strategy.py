"""
Simplified Strategy Engine.
Replaces the 608-line enterprise coordinator with a simple loop.
"""

import asyncio
import logging
from collections import defaultdict
from decimal import Decimal
from typing import Any

from gpt_trader.features.live_trade.engines.base import (
    BaseEngine,
    CoordinatorContext,
    HealthStatus,
)
from gpt_trader.features.live_trade.risk.manager import ValidationError
from gpt_trader.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
)

logger = logging.getLogger(__name__)


class TradingEngine(BaseEngine):
    """
    Simple trading loop that fetches data and executes strategy.
    """

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self.running = False
        self.strategy = BaselinePerpsStrategy(config=self.context.config)
        self.price_history: dict[str, list[Decimal]] = defaultdict(list)

    @property
    def name(self) -> str:
        return "strategy"

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        """Start the main trading loop."""
        self.running = True
        task = asyncio.create_task(self._run_loop())
        self._register_background_task(task)
        return [task]

    async def _run_loop(self) -> None:
        logger.info("Starting strategy loop...")
        while self.running:
            try:
                await self._cycle()
            except Exception as e:
                logger.error(f"Error in strategy cycle: {e}", exc_info=True)

            await asyncio.sleep(self.context.config.interval)

    async def _cycle(self) -> None:
        """One trading cycle."""
        assert self.context.broker is not None, "Broker not initialized"
        # 1. Fetch Data
        for symbol in self.context.config.symbols:
            # Offload blocking network call
            try:
                ticker = await asyncio.to_thread(self.context.broker.get_ticker, symbol)
            except Exception as e:
                logger.error(f"Failed to fetch ticker for {symbol}: {e}")
                continue

            price = Decimal(str(ticker.get("price", 0)))
            logger.info(f"{symbol} price: {price}")

            self.price_history[symbol].append(price)
            if len(self.price_history[symbol]) > 20:
                self.price_history[symbol].pop(0)

            decision = self.strategy.decide(
                symbol=symbol,
                current_mark=price,
                position_state=None,
                recent_marks=self.price_history[symbol],
                equity=Decimal("1000"),
                product=None,
            )

            logger.info(f"Strategy Decision for {symbol}: {decision.action} ({decision.reason})")

            if decision.action in (Action.BUY, Action.SELL):
                logger.info(f"EXECUTING {decision.action} for {symbol}")
                try:
                    await self._validate_and_place_order(
                        symbol=symbol,
                        decision=decision,
                        price=price,
                    )
                except ValidationError as e:
                    logger.warning(f"Risk validation failed for {symbol}: {e}")
                except Exception as e:
                    logger.error(f"Order placement failed: {e}")

    async def _validate_and_place_order(
        self,
        symbol: str,
        decision: Decision,
        price: Decimal,
    ) -> None:
        """Validate order with risk manager before execution.

        Raises:
            ValidationError: If risk validation fails.
        """
        from gpt_trader.features.brokerages.core.interfaces import OrderSide, OrderType

        side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL
        quantity = Decimal("0.001")  # TODO: Use proper position sizing
        equity = Decimal("1000")  # TODO: Get real equity from runtime state

        # Run pre-trade validation if risk manager is available
        if self.context.risk_manager is not None:
            self.context.risk_manager.pre_trade_validate(
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=price,
                product=None,
                equity=equity,
                current_positions={},  # TODO: Track positions from runtime state
            )
            logger.info(f"Risk validation passed for {symbol} {side.value}")
        else:
            logger.warning("No risk manager configured - skipping validation")

        # Place order only after validation passes
        assert self.context.broker is not None, "Broker not initialized"
        await asyncio.to_thread(
            self.context.broker.place_order,
            symbol,
            side,
            OrderType.MARKET,
            quantity,
        )

    async def shutdown(self) -> None:
        self.running = False
        await super().shutdown()

    def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self.running, component=self.name)
