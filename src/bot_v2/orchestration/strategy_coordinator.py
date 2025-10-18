"""Legacy strategy coordinator facade wrapping the coordinator package."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.coordinator_facades import (
    BaseCoordinatorFacade,
    ContextPreservingCoordinator,
)
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.strategy import StrategyCoordinator as _StrategyCoordinator
from bot_v2.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.core.interfaces import Order, Product
    from bot_v2.orchestration.perps_bot import PerpsBot
    from bot_v2.orchestration.symbol_processor import SymbolProcessor

logger = get_logger(__name__, component="strategy_coordinator_facade")


class StrategyCoordinator(
    BaseCoordinatorFacade,
    ContextPreservingCoordinator,
    _StrategyCoordinator,
):
    """Compatibility layer preserving the historical StrategyCoordinator API."""

    def __init__(self, bot: PerpsBot) -> None:
        context = self._setup_facade(
            bot,
            overrides={"strategy_coordinator": self},
        )
        super().__init__(context)
        self._sync_bot(context)

    # ------------------------------------------------------------------
    @property
    def symbol_processor(self) -> SymbolProcessor:
        self._refresh_context_from_bot()
        return super().symbol_processor

    @ContextPreservingCoordinator.context_action()
    def set_symbol_processor(self, processor: SymbolProcessor | None) -> None:
        super().set_symbol_processor(processor)

    @ContextPreservingCoordinator.context_action()
    async def process_symbol(
        self,
        symbol: str,
        balances: Any | None = None,
        position_map: dict[str, Any] | None = None,
    ) -> None:
        await super().process_symbol(symbol, balances, position_map)

    @ContextPreservingCoordinator.context_action()
    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        await super().execute_decision(symbol, decision, mark, product, position_state)

    @ContextPreservingCoordinator.context_action()
    def ensure_order_lock(self) -> asyncio.Lock:
        return super().ensure_order_lock()

    @ContextPreservingCoordinator.context_action()
    async def place_order(self, **kwargs: Any) -> Order | None:
        return await super().place_order(**kwargs)

    @ContextPreservingCoordinator.context_action()
    async def place_order_inner(self, **kwargs: Any) -> Order | None:
        return await super().place_order_inner(**kwargs)

    @ContextPreservingCoordinator.context_action()
    async def run_cycle(self) -> None:
        await super().run_cycle()

    @ContextPreservingCoordinator.context_action()
    async def update_marks(self) -> None:
        await super().update_marks()

    @ContextPreservingCoordinator.context_action()
    def update_mark_window(self, symbol: str, mark: Decimal) -> None:
        super()._update_mark_window(symbol, mark)

    @staticmethod
    def calculate_spread_bps(bid_price: Decimal, ask_price: Decimal) -> Decimal:
        return _StrategyCoordinator.calculate_spread_bps(bid_price, ask_price)

    # ------------------------------------------------------------------
    def _context_overrides(self, bot: PerpsBot) -> dict[str, Any]:  # type: ignore[override]
        return {
            "strategy_coordinator": self,
            "execution_coordinator": getattr(bot, "execution_coordinator", None),
        }

    def _sync_bot(self, context: CoordinatorContext) -> None:
        bot = self._bot
        if bot is None:
            return
        if hasattr(bot, "registry"):
            bot.registry = context.registry
        if context.event_store is not None and hasattr(bot, "event_store"):
            bot.event_store = context.event_store
        if context.orders_store is not None and hasattr(bot, "orders_store"):
            bot.orders_store = context.orders_store
        if context.product_cache is not None and hasattr(bot, "_state"):
            bot._state.product_map = context.product_cache


__all__ = ["StrategyCoordinator"]
