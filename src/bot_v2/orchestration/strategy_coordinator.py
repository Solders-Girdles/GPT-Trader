"""Legacy strategy coordinator facade wrapping the coordinator package."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.context_builder import build_coordinator_context
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.strategy import StrategyCoordinator as _StrategyCoordinator

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.core.interfaces import Order, Product
    from bot_v2.orchestration.perps_bot import PerpsBot
    from bot_v2.orchestration.symbol_processor import SymbolProcessor

logger = logging.getLogger(__name__)


class StrategyCoordinator(_StrategyCoordinator):
    """Compatibility layer preserving the historical StrategyCoordinator API."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot
        context = self._build_context(bot)
        super().__init__(context)
        self._sync_bot(context)

    # ------------------------------------------------------------------
    @property
    def symbol_processor(self) -> SymbolProcessor:
        self._refresh_context_from_bot()
        return super().symbol_processor

    def set_symbol_processor(self, processor: SymbolProcessor | None) -> None:
        self._refresh_context_from_bot()
        super().set_symbol_processor(processor)

    async def process_symbol(
        self,
        symbol: str,
        balances: Any | None = None,
        position_map: dict[str, Any] | None = None,
    ) -> None:
        self._refresh_context_from_bot()
        await super().process_symbol(symbol, balances, position_map)

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        self._refresh_context_from_bot()
        await super().execute_decision(symbol, decision, mark, product, position_state)

    def ensure_order_lock(self) -> asyncio.Lock:
        self._refresh_context_from_bot()
        return super().ensure_order_lock()

    async def place_order(self, **kwargs: Any) -> Order | None:
        self._refresh_context_from_bot()
        return await super().place_order(**kwargs)

    async def place_order_inner(self, **kwargs: Any) -> Order | None:
        self._refresh_context_from_bot()
        return await super().place_order_inner(**kwargs)

    async def run_cycle(self) -> None:
        self._refresh_context_from_bot()
        await super().run_cycle()

    async def update_marks(self) -> None:
        self._refresh_context_from_bot()
        await super().update_marks()

    def update_mark_window(self, symbol: str, mark: Decimal) -> None:
        self._refresh_context_from_bot()
        super()._update_mark_window(symbol, mark)

    @staticmethod
    def calculate_spread_bps(bid_price: Decimal, ask_price: Decimal) -> Decimal:
        return _StrategyCoordinator.calculate_spread_bps(bid_price, ask_price)

    # ------------------------------------------------------------------
    def _refresh_context_from_bot(self) -> CoordinatorContext:
        updated = self._build_context(self._bot)
        super().update_context(updated)
        return self.context

    def _build_context(self, bot: PerpsBot) -> CoordinatorContext:
        return build_coordinator_context(
            bot,
            overrides={
                "strategy_coordinator": self,
            },
        )

    def _sync_bot(self, context: CoordinatorContext) -> None:
        bot = self._bot
        if hasattr(bot, "registry"):
            bot.registry = context.registry
        if context.event_store is not None and hasattr(bot, "event_store"):
            bot.event_store = context.event_store
        if context.orders_store is not None and hasattr(bot, "orders_store"):
            bot.orders_store = context.orders_store
        if context.product_cache is not None and hasattr(bot, "_state"):
            bot._state.product_map = context.product_cache
        if context.execution_coordinator is None and hasattr(bot, "execution_coordinator"):
            context = context.with_updates(execution_coordinator=bot.execution_coordinator)
            super().update_context(context)


__all__ = ["StrategyCoordinator"]
