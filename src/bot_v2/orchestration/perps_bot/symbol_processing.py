"""Symbol processing helpers and adapter implementations."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bot_v2.features.brokerages.core.interfaces import Balance, Order, Position, Product
from bot_v2.logging import symbol_context
from bot_v2.orchestration.coordinators import StrategyCoordinator

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from bot_v2.orchestration.perps_bot import PerpsBot
    from bot_v2.orchestration.symbol_processor import SymbolProcessor


class _CallableSymbolProcessor:
    """Adapter allowing bare callables to masquerade as SymbolProcessor implementations."""

    def __init__(
        self,
        func: Any,
        *,
        requires_context: bool,
    ) -> None:
        self._func = func
        self.requires_context = requires_context

    def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> Any:
        if self.requires_context:
            return self._func(symbol, balances, position_map)
        return self._func(symbol)

    @property
    def function(self) -> Any:
        return self._func


class PerpsBotSymbolProcessingMixin:
    """Encapsulates symbol processing orchestration."""

    @property
    def symbol_processor(self) -> SymbolProcessor | None:
        return self.strategy_coordinator.symbol_processor

    def set_symbol_processor(self, processor: SymbolProcessor | None) -> None:
        if isinstance(processor, _CallableSymbolProcessor):
            self._symbol_processor_override = processor
        else:
            self._symbol_processor_override = None
        self.strategy_coordinator.set_symbol_processor(processor)

    async def process_symbol(
        self: PerpsBot,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> None:
        # Symbol context will be added by the strategy coordinator
        await self.strategy_coordinator.process_symbol(symbol, balances, position_map)

    def _process_symbol_expects_context(self) -> bool:
        return bool(self.strategy_coordinator._process_symbol_expects_context())

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        # Create symbol context for decision execution
        with symbol_context(symbol):
            await self.strategy_coordinator.execute_decision(
                symbol, decision, mark, product, position_state
            )

    def _ensure_order_lock(self):
        lock = self.strategy_coordinator.ensure_order_lock()
        return cast(asyncio.Lock, lock)

    async def _place_order(self, **kwargs: Any) -> Order | None:
        return await self.strategy_coordinator.place_order(**kwargs)

    async def _place_order_inner(self, **kwargs: Any) -> Order | None:
        return await self.strategy_coordinator.place_order_inner(**kwargs)

    def _update_mark_window(self, symbol: str, mark: Decimal) -> None:
        self.strategy_coordinator.update_mark_window(symbol, mark)

    @staticmethod
    def _calculate_spread_bps(bid_price: Decimal, ask_price: Decimal) -> Decimal:
        return cast(Decimal, StrategyCoordinator.calculate_spread_bps(bid_price, ask_price))

    def _wrap_symbol_processor(self, handler: Any) -> _CallableSymbolProcessor:
        """Convert a callable into a SymbolProcessor compatible adapter."""

        if not callable(handler):
            raise TypeError("process_symbol handler must be callable")

        try:
            parameters = inspect.signature(handler).parameters
        except (TypeError, ValueError):
            parameters = inspect.Signature(parameters=()).parameters
        requires_context = len(parameters) > 1
        return _CallableSymbolProcessor(handler, requires_context=requires_context)

    def _install_symbol_processor_override(self, handler: Any) -> None:
        """Install or remove a legacy process_symbol override."""

        if handler is None:
            self._symbol_processor_override = None
            self.strategy_coordinator.set_symbol_processor(None)
            return
        wrapped = self._wrap_symbol_processor(handler)
        self._symbol_processor_override = wrapped
        self.strategy_coordinator.set_symbol_processor(wrapped)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "process_symbol":
            if value is not None and not callable(value):
                raise TypeError("process_symbol override must be callable or None")
            self._install_symbol_processor_override(value)
            return
        super().__setattr__(name, value)


__all__ = ["_CallableSymbolProcessor", "PerpsBotSymbolProcessingMixin"]
