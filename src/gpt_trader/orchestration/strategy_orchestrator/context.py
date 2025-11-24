"""Context preparation helpers for strategy orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, cast

from gpt_trader.features.brokerages.core.interfaces import Balance, Position
from gpt_trader.features.live_trade.risk_runtime import CircuitBreakerAction
from gpt_trader.utilities.quantities import quantity_from

from .logging_utils import logger
from .models import SymbolProcessingContext


class ContextBuilderMixin:
    """Prepare symbol processing context for strategies."""

    async def _prepare_context(
        self,
        symbol: str,
        balances: Sequence[Balance] | None,
        position_map: dict[str, Position] | None,
    ) -> SymbolProcessingContext | None:
        balances = await self._ensure_balances(balances)
        equity = self._extract_equity(balances)
        if self._kill_switch_engaged():
            return None

        positions_lookup = await self._ensure_positions(position_map)
        position_state, position_quantity = self._build_position_state(symbol, positions_lookup)

        marks = self._get_marks(symbol)
        if not marks:
            logger.warning(
                "No marks for %s",
                symbol,
                operation="strategy_prepare",
                stage="marks",
                symbol=symbol,
            )
            return None

        adjusted_equity = self._adjust_equity(equity, position_quantity, marks, symbol)
        if adjusted_equity == Decimal("0"):
            logger.error(
                "No equity info for %s",
                symbol,
                operation="strategy_prepare",
                stage="equity",
                symbol=symbol,
            )
            return None

        product = None
        try:
            product = self._bot.get_product(symbol)
        except Exception:
            logger.debug(
                "Failed to fetch product metadata for %s",
                symbol,
                exc_info=True,
                operation="strategy_prepare",
                stage="product",
                symbol=symbol,
            )

        context = SymbolProcessingContext(
            symbol=symbol,
            balances=balances,
            equity=adjusted_equity,
            positions=positions_lookup,
            position_state=position_state,
            position_quantity=position_quantity,
            marks=list(marks),
            product=product,
        )

        if not self._run_risk_gates(context):
            return None
        return context

    async def _ensure_balances(self, balances: Sequence[Balance] | None) -> Sequence[Balance]:
        if balances is not None:
            return balances
        return await asyncio.to_thread(self._bot.broker.list_balances)

    def _extract_equity(self, balances: Sequence[Balance]) -> Decimal:
        cash_assets = {"USD", "USDC"}
        usd_balance = next(
            (b for b in balances if getattr(b, "asset", "").upper() in cash_assets),
            None,
        )
        return cast(Decimal, usd_balance.total) if usd_balance is not None else Decimal("0")

    def _kill_switch_engaged(self) -> bool:
        bot = self._bot
        if getattr(bot.risk_manager.config, "kill_switch_enabled", False):
            logger.warning("Kill switch enabled - skipping trading loop")
            return True
        return False

    async def _ensure_positions(
        self, position_map: dict[str, Position] | None
    ) -> dict[str, Position]:
        if position_map is not None:
            return position_map
        positions = await asyncio.to_thread(self._bot.broker.list_positions)
        return {p.symbol: p for p in positions if hasattr(p, "symbol")}

    def _build_position_state(
        self, symbol: str, positions_lookup: dict[str, Position]
    ) -> tuple[dict[str, Any] | None, Decimal]:
        if symbol not in positions_lookup:
            return None, Decimal("0")

        pos = positions_lookup[symbol]
        quantity_val = quantity_from(pos, default=Decimal("0"))
        state = {
            "quantity": quantity_val,
            "side": getattr(pos, "side", "long"),
            "entry": getattr(pos, "entry_price", None),
        }
        try:
            quantity = Decimal(str(quantity_val))
        except Exception:
            quantity = Decimal("0")
        return state, quantity

    def _get_marks(self, symbol: str) -> list[Decimal]:
        raw_marks = self._bot.mark_windows.get(symbol, [])
        return [Decimal(str(mark)) for mark in raw_marks]

    def _adjust_equity(
        self, equity: Decimal, position_quantity: Decimal, marks: Sequence[Decimal], symbol: str
    ) -> Decimal:
        if position_quantity and marks:
            try:
                equity += abs(position_quantity) * marks[-1]
            except Exception as exc:
                logger.debug(
                    "Failed to adjust equity for %s position: %s", symbol, exc, exc_info=True
                )
        return equity

    def _run_risk_gates(self, context: SymbolProcessingContext) -> bool:
        bot = self._bot
        try:
            window = context.marks[-max(bot.config.long_ma, 20) :]
            cb = bot.risk_manager.check_volatility_circuit_breaker(context.symbol, list(window))
            if cb.triggered and cb.action is CircuitBreakerAction.KILL_SWITCH:
                logger.warning("Kill switch tripped by volatility CB for %s", context.symbol)
                return False
        except Exception as exc:
            logger.debug(
                "Volatility circuit breaker check failed for %s: %s",
                context.symbol,
                exc,
                exc_info=True,
            )

        try:
            if bot.risk_manager.check_mark_staleness(context.symbol):
                logger.warning("Skipping %s due to stale market data", context.symbol)
                return False
        except Exception as exc:
            logger.debug(
                "Mark staleness check failed for %s: %s", context.symbol, exc, exc_info=True
            )

        return True


__all__ = ["ContextBuilderMixin"]
