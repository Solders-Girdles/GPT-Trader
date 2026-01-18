"""Context preparation helpers for strategy orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol, cast

from gpt_trader.core import Balance, Position
from gpt_trader.utilities.quantities import quantity_from

from .logging_utils import logger  # naming: allow
from .models import SymbolProcessingContext

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.market_data_features import DepthSnapshot
    from gpt_trader.features.live_trade.bot import TradingBot


class _HasBot(Protocol):
    """Protocol for mixins that expect a _bot attribute."""

    _bot: TradingBot


class ContextBuilderMixin(_HasBot):
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

        # Get orderbook snapshot and trade stats (optional, for advanced strategies)
        orderbook_snapshot = self._get_orderbook_snapshot(symbol)
        trade_stats = self._get_trade_volume_stats(symbol)
        spread_bps = (
            Decimal(str(orderbook_snapshot.spread_bps))
            if orderbook_snapshot and orderbook_snapshot.spread_bps
            else None
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
            orderbook_snapshot=orderbook_snapshot,
            trade_volume_stats=trade_stats,
            spread_bps=spread_bps,
        )

        if not self._run_risk_gates(context):
            return None
        return context

    async def _ensure_balances(self, balances: Sequence[Balance] | None) -> Sequence[Balance]:
        if balances is not None:
            return balances
        assert self._bot.broker is not None, "Broker not initialized"
        broker_calls = getattr(getattr(self._bot, "context", None), "broker_calls", None)
        if broker_calls is not None and asyncio.iscoroutinefunction(
            getattr(broker_calls, "__call__", None)
        ):
            return await broker_calls(self._bot.broker.list_balances)
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
        if bot.risk_manager is None:
            return False
        if getattr(bot.risk_manager.config, "kill_switch_enabled", False):
            logger.warning("Kill switch enabled - skipping trading loop")
            return True
        return False

    async def _ensure_positions(
        self, position_map: dict[str, Position] | None
    ) -> dict[str, Position]:
        if position_map is not None:
            return position_map
        assert self._bot.broker is not None, "Broker not initialized"
        broker_calls = getattr(getattr(self._bot, "context", None), "broker_calls", None)
        if broker_calls is not None and asyncio.iscoroutinefunction(
            getattr(broker_calls, "__call__", None)
        ):
            positions = await broker_calls(self._bot.broker.list_positions)
        else:
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
        except (ValueError, ArithmeticError):
            quantity = Decimal("0")
        return state, quantity

    def _get_marks(self, symbol: str) -> list[Decimal]:
        raw_marks = self._bot.mark_windows.get(symbol, [])  # type: ignore[attr-defined]
        return [Decimal(str(mark)) for mark in raw_marks]

    def _get_orderbook_snapshot(self, symbol: str) -> DepthSnapshot | None:
        """Get latest orderbook snapshot for symbol."""
        runtime_state = getattr(self._bot, "runtime_state", None)
        if runtime_state is None:
            return None
        if not hasattr(runtime_state, "orderbook_lock"):
            return None
        from gpt_trader.features.brokerages.coinbase.market_data_features import DepthSnapshot

        with runtime_state.orderbook_lock:
            snapshot = runtime_state.orderbook_snapshots.get(symbol)
        return cast(DepthSnapshot | None, snapshot)

    def _get_trade_volume_stats(self, symbol: str) -> dict[str, Any] | None:
        """Get trade volume statistics for symbol."""
        runtime_state = getattr(self._bot, "runtime_state", None)
        if runtime_state is None:
            return None
        if not hasattr(runtime_state, "trade_lock"):
            return None
        with runtime_state.trade_lock:
            agg = runtime_state.trade_aggregators.get(symbol)
            return agg.get_stats() if agg else None

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
        if bot.risk_manager is None:
            return True  # Skip risk checks if no risk manager

        try:
            try:
                long_period = int(
                    getattr(getattr(bot.config, "strategy", None), "long_ma_period", 20)
                )
            except (TypeError, ValueError):
                long_period = 20
            window = context.marks[-max(long_period, 20) :]
            cb = bot.risk_manager.check_volatility_circuit_breaker(context.symbol, list(window))
            if cb is not None and cb.triggered:
                logger.warning(
                    "Volatility circuit breaker tripped for %s: %s", context.symbol, cb.reason
                )
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
