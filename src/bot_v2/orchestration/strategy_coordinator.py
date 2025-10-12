"""Strategy coordination helpers for :class:`bot_v2.orchestration.perps_bot.PerpsBot`."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.features.brokerages.core.interfaces import Balance, Order, Position, Product
from bot_v2.orchestration.symbol_processor import SymbolProcessor
from bot_v2.utilities import utc_now
from bot_v2.utilities.async_tools import gather_with_concurrency
from bot_v2.utilities.config import ConfigBaselinePayload

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from bot_v2.orchestration.perps_bot import PerpsBot


logger = logging.getLogger(__name__)

MAX_QUOTE_FETCH_CONCURRENCY = 10


class StrategyCoordinator:
    """Own the trading cycle and execution hand-off for ``PerpsBot``."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot
        self._symbol_processor: SymbolProcessor | None = None

    @property
    def symbol_processor(self) -> SymbolProcessor:
        """Return the active symbol processor, falling back to the orchestrator."""
        return self._resolve_symbol_processor()

    def set_symbol_processor(self, processor: SymbolProcessor | None) -> None:
        """Register a custom symbol processor or reset to the default."""
        self._symbol_processor = processor
        state = self._bot.runtime_state
        state.process_symbol_dispatch = None
        state.process_symbol_needs_context = None

    def _resolve_symbol_processor(self) -> SymbolProcessor:
        processor = self._symbol_processor
        if processor is None:
            processor = self._bot.strategy_orchestrator
        return processor

    # ------------------------------------------------------------------
    async def run_cycle(self) -> None:
        """Execute a single trading cycle with validation and trading logic."""

        bot = self._bot
        logger.debug("Running update cycle")

        current_state = await self._fetch_current_state()
        if not await self._validate_configuration_and_handle_drift(current_state):
            return

        await bot.update_marks()

        if not bot._session_guard.should_trade():
            logger.info("Outside trading window; skipping trading actions this cycle")
            await bot.system_monitor.log_status()
            return

        trading_state = dict(current_state)
        await self._execute_trading_cycle(trading_state)
        await bot.system_monitor.log_status()

    # ------------------------------------------------------------------
    async def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> None:
        processor = self._resolve_symbol_processor()
        needs_context = self._process_symbol_expects_context()
        args: list[Any] = [symbol]
        if needs_context:
            args.extend([balances, position_map])
        result = processor.process_symbol(*args)
        if inspect.isawaitable(result):
            await result
        elif result is not None:
            logger.debug("Symbol processor %s returned non-awaitable result", processor)

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        await self._bot.execution_coordinator.execute_decision(
            symbol, decision, mark, product, position_state
        )

    def ensure_order_lock(self) -> asyncio.Lock:
        return self._bot.execution_coordinator.ensure_order_lock()

    async def place_order(self, **kwargs: Any) -> Order | None:
        return await self._bot.execution_coordinator.place_order(self._bot.exec_engine, **kwargs)

    async def place_order_inner(self, **kwargs: Any) -> Order | None:
        return await self._bot.execution_coordinator.place_order_inner(**kwargs)

    # ------------------------------------------------------------------
    async def update_marks(self) -> None:
        bot = self._bot
        symbols = tuple(bot.symbols)
        if not symbols:
            return

        quote_coroutines = [asyncio.to_thread(bot.broker.get_quote, symbol) for symbol in symbols]
        quotes = await gather_with_concurrency(
            quote_coroutines,
            max_concurrency=MAX_QUOTE_FETCH_CONCURRENCY,
            return_exceptions=True,
        )

        for symbol, result in zip(symbols, quotes):
            if isinstance(result, Exception):
                logger.error("Error fetching quote for %s: %s", symbol, result)
                continue

            try:
                quote = result
                if quote is None:
                    raise RuntimeError(f"No quote for {symbol}")
                last_price = getattr(quote, "last", getattr(quote, "last_price", None))
                if last_price is None:
                    raise RuntimeError(f"Quote missing price for {symbol}")
                mark = Decimal(str(last_price))
                if mark <= 0:
                    raise RuntimeError(f"Invalid mark price: {mark} for {symbol}")
                ts = getattr(quote, "ts", datetime.now(UTC))
                self.update_mark_window(symbol, mark)
                try:
                    timestamp = ts if isinstance(ts, datetime) else utc_now()
                    risk_manager = bot.risk_manager
                    record_fn = getattr(risk_manager, "record_mark_update", None)
                    stored = record_fn(symbol, timestamp) if callable(record_fn) else timestamp
                    risk_manager.last_mark_update[symbol] = stored
                except Exception as exc:
                    logger.debug(
                        "Failed to update mark timestamp for %s: %s", symbol, exc, exc_info=True
                    )
            except Exception as exc:
                logger.error("Error updating mark for %s: %s", symbol, exc)

    def update_mark_window(self, symbol: str, mark: Decimal) -> None:
        bot = self._bot
        state = bot.runtime_state
        with state.mark_lock:
            window = state.mark_windows.setdefault(symbol, [])
            window.append(mark)
            max_size = max(bot.config.short_ma, bot.config.long_ma) + 5
            if len(window) > max_size:
                state.mark_windows[symbol] = window[-max_size:]

    @staticmethod
    def calculate_spread_bps(bid_price: Decimal, ask_price: Decimal) -> Decimal:
        try:
            mid = (bid_price + ask_price) / Decimal("2")
            if mid <= 0:
                return Decimal("0")
            return ((ask_price - bid_price) / mid) * Decimal("10000")
        except Exception:
            return Decimal("0")

    # ------------------------------------------------------------------
    async def _fetch_current_state(self) -> dict[str, Any]:
        bot = self._bot
        balances: Sequence[Balance] = []
        positions: Sequence[Position] = []
        account_equity: Decimal | None = None

        async def _call_in_thread(
            getter: Callable[[], Callable[[], Any]],
            log_fn: Callable[[Exception], None],
        ) -> Any:
            try:
                func = getter()
            except Exception as exc:
                log_fn(exc)
                return exc
            try:
                return await asyncio.to_thread(func)
            except Exception as exc:
                log_fn(exc)
                return exc

        balances_result, positions_result, account_info_result = await asyncio.gather(
            _call_in_thread(
                lambda: bot.broker.list_balances,
                lambda exc: logger.warning("Unable to fetch balances: %s", exc),
            ),
            _call_in_thread(
                lambda: bot.broker.list_positions,
                lambda exc: logger.warning("Unable to fetch positions: %s", exc),
            ),
            _call_in_thread(
                lambda: bot.broker.get_account_info,
                lambda exc: logger.debug("Unable to fetch account equity: %s", exc),
            ),
        )

        if not isinstance(balances_result, Exception):
            balances = balances_result if balances_result is not None else []

        if not isinstance(positions_result, Exception):
            positions = positions_result if positions_result is not None else []

        if not isinstance(account_info_result, Exception) and account_info_result is not None:
            account_equity = getattr(account_info_result, "equity", None)
            if account_equity is not None:
                account_equity = Decimal(str(account_equity))

        position_map = {}
        if positions:
            position_map = {p.symbol: p for p in positions if hasattr(p, "symbol")}

        return {
            "balances": balances,
            "positions": positions,
            "position_map": position_map,
            "account_equity": account_equity,
        }

    async def _validate_configuration_and_handle_drift(self, current_state: dict[str, Any]) -> bool:
        bot = self._bot
        current_config_dict = ConfigBaselinePayload.from_config(
            bot.config,
            derivatives_enabled=bool(getattr(bot.config, "derivatives_enabled", False)),
        ).to_dict()

        validation_result = bot.configuration_guardian.pre_cycle_check(
            proposed_config_dict=current_config_dict,
            current_balances=current_state["balances"],
            current_positions=current_state["positions"],
            current_equity=current_state["account_equity"],
        )

        if validation_result.is_valid:
            return True

        logger.warning("Configuration drift detected: %s", validation_result.errors)
        for error in validation_result.errors:
            logger.error("Configuration error: %s", error)

        has_critical_errors = any(
            "critical" in error.lower() or "emergency_shutdown" in error.lower()
            for error in validation_result.errors
        )

        if has_critical_errors:
            logger.critical(
                "Critical configuration violations detected - initiating emergency shutdown"
            )
            bot.running = False
            await bot.shutdown()
            return False

        logger.warning(
            "High-severity configuration violations detected - switching to reduce-only mode"
        )
        bot.set_reduce_only_mode(True, "Configuration drift detected")
        return False

    async def _execute_trading_cycle(self, trading_state: dict[str, Any]) -> None:
        expects_context = self._process_symbol_expects_context()

        tasks = []
        for symbol in self._bot.symbols:
            if expects_context:
                tasks.append(
                    self.process_symbol(
                        symbol, trading_state["balances"], trading_state["position_map"]
                    )
                )
            else:
                tasks.append(self.process_symbol(symbol))

        await asyncio.gather(*tasks)

    def _process_symbol_expects_context(self) -> bool:
        bot = self._bot
        state = bot.runtime_state
        processor = self._resolve_symbol_processor()
        bound_callable = processor.process_symbol
        dispatch = getattr(bound_callable, "__func__", bound_callable)
        needs_context = state.process_symbol_needs_context
        if needs_context is None or dispatch is not state.process_symbol_dispatch:
            requires_context = getattr(processor, "requires_context", None)
            if requires_context is None:
                process_sig = inspect.signature(dispatch)
                requires_context = len(process_sig.parameters) > 1
            needs_context = bool(requires_context)
            state.process_symbol_needs_context = needs_context
            state.process_symbol_dispatch = dispatch
        return bool(needs_context)
