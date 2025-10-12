"""Strategy coordination helpers for :class:`bot_v2.orchestration.perps_bot.PerpsBot`."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.features.brokerages.core.interfaces import Balance, Order, Position, Product
from bot_v2.utilities import utc_now
from bot_v2.utilities.config import ConfigBaselinePayload

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from bot_v2.orchestration.perps_bot import PerpsBot


logger = logging.getLogger(__name__)


class StrategyCoordinator:
    """Own the trading cycle and execution hand-off for ``PerpsBot``."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot

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
        bot = self._bot
        override = getattr(bot, "__dict__", {}).get("process_symbol")
        if override is not None:
            needs_context = self._process_symbol_expects_context()
            args: list[Any] = [symbol]
            if needs_context:
                args.extend([balances, position_map])
            result = override(*args)
            if inspect.isawaitable(result):
                await result
            return

        await bot.strategy_orchestrator.process_symbol(symbol, balances, position_map)

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
        return self._bot.execution_coordinator._ensure_order_lock()

    async def place_order(self, **kwargs: Any) -> Order | None:
        return await self._bot.execution_coordinator._place_order(self._bot.exec_engine, **kwargs)

    async def place_order_inner(self, **kwargs: Any) -> Order | None:
        return await self._bot.execution_coordinator._place_order_inner(**kwargs)

    # ------------------------------------------------------------------
    async def update_marks(self) -> None:
        bot = self._bot
        symbols = tuple(bot.symbols)
        if not symbols:
            return

        quotes = await asyncio.gather(
            *(asyncio.to_thread(bot.broker.get_quote, symbol) for symbol in symbols),
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
                    bot.risk_manager.last_mark_update[symbol] = (
                        ts if isinstance(ts, datetime) else utc_now()
                    )
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

        try:
            balances = await asyncio.to_thread(bot.broker.list_balances)
        except Exception as exc:
            logger.warning("Unable to fetch balances: %s", exc)

        try:
            positions = await asyncio.to_thread(bot.broker.list_positions)
        except Exception as exc:
            logger.warning("Unable to fetch positions: %s", exc)

        try:
            account_info = await asyncio.to_thread(bot.broker.get_account_info)
            account_equity = getattr(account_info, "equity", None)
            if account_equity is not None:
                account_equity = Decimal(str(account_equity))
        except Exception as exc:
            logger.debug("Unable to fetch account equity: %s", exc)

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
        override = getattr(bot, "__dict__", {}).get("process_symbol")
        if override is not None:
            current_dispatch = override
            target = override
        else:
            orchestrator_callable = bot.strategy_orchestrator.process_symbol
            current_dispatch = getattr(orchestrator_callable, "__func__", orchestrator_callable)
            target = orchestrator_callable
        needs_context = state.process_symbol_needs_context
        if needs_context is None or current_dispatch is not state.process_symbol_dispatch:
            process_sig = inspect.signature(target)
            needs_context = len(process_sig.parameters) > 1
            state.process_symbol_needs_context = needs_context
            state.process_symbol_dispatch = current_dispatch
        return bool(needs_context)
