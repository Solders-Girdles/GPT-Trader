"""Strategy coordinator responsible for the trading cycle orchestration."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Balance, Order, Position, Product
from bot_v2.utilities import utc_now
from bot_v2.utilities.async_tools import gather_with_concurrency
from bot_v2.utilities.config import ConfigBaselinePayload
from bot_v2.utilities.logging_patterns import get_logger

from ..symbol_processor import SymbolProcessor
from .base import BaseCoordinator, CoordinatorContext, HealthStatus
from .execution import ExecutionCoordinator

logger = get_logger(__name__, component="strategy_coordinator")

MAX_QUOTE_FETCH_CONCURRENCY = 10


class StrategyCoordinator(BaseCoordinator):
    """Owns the trading cycle and execution hand-off."""

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self._symbol_processor: SymbolProcessor | None = None

    @property
    def name(self) -> str:
        return "strategy"

    def initialize(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        return context or self.context

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        return []

    def health_check(self) -> HealthStatus:
        ctx = self.context
        healthy = ctx.broker is not None and ctx.risk_manager is not None
        runtime_state = ctx.runtime_state
        details = {
            "symbols_tracked": len(ctx.symbols),
            "has_broker": ctx.broker is not None,
            "has_risk_manager": ctx.risk_manager is not None,
        }
        if runtime_state is not None:
            details["last_decisions"] = len(getattr(runtime_state, "last_decisions", {}))
        return HealthStatus(healthy=healthy, component=self.name, details=details)

    # ------------------------------------------------------------------ symbol processor helpers
    @property
    def symbol_processor(self) -> SymbolProcessor:
        return self._resolve_symbol_processor()

    def set_symbol_processor(self, processor: SymbolProcessor | None) -> None:
        self._symbol_processor = processor
        runtime_state = self.context.runtime_state
        if runtime_state is not None:
            runtime_state.process_symbol_dispatch = None
            runtime_state.process_symbol_needs_context = None

    def _resolve_symbol_processor(self) -> SymbolProcessor:
        processor = self._symbol_processor
        if processor is None:
            orchestrator = self.context.strategy_orchestrator
            if orchestrator is None:
                raise RuntimeError("Strategy orchestrator not available in context")
            processor = orchestrator
        return processor

    # ------------------------------------------------------------------ trading cycle
    async def run_cycle(self) -> None:
        ctx = self.context
        logger.debug(
            "Running update cycle",
            operation="strategy_cycle",
            stage="start",
        )

        current_state = await self._fetch_current_state()
        if not await self._validate_configuration_and_handle_drift(current_state):
            return

        await self.update_marks()

        system_monitor = getattr(ctx, "system_monitor", None)
        session_guard = getattr(ctx, "session_guard", None)

        if session_guard is not None and hasattr(session_guard, "should_trade"):
            try:
                should_trade = bool(session_guard.should_trade())
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Session guard failed during should_trade",
                    error=str(exc),
                    operation="strategy_cycle",
                    stage="session_guard",
                )
                should_trade = True
        else:
            should_trade = True

        if not should_trade:
            logger.info(
                "Outside trading window; skipping trading actions this cycle",
                operation="strategy_cycle",
                stage="skip",
            )
            if system_monitor is not None:
                await system_monitor.log_status()
            return

        trading_state = dict(current_state)
        await self._execute_trading_cycle(trading_state)

        if system_monitor is not None:
            await system_monitor.log_status()

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
            logger.debug(
                "Symbol processor returned non-awaitable result",
                processor=str(processor),
                operation="process_symbol",
                stage="result",
            )

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        execution_coordinator: ExecutionCoordinator | None = self.context.execution_coordinator
        if execution_coordinator is None:
            raise RuntimeError("Execution coordinator not available for decision execution")
        await execution_coordinator.execute_decision(
            symbol, decision, mark, product, position_state
        )

    def ensure_order_lock(self) -> asyncio.Lock:
        execution_coordinator: ExecutionCoordinator | None = self.context.execution_coordinator
        if execution_coordinator is None:
            raise RuntimeError("Execution coordinator not available to ensure order lock")
        return execution_coordinator.ensure_order_lock()

    async def place_order(self, **kwargs: Any) -> Order | None:
        execution_coordinator: ExecutionCoordinator | None = self.context.execution_coordinator
        exec_engine = (
            getattr(self.context.runtime_state, "exec_engine", None)
            if self.context.runtime_state
            else None
        )
        if execution_coordinator is None or exec_engine is None:
            raise RuntimeError("Execution engine not initialized")
        return await execution_coordinator.place_order(exec_engine, **kwargs)

    async def place_order_inner(self, **kwargs: Any) -> Order | None:
        execution_coordinator: ExecutionCoordinator | None = self.context.execution_coordinator
        if execution_coordinator is None:
            raise RuntimeError("Execution coordinator not available")
        return await execution_coordinator.place_order_inner(**kwargs)

    # ------------------------------------------------------------------ mark updates
    async def update_marks(self) -> None:
        ctx = self.context
        broker = ctx.broker
        runtime_state = ctx.runtime_state
        if broker is None or runtime_state is None:
            return

        symbols = tuple(ctx.symbols)
        if not symbols:
            return

        quote_coroutines = [asyncio.to_thread(broker.get_quote, symbol) for symbol in symbols]
        quotes = await gather_with_concurrency(
            quote_coroutines,
            max_concurrency=MAX_QUOTE_FETCH_CONCURRENCY,
            return_exceptions=True,
        )

        for symbol, result in zip(symbols, quotes):
            if isinstance(result, Exception):
                logger.error(
                    "Error fetching quote",
                    symbol=symbol,
                    error=str(result),
                    operation="update_marks",
                    stage="fetch",
                    exc_info=True,
                )
                continue
            try:
                self._process_quote_update(symbol, result)
            except Exception as exc:
                logger.error(
                    "Error updating mark",
                    symbol=symbol,
                    error=str(exc),
                    operation="update_marks",
                    stage="process",
                    exc_info=True,
                )

    def _process_quote_update(self, symbol: str, quote: Any) -> None:
        if quote is None:
            raise RuntimeError(f"No quote for {symbol}")

        last_price = getattr(quote, "last", getattr(quote, "last_price", None))
        if last_price is None:
            raise RuntimeError(f"Quote missing price for {symbol}")

        mark = Decimal(str(last_price))
        if mark <= 0:
            raise RuntimeError(f"Invalid mark price: {mark} for {symbol}")

        self._update_mark_window(symbol, mark)
        timestamp = getattr(quote, "ts", datetime.now(UTC))
        self._record_mark_timestamp(symbol, timestamp)

    def _update_mark_window(self, symbol: str, mark: Decimal) -> None:
        runtime_state = self.context.runtime_state
        if runtime_state is None:
            return

        with runtime_state.mark_lock:
            window = runtime_state.mark_windows.setdefault(symbol, [])
            window.append(mark)
            max_size = max(self.context.config.short_ma, self.context.config.long_ma) + 5
            if len(window) > max_size:
                runtime_state.mark_windows[symbol] = window[-max_size:]

    def _record_mark_timestamp(self, symbol: str, ts: datetime) -> None:
        risk_manager = self.context.risk_manager
        if risk_manager is None:
            return
        try:
            timestamp = ts if isinstance(ts, datetime) else utc_now()
            record_fn = getattr(risk_manager, "record_mark_update", None)
            stored = record_fn(symbol, timestamp) if callable(record_fn) else timestamp
            if hasattr(risk_manager, "last_mark_update"):
                risk_manager.last_mark_update[symbol] = stored
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to update mark timestamp for %s: %s", symbol, exc, exc_info=True)

    # ------------------------------------------------------------------ helpers
    async def _fetch_current_state(self) -> dict[str, Any]:
        broker = self.context.broker
        if broker is None:
            return {
                "balances": [],
                "positions": [],
                "position_map": {},
                "account_equity": None,
            }

        async def _call_in_thread(method_name: str, log_fn: Any) -> Any:
            try:
                func = getattr(broker, method_name)
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
                "list_balances",
                lambda exc: logger.warning("Unable to fetch balances: %s", exc),
            ),
            _call_in_thread(
                "list_positions",
                lambda exc: logger.warning("Unable to fetch positions: %s", exc),
            ),
            _call_in_thread(
                "get_account_info",
                lambda exc: logger.debug("Unable to fetch account equity: %s", exc),
            ),
        )

        balances: Sequence[Balance] = []
        positions: Sequence[Position] = []
        account_equity: Decimal | None = None

        if not isinstance(balances_result, Exception) and balances_result is not None:
            try:
                balances = list(balances_result)
            except TypeError:
                balances = []
        if not isinstance(positions_result, Exception) and positions_result is not None:
            try:
                positions = list(positions_result)
            except TypeError:
                positions = []
        if not isinstance(account_info_result, Exception) and account_info_result is not None:
            equity_value = getattr(account_info_result, "equity", None)
            if equity_value is not None:
                try:
                    account_equity = Decimal(str(equity_value))
                except Exception:
                    account_equity = None

        position_map = {p.symbol: p for p in positions if hasattr(p, "symbol")}

        return {
            "balances": balances,
            "positions": positions,
            "position_map": position_map,
            "account_equity": account_equity,
        }

    async def _validate_configuration_and_handle_drift(self, current_state: dict[str, Any]) -> bool:
        guardian = getattr(self.context, "configuration_guardian", None)
        if guardian is None:
            return True

        config_payload = ConfigBaselinePayload.from_config(
            self.context.config,
            derivatives_enabled=bool(getattr(self.context.config, "derivatives_enabled", False)),
        )
        validation_result = guardian.pre_cycle_check(
            proposed_config_dict=config_payload.to_dict(),
            current_balances=current_state["balances"],
            current_positions=current_state["positions"],
            current_equity=current_state["account_equity"],
        )

        if validation_result.is_valid:
            return True

        logger.warning(
            "Configuration drift detected",
            errors=validation_result.errors,
            operation="strategy_cycle",
            stage="config_drift",
        )
        for error in validation_result.errors:
            logger.error(
                "Configuration error",
                error=error,
                operation="strategy_cycle",
                stage="config_drift",
            )

        has_critical_errors = any(
            "critical" in error.lower() or "emergency_shutdown" in error.lower()
            for error in validation_result.errors
        )

        if has_critical_errors:
            set_running_flag = getattr(self.context, "set_running_flag", None)
            if callable(set_running_flag):
                try:
                    set_running_flag(False)
                except Exception:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to update running flag during shutdown",
                        exc_info=True,
                        operation="strategy_cycle",
                        stage="shutdown",
                    )
            shutdown_hook = getattr(self.context, "shutdown_hook", None)
            if callable(shutdown_hook):
                try:
                    await shutdown_hook()
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Failed to execute shutdown hook after critical drift",
                        operation="strategy_cycle",
                        stage="shutdown",
                    )
            return False

        logger.warning(
            "High-severity configuration violations detected - switching to reduce-only mode",
            operation="strategy_cycle",
            stage="reduce_only",
        )
        set_reduce_only = getattr(self.context, "set_reduce_only_mode", None)
        if callable(set_reduce_only):
            try:
                set_reduce_only(True, "Configuration drift detected")
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to enable reduce-only mode after configuration drift",
                    operation="strategy_cycle",
                    stage="reduce_only",
                )
        return False

    async def _execute_trading_cycle(self, trading_state: dict[str, Any]) -> None:
        expects_context = self._process_symbol_expects_context()
        tasks = []
        for symbol in self.context.symbols:
            if expects_context:
                tasks.append(
                    self.process_symbol(
                        symbol,
                        trading_state["balances"],
                        trading_state["position_map"],
                    )
                )
            else:
                tasks.append(self.process_symbol(symbol))
        if tasks:
            await asyncio.gather(*tasks)

    def _process_symbol_expects_context(self) -> bool:
        runtime_state = self.context.runtime_state
        if runtime_state is None:
            return False
        processor = self._resolve_symbol_processor()
        bound_callable = processor.process_symbol
        dispatch = getattr(bound_callable, "__func__", bound_callable)
        needs_context = runtime_state.process_symbol_needs_context
        if needs_context is None or dispatch is not runtime_state.process_symbol_dispatch:
            requires_context = getattr(processor, "requires_context", None)
            if requires_context is None:
                process_sig = inspect.signature(dispatch)
                requires_context = len(process_sig.parameters) > 1
            needs_context = bool(requires_context)
            runtime_state.process_symbol_needs_context = needs_context
            runtime_state.process_symbol_dispatch = dispatch
        return bool(needs_context)

    @staticmethod
    def calculate_spread_bps(bid_price: Decimal, ask_price: Decimal) -> Decimal:
        try:
            mid = (bid_price + ask_price) / Decimal("2")
            if mid <= 0:
                return Decimal("0")
            return ((ask_price - bid_price) / mid) * Decimal("10000")
        except Exception:
            return Decimal("0")


__all__ = ["StrategyCoordinator"]
