from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    MarketType,
    Order,
    Position,
    Product,
)
from bot_v2.monitoring.system import get_logger as _get_plog
from bot_v2.orchestration.account_telemetry import AccountTelemetryService
from bot_v2.orchestration.config_controller import ConfigChange, ConfigController
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.execution_coordinator import ExecutionCoordinator
from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.orchestration.runtime_coordinator import RuntimeCoordinator
from bot_v2.orchestration.service_registry import ServiceRegistry, empty_registry
from bot_v2.orchestration.session_guard import TradingSessionGuard
from bot_v2.orchestration.storage import StorageBootstrapper
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
from bot_v2.orchestration.symbols import (
    derivatives_enabled as _resolve_derivatives_enabled,
)
from bot_v2.orchestration.symbols import (
    normalize_symbols,
)
from bot_v2.orchestration.system_monitor import SystemMonitor

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.live_execution import LiveExecutionEngine

logger = logging.getLogger(__name__)


class PerpsBot:
    """Main Coinbase trading bot (spot by default, perps optional)."""

    def __init__(self, config: BotConfig, registry: ServiceRegistry | None = None) -> None:
        self.config_controller = ConfigController(config)
        self.config = self.config_controller.current
        self.registry: ServiceRegistry = registry or empty_registry(self.config)
        if self.registry.config is not self.config:
            self.registry = self.registry.with_updates(config=self.config)

        self.bot_id = "perps_bot"
        self.start_time = datetime.now(UTC)
        self.running = False

        self._session_guard = TradingSessionGuard(
            start=self.config.trading_window_start,
            end=self.config.trading_window_end,
            trading_days=self.config.trading_days,
        )

        self._derivatives_enabled = False
        try:
            normalized_symbols, allow_derivatives = normalize_symbols(
                self.config.profile, list(self.config.symbols or [])
            )
            self.symbols = list(normalized_symbols)
            self._derivatives_enabled = allow_derivatives
        except Exception as exc:
            logger.warning("Failed to normalize symbol list: %s", exc, exc_info=True)
            self._derivatives_enabled = _resolve_derivatives_enabled(self.config.profile)
            self.symbols = list(self.config.symbols or [])
        if not self.symbols:
            self.symbols = list(self.config.symbols or [])
        self.config.symbols = tuple(self.symbols)
        self.config.derivatives_enabled = self._derivatives_enabled  # type: ignore[attr-defined]

        storage_ctx = StorageBootstrapper(self.config, self.registry).bootstrap()
        self.event_store = storage_ctx.event_store
        self.orders_store = storage_ctx.orders_store
        self.registry = storage_ctx.registry

        self._product_map: dict[str, Product] = {}
        self.mark_windows: dict[str, list[Decimal]] = {s: [] for s in self.symbols}
        self.last_decisions: dict[str, Any] = {}
        self._last_positions: dict[str, dict[str, Any]] = {}
        self.order_stats = {"attempted": 0, "successful": 0, "failed": 0}
        self._order_lock: asyncio.Lock | None = None
        self._mark_lock = threading.RLock()
        self._symbol_strategies: dict[str, Any] = {}
        self.strategy: Any | None = None
        self._exec_engine: LiveExecutionEngine | AdvancedExecutionEngine | None = None

        self.strategy_orchestrator = StrategyOrchestrator(self)
        self.execution_coordinator = ExecutionCoordinator(self)
        self.system_monitor = SystemMonitor(self)
        self.runtime_coordinator = RuntimeCoordinator(self)

        self.runtime_coordinator.bootstrap()

        self.account_manager = CoinbaseAccountManager(self.broker, event_store=self.event_store)
        self.account_telemetry = AccountTelemetryService(
            broker=self.broker,
            account_manager=self.account_manager,
            event_store=self.event_store,
            bot_id=self.bot_id,
            profile=self.config.profile.value,
        )
        if not self.account_telemetry.supports_snapshots():
            logger.info("Account snapshot telemetry disabled; broker lacks required endpoints")
        self.system_monitor.attach_account_telemetry(self.account_telemetry)

        def _log_market_heartbeat(**payload: Any) -> None:
            try:
                _get_plog().log_market_heartbeat(**payload)
            except Exception as exc:
                logger.debug(
                    "Failed to record market heartbeat for %s: %s",
                    payload.get("symbol") or payload.get("source"),
                    exc,
                    exc_info=True,
                )

        self._market_monitor = MarketActivityMonitor(
            tuple(self.symbols),
            heartbeat_logger=_log_market_heartbeat,
        )

        enable_stream = os.getenv("PERPS_ENABLE_STREAMING", "").lower() in {"1", "true", "yes"}
        if enable_stream and self.config.profile in {Profile.CANARY, Profile.PROD}:
            try:
                self._start_streaming_background()
            except Exception:
                logger.exception("Failed to start streaming background worker")

    @property
    def broker(self) -> IBrokerage:
        broker = self.registry.broker
        if broker is None:
            raise RuntimeError("Broker is not configured in the service registry")
        return broker

    @property
    def risk_manager(self) -> LiveRiskManager:
        risk = self.registry.risk_manager
        if risk is None:
            raise RuntimeError("Risk manager is not configured in the service registry")
        return risk

    @property
    def exec_engine(self) -> LiveExecutionEngine | AdvancedExecutionEngine:
        if self._exec_engine is None:
            raise RuntimeError("Execution engine not initialized")
        return self._exec_engine

    @exec_engine.setter
    def exec_engine(self, engine: LiveExecutionEngine | AdvancedExecutionEngine) -> None:
        self._exec_engine = engine

    # ------------------------------------------------------------------
    async def run(self, single_cycle: bool = False) -> None:
        logger.info("Starting Perps Bot - Profile: %s", self.config.profile.value)
        self.running = True
        background_tasks: list[asyncio.Task[Any]] = []
        try:
            if not self.config.dry_run:
                await self.runtime_coordinator.reconcile_state_on_startup()
                if not single_cycle:
                    background_tasks.append(
                        asyncio.create_task(self.execution_coordinator._run_runtime_guards())
                    )
                    background_tasks.append(
                        asyncio.create_task(self.execution_coordinator._run_order_reconciliation())
                    )
                    background_tasks.append(
                        asyncio.create_task(self.system_monitor._run_position_reconciliation())
                    )
                    if self.account_telemetry.supports_snapshots():
                        background_tasks.append(
                            asyncio.create_task(
                                self._run_account_telemetry(self.config.account_telemetry_interval)
                            )
                        )
            else:
                logger.info("Dry-run: skipping startup reconciliation and background guard loops")

            await self.run_cycle()
            self.system_monitor.write_health_status(ok=True)
            self.system_monitor.check_config_updates()

            if not single_cycle and not self.config.dry_run:
                while self.running:
                    await asyncio.sleep(self.config.update_interval)
                    if not self.running:
                        break
                    await self.run_cycle()
                    self.system_monitor.write_health_status(ok=True)
                    self.system_monitor.check_config_updates()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as exc:
            logger.error("Bot error: %s", exc, exc_info=True)
            self.system_monitor.write_health_status(ok=False, error=str(exc))
        finally:
            self.running = False
            for task in background_tasks:
                if not task.done():
                    task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            await self.shutdown()

    async def run_cycle(self) -> None:
        logger.debug("Running update cycle")
        await self.update_marks()
        if not self._session_guard.should_trade():
            logger.info("Outside trading window; skipping trading actions this cycle")
            await self.system_monitor.log_status()
            return

        balances: Sequence[Balance] = []
        positions: Sequence[Position] = []
        position_map: dict[str, Position] = {}

        try:
            balances = await asyncio.to_thread(self.broker.list_balances)
        except Exception as exc:
            logger.warning("Unable to fetch balances for trading cycle: %s", exc)

        try:
            positions = await asyncio.to_thread(self.broker.list_positions)
        except Exception as exc:
            logger.warning("Unable to fetch positions for trading cycle: %s", exc)

        if positions:
            position_map = {p.symbol: p for p in positions if hasattr(p, "symbol")}

        process_sig = inspect.signature(self.process_symbol)
        expects_context = len(process_sig.parameters) > 1

        tasks = []
        for symbol in self.symbols:
            if expects_context:
                tasks.append(self.process_symbol(symbol, balances, position_map))
            else:
                tasks.append(self.process_symbol(symbol))
        await asyncio.gather(*tasks)
        await self.system_monitor.log_status()

    async def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> None:
        await self.strategy_orchestrator.process_symbol(symbol, balances, position_map)

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        await self.execution_coordinator.execute_decision(
            symbol, decision, mark, product, position_state
        )

    def _ensure_order_lock(self) -> asyncio.Lock:
        return self.execution_coordinator._ensure_order_lock()

    async def _place_order(self, **kwargs: Any) -> Order | None:
        return await self.execution_coordinator._place_order(**kwargs)

    async def _place_order_inner(self, **kwargs: Any) -> Order | None:
        return await self.execution_coordinator._place_order_inner(**kwargs)

    # ------------------------------------------------------------------
    async def update_marks(self) -> None:
        for symbol in self.symbols:
            try:
                quote = await asyncio.to_thread(self.broker.get_quote, symbol)
                if quote is None:
                    raise RuntimeError(f"No quote for {symbol}")
                last_price = getattr(quote, "last", getattr(quote, "last_price", None))
                if last_price is None:
                    raise RuntimeError(f"Quote missing price for {symbol}")
                mark = Decimal(str(last_price))
                if mark <= 0:
                    raise RuntimeError(f"Invalid mark price: {mark} for {symbol}")
                ts = getattr(quote, "ts", datetime.now(UTC))
                self._update_mark_window(symbol, mark)
                try:
                    self.risk_manager.last_mark_update[symbol] = (
                        ts if isinstance(ts, datetime) else datetime.utcnow()
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to update mark timestamp for %s: %s", symbol, exc, exc_info=True
                    )
            except Exception as exc:
                logger.error("Error updating mark for %s: %s", symbol, exc)

    def get_product(self, symbol: str) -> Product:
        if symbol in self._product_map:
            return self._product_map[symbol]
        base, _, quote = symbol.partition("-")
        quote = quote or os.getenv("COINBASE_DEFAULT_QUOTE", "USD").upper()
        market_type = MarketType.PERPETUAL if symbol.upper().endswith("-PERP") else MarketType.SPOT
        # Provide conservative defaults; execution will re-quantize via product catalog once populated
        return Product(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            market_type=market_type,
            step_size=Decimal("0.00000001"),
            min_size=Decimal("0.00000001"),
            price_increment=Decimal("0.01"),
            min_notional=Decimal("10"),
        )

    # ------------------------------------------------------------------
    def is_reduce_only_mode(self) -> bool:
        return self.runtime_coordinator.is_reduce_only_mode()

    def set_reduce_only_mode(self, enabled: bool, reason: str) -> None:
        self.runtime_coordinator.set_reduce_only_mode(enabled, reason)

    def write_health_status(self, ok: bool, message: str = "", error: str = "") -> None:
        self.system_monitor.write_health_status(ok=ok, message=message, error=error)

    async def shutdown(self) -> None:
        logger.info("Shutting down bot...")
        self.running = False
        try:
            if hasattr(self, "_ws_stop") and self._ws_stop:
                self._ws_stop.set()
            if hasattr(self, "_ws_thread") and self._ws_thread and self._ws_thread.is_alive():
                self._ws_thread.join(timeout=2.0)
        except Exception as exc:
            logger.debug("Failed to stop WS thread cleanly: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    def apply_config_change(self, change: ConfigChange) -> None:
        logger.info("Applying configuration change diff=%s", change.diff)
        self.config = change.updated
        self.registry = self.registry.with_updates(config=self.config)
        self.config_controller.sync_with_risk_manager(self.risk_manager)
        self._session_guard = TradingSessionGuard(
            start=self.config.trading_window_start,
            end=self.config.trading_window_end,
            trading_days=self.config.trading_days,
        )
        for symbol in self.symbols:
            self.mark_windows.setdefault(symbol, [])
        for symbol in list(self.mark_windows.keys()):
            if symbol not in self.symbols:
                del self.mark_windows[symbol]
        self.strategy_orchestrator.init_strategy()

    # ------------------------------------------------------------------
    def _start_streaming_background(self) -> None:  # pragma: no cover - gated by env/profile
        symbols = list(self.symbols)
        if not symbols:
            return
        level = int(os.getenv("PERPS_STREAM_LEVEL", "1") or "1")
        try:
            self._ws_stop = threading.Event()
        except Exception:
            self._ws_stop = None
        self._ws_thread = threading.Thread(
            target=self._run_stream_loop, args=(symbols, level), daemon=True
        )
        self._ws_thread.start()
        logger.info("Started WS streaming thread for symbols=%s level=%s", symbols, level)

    def _run_stream_loop(self, symbols: list[str], level: int) -> None:
        try:
            stream = None
            try:
                stream = self.broker.stream_orderbook(symbols, level=level)
            except Exception as exc:
                logger.warning("Orderbook stream unavailable, falling back to trades: %s", exc)
                try:
                    stream = self.broker.stream_trades(symbols)
                except Exception as trade_exc:
                    logger.error("Failed to start streaming trades: %s", trade_exc)
                    return

            for msg in stream or []:
                if hasattr(self, "_ws_stop") and self._ws_stop and self._ws_stop.is_set():
                    break
                if not isinstance(msg, dict):
                    continue
                sym = str(msg.get("product_id") or msg.get("symbol") or "")
                if not sym:
                    continue
                mark = None
                bid = msg.get("best_bid") or msg.get("bid")
                ask = msg.get("best_ask") or msg.get("ask")
                if bid is not None and ask is not None:
                    try:
                        mark = (Decimal(str(bid)) + Decimal(str(ask))) / Decimal("2")
                    except Exception:
                        mark = None
                if mark is None:
                    raw_mark = msg.get("last") or msg.get("price")
                    if raw_mark is None:
                        continue
                    mark = Decimal(str(raw_mark))
                if mark <= 0:
                    continue

                self._update_mark_window(sym, mark)
                try:
                    self._market_monitor.record_update(sym)
                    self.risk_manager.last_mark_update[sym] = datetime.utcnow()
                    self.event_store.append_metric(
                        self.bot_id,
                        {"event_type": "ws_mark_update", "symbol": sym, "mark": str(mark)},
                    )
                except Exception:
                    logger.exception("WS mark update bookkeeping failed for %s", sym)
        except Exception as exc:
            try:
                self.event_store.append_metric(
                    self.bot_id,
                    {"event_type": "ws_stream_error", "message": str(exc)},
                )
            except Exception:
                logger.exception("Failed to record WS stream error metric")
        finally:
            try:
                self.event_store.append_metric(self.bot_id, {"event_type": "ws_stream_exit"})
            except Exception:
                logger.exception("Failed to record WS stream exit metric")

    # ------------------------------------------------------------------
    def _update_mark_window(self, symbol: str, mark: Decimal) -> None:
        with self._mark_lock:
            if symbol not in self.mark_windows:
                self.mark_windows[symbol] = []
            self.mark_windows[symbol].append(mark)
            max_size = max(self.config.short_ma, self.config.long_ma) + 5
            if len(self.mark_windows[symbol]) > max_size:
                self.mark_windows[symbol] = self.mark_windows[symbol][-max_size:]

    @staticmethod
    def _calculate_spread_bps(bid_price: Decimal, ask_price: Decimal) -> Decimal:
        try:
            mid = (bid_price + ask_price) / Decimal("2")
            if mid <= 0:
                return Decimal("0")
            return ((ask_price - bid_price) / mid) * Decimal("10000")
        except Exception:
            return Decimal("0")

    # ------------------------------------------------------------------
    async def _run_account_telemetry(self, interval_seconds: int = 300) -> None:
        if not self.account_telemetry.supports_snapshots():
            return
        await self.account_telemetry.run(interval_seconds)
