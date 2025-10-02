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
from bot_v2.orchestration.lifecycle_service import LifecycleService
from bot_v2.orchestration.market_data_service import MarketDataService
from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.orchestration.runtime_coordinator import RuntimeCoordinator
from bot_v2.orchestration.service_registry import ServiceRegistry, empty_registry
from bot_v2.orchestration.session_guard import TradingSessionGuard
from bot_v2.orchestration.storage import StorageBootstrapper
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
from bot_v2.orchestration.streaming_service import StreamingService
from bot_v2.orchestration.system_monitor import SystemMonitor

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.live_execution import LiveExecutionEngine
    from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder

logger = logging.getLogger(__name__)


class PerpsBot:
    """Main Coinbase trading bot (spot by default, perps optional)."""

    def __init__(self, config: BotConfig, registry: ServiceRegistry | None = None) -> None:
        # Feature flag: use builder pattern for construction (default: true)
        use_builder = os.getenv("USE_PERPS_BOT_BUILDER", "true").lower() == "true"

        if use_builder:
            # New construction path: use PerpsBotBuilder
            from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder

            builder = PerpsBotBuilder(config)
            if registry is not None:
                builder = builder.with_registry(registry)

            built = builder.build()
            self._apply_built_state(built)
        else:
            # Legacy construction path (rollback)
            import warnings

            warnings.warn(
                "Using legacy PerpsBot construction path. "
                "This will be deprecated in a future release. "
                "See docs/architecture/PHASE_3_COMPLETE_SUMMARY.md for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )
            self._legacy_init(config, registry)

    @classmethod
    def from_builder(cls, builder: PerpsBotBuilder) -> PerpsBot:
        """Create PerpsBot instance from builder (future-proof API).

        This classmethod provides an explicit way to construct PerpsBot from a builder,
        useful for future refactors that want to avoid the __init__ shim.

        Args:
            builder: Configured PerpsBotBuilder instance

        Returns:
            Fully initialized PerpsBot instance

        Example:
            builder = PerpsBotBuilder(config).with_registry(registry)
            bot = PerpsBot.from_builder(builder)
        """
        return builder.build()

    def _apply_built_state(self, built: PerpsBot) -> None:
        """Apply state from builder-constructed instance.

        Safely copies all attributes from built instance to self, avoiding
        descriptor bypass and ensuring proper initialization.

        IMPORTANT: After copying attributes, this method rebinds all coordinator
        services to point to the real bot instance (self) instead of the temporary
        builder instance. This ensures that:
        - ShutdownHandler can stop the bot by setting bot.running = False
        - Lifecycle loops check the correct bot's running flag
        - All state changes affect the actual runtime instance

        **For Future Maintainers:**
        If you add a new service that stores a `_bot` reference in its __init__,
        you MUST add a rebinding line here, otherwise the service will operate on
        the throwaway builder instance instead of the real bot!

        Pattern:
            if hasattr(self, "your_new_service"):
                self.your_new_service._bot = self

        Args:
            built: Fully constructed PerpsBot instance from builder
        """
        # Copy all attributes from built instance
        for key, value in built.__dict__.items():
            setattr(self, key, value)

        # Rebind coordinator services to point to the real bot (self) instead of the
        # temporary builder instance. These services hold _bot references that must
        # point to the actual runtime instance so that state changes (like running=False)
        # affect the correct object.
        #
        # WARNING: When adding new coordinator services, ensure they are rebind here!
        if hasattr(self, "strategy_orchestrator"):
            self.strategy_orchestrator._bot = self
        if hasattr(self, "execution_coordinator"):
            self.execution_coordinator._bot = self
        if hasattr(self, "system_monitor"):
            self.system_monitor._bot = self
        if hasattr(self, "runtime_coordinator"):
            self.runtime_coordinator._bot = self
        if hasattr(self, "lifecycle_service"):
            self.lifecycle_service._bot = self

    def _legacy_init(self, config: BotConfig, registry: ServiceRegistry | None = None) -> None:
        """Legacy initialization path (for rollback).

        This is the original __init__ logic, preserved for backward compatibility
        when USE_PERPS_BOT_BUILDER=false.
        """
        self.bot_id = "perps_bot"
        self.start_time = datetime.now(UTC)
        self.running = False

        self._init_configuration_state(config, registry)
        self._init_runtime_state()
        self._bootstrap_storage()
        self._construct_services()
        self.runtime_coordinator.bootstrap()
        self._init_market_data_service()
        self._init_accounting_services()
        self._init_market_services()
        self._init_streaming_service()
        self._start_streaming_if_configured()

    def _init_configuration_state(
        self, config: BotConfig, registry: ServiceRegistry | None
    ) -> None:
        self.config_controller = ConfigController(config)
        self.config = self.config_controller.current
        base_registry = registry or empty_registry(self.config)
        if base_registry.config is not self.config:
            base_registry = base_registry.with_updates(config=self.config)
        self.registry = base_registry

        self._session_guard = TradingSessionGuard(
            start=self.config.trading_window_start,
            end=self.config.trading_window_end,
            trading_days=self.config.trading_days,
        )

        self.symbols = list(self.config.symbols or [])
        if not self.symbols:
            logger.warning("No symbols configured; continuing with empty symbol list")
        self._derivatives_enabled = bool(getattr(self.config, "derivatives_enabled", False))

    def _init_runtime_state(self) -> None:
        self.mark_windows: dict[str, list[Decimal]] = {s: [] for s in self.symbols}
        self.last_decisions: dict[str, Any] = {}
        self._last_positions: dict[str, dict[str, Any]] = {}
        self.order_stats = {"attempted": 0, "successful": 0, "failed": 0}
        self._order_lock: asyncio.Lock | None = None
        self._mark_lock = threading.RLock()
        self._symbol_strategies: dict[str, Any] = {}
        self.strategy: Any | None = None
        self._exec_engine: LiveExecutionEngine | AdvancedExecutionEngine | None = None
        self._streaming_service: StreamingService | None = None
        self._product_map: dict[str, Product] = {}

    def _init_market_data_service(self) -> None:
        """Initialize MarketDataService (Phase 1 refactoring)."""
        # Feature flag (default: use new MarketDataService)
        use_new_service = os.getenv("USE_NEW_MARKET_DATA_SERVICE", "true").lower() == "true"

        if use_new_service:
            self._market_data_service = MarketDataService(
                symbols=self.symbols,
                broker=self.broker,
                risk_manager=self.risk_manager,
                long_ma=self.config.long_ma,
                short_ma=self.config.short_ma,
                mark_lock=self._mark_lock,
                mark_windows=self.mark_windows,  # Share existing dict for backward compat
            )
        else:
            self._market_data_service = None

    def _init_streaming_service(self) -> None:
        """Initialize StreamingService (Phase 2 refactoring)."""
        # Feature flag (default: use new StreamingService)
        use_new_streaming = os.getenv("USE_NEW_STREAMING_SERVICE", "true").lower() == "true"

        if use_new_streaming and self._market_data_service is not None:
            self._streaming_service = StreamingService(
                symbols=self.symbols,
                broker=self.broker,
                market_data_service=self._market_data_service,
                risk_manager=self.risk_manager,
                event_store=self.event_store,
                market_monitor=self._market_monitor,
                bot_id=self.bot_id,
            )
        else:
            self._streaming_service = None

    def _bootstrap_storage(self) -> None:
        storage_ctx = StorageBootstrapper(self.config, self.registry).bootstrap()
        self.event_store = storage_ctx.event_store
        self.orders_store = storage_ctx.orders_store
        self.registry = storage_ctx.registry

    def _construct_services(self) -> None:
        self.strategy_orchestrator = StrategyOrchestrator(self)
        self.execution_coordinator = ExecutionCoordinator(self)
        self.system_monitor = SystemMonitor(self)
        self.runtime_coordinator = RuntimeCoordinator(self)
        self.lifecycle_service = LifecycleService(self)

    def _init_accounting_services(self) -> None:
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

    def _init_market_services(self) -> None:
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

    def _start_streaming_if_configured(self) -> None:
        if getattr(self.config, "perps_enable_streaming", False) and self.config.profile in {
            Profile.CANARY,
            Profile.PROD,
        }:
            try:
                if self._streaming_service is not None:
                    configured_level = getattr(self.config, "perps_stream_level", 1) or 1
                    self._streaming_service.start(level=configured_level)
                else:
                    self._start_streaming_background_legacy()
            except Exception:
                logger.exception("Failed to start streaming background worker")

    @property
    def broker(self) -> IBrokerage:
        broker = self.registry.broker
        if broker is None:
            raise RuntimeError("Broker is not configured in the service registry")
        return broker

    @broker.setter
    def broker(self, value: IBrokerage) -> None:
        self.registry = self.registry.with_updates(broker=value)

    @property
    def risk_manager(self) -> LiveRiskManager:
        risk = self.registry.risk_manager
        if risk is None:
            raise RuntimeError("Risk manager is not configured in the service registry")
        return risk

    @risk_manager.setter
    def risk_manager(self, value: LiveRiskManager) -> None:
        self.registry = self.registry.with_updates(risk_manager=value)

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
        """Run the trading bot (delegates to LifecycleService).

        Args:
            single_cycle: If True, run only one cycle and exit

        Raises:
            KeyboardInterrupt: Re-raised after cleanup
            Exception: Logged and written to health status
        """
        # Feature flag: use LifecycleService (default: true)
        use_lifecycle_service = os.getenv("USE_LIFECYCLE_SERVICE", "true").lower() == "true"

        if use_lifecycle_service:
            self.lifecycle_service.configure_background_tasks(single_cycle)
            await self.lifecycle_service.run(single_cycle)
        else:
            # Legacy run implementation (rollback path)
            await self._run_legacy(single_cycle)

    async def _run_legacy(self, single_cycle: bool = False) -> None:
        """Legacy run implementation (for rollback when USE_LIFECYCLE_SERVICE=false)."""
        logger.info("Starting Perps Bot - Profile: %s", self.config.profile.value)
        self.running = True
        background_tasks: list[asyncio.Task[Any]] = []
        try:
            if not self.config.dry_run:
                await self.runtime_coordinator.reconcile_state_on_startup()
                if not single_cycle:
                    background_tasks.append(
                        asyncio.create_task(self.execution_coordinator.run_runtime_guards())
                    )
                    background_tasks.append(
                        asyncio.create_task(self.execution_coordinator.run_order_reconciliation())
                    )
                    background_tasks.append(
                        asyncio.create_task(self.system_monitor.run_position_reconciliation())
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
        return await self.execution_coordinator._place_order(self.exec_engine, **kwargs)

    async def _place_order_inner(self, **kwargs: Any) -> Order | None:
        return await self.execution_coordinator._place_order_inner(**kwargs)

    # ------------------------------------------------------------------
    async def update_marks(self) -> None:
        """Update mark prices for all symbols (delegates to MarketDataService if enabled)."""
        if self._market_data_service is not None:
            await self._market_data_service.update_marks()
        else:
            await self._update_marks_legacy()

    async def _update_marks_legacy(self) -> None:
        """Legacy update_marks implementation (rollback path)."""
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
        """Build Product on-the-fly (no caching needed - cheap construction)."""
        base, _, quote = symbol.partition("-")
        quote = quote or os.getenv("COINBASE_DEFAULT_QUOTE", "USD").upper()
        is_perp = symbol.upper().endswith("-PERP")
        market_type = MarketType.PERPETUAL if is_perp else MarketType.SPOT
        # Provide conservative defaults; execution will re-quantize via product catalog
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
            if self._streaming_service is not None:
                self._streaming_service.stop()
            else:
                self._stop_streaming_background()
        except Exception as exc:
            logger.debug("Failed to stop WS thread cleanly: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    def apply_config_change(self, change: ConfigChange) -> None:
        logger.info("Applying configuration change diff=%s", change.diff)
        self.config = change.updated
        self.symbols = list(self.config.symbols or [])
        self._derivatives_enabled = bool(getattr(self.config, "derivatives_enabled", False))
        self.registry = self.registry.with_updates(config=self.config)
        self.execution_coordinator.reset_order_reconciler()
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
        self._init_market_services()
        self._init_streaming_service()  # Reinit streaming service after config change
        self.strategy_orchestrator.init_strategy()
        self._restart_streaming_if_needed(change.diff)

    # ------------------------------------------------------------------
    # Legacy streaming methods (rollback path when USE_NEW_STREAMING_SERVICE=false)
    # ------------------------------------------------------------------
    def _start_streaming_background_legacy(self) -> None:  # pragma: no cover - gated by env/profile
        symbols = list(self.symbols)
        if not symbols:
            return
        configured_level = getattr(self.config, "perps_stream_level", 1) or 1
        try:
            level = max(int(configured_level), 1)
        except (TypeError, ValueError):
            logger.warning("Invalid streaming level %s; defaulting to 1", configured_level)
            level = 1
        try:
            self._ws_stop = threading.Event()
        except Exception:
            self._ws_stop = None
        self._ws_thread = threading.Thread(
            target=self._run_stream_loop, args=(symbols, level), daemon=True
        )
        self._ws_thread.start()
        logger.info("Started WS streaming thread for symbols=%s level=%s", symbols, level)

    def _stop_streaming_background(self) -> None:
        if not hasattr(self, "_ws_thread"):
            return
        try:
            stop_event = getattr(self, "_ws_stop", None)
            if stop_event:
                stop_event.set()
            thread = getattr(self, "_ws_thread", None)
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        except Exception as exc:
            logger.debug("Failed to stop WS streaming thread cleanly: %s", exc, exc_info=True)
        finally:
            self._ws_thread = None
            self._ws_stop = None

    def _restart_streaming_if_needed(self, diff: dict[str, Any]) -> None:
        streaming_enabled = bool(getattr(self.config, "perps_enable_streaming", False))
        toggle_changed = "perps_enable_streaming" in diff or "perps_stream_level" in diff

        if not streaming_enabled:
            if toggle_changed:
                if self._streaming_service is not None:
                    self._streaming_service.stop()
                else:
                    self._stop_streaming_background()
            return

        # For enabled streaming ensure we refresh when toggle flips or level changes.
        if toggle_changed:
            if self._streaming_service is not None:
                self._streaming_service.stop()
            else:
                self._stop_streaming_background()
        self._start_streaming_if_configured()

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
