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
from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian
from bot_v2.monitoring.system import get_logger as _get_plog
from bot_v2.orchestration.account_telemetry import AccountTelemetryService
from bot_v2.orchestration.config_controller import ConfigChange, ConfigController
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.execution_coordinator import ExecutionCoordinator
from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.runtime_coordinator import RuntimeCoordinator
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
from bot_v2.orchestration.system_monitor import SystemMonitor
from bot_v2.utilities import emit_metric, utc_now

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.live_execution import LiveExecutionEngine
    from bot_v2.persistence.event_store import EventStore
    from bot_v2.persistence.orders_store import OrdersStore

logger = logging.getLogger(__name__)


class PerpsBot:
    """Main Coinbase trading bot implementation.

    This module contains the core PerpsBot class that orchestrates trading activities
    including market data collection, strategy execution, risk management, and
    position monitoring. The bot supports both spot and perpetual trading with
    comprehensive error handling, configuration validation, and runtime monitoring.

    Key Features:
    - Multi-symbol trading with configurable strategies
    - Real-time market data streaming and processing
    - Risk management with position sizing and loss limits
    - Configuration drift detection and validation
    - Comprehensive logging and monitoring
    - Graceful shutdown and error recovery
    - Support for both dry-run and live trading modes

    Architecture:
    The bot follows a modular architecture with separate coordinators for:
    - Execution: Order placement and management
    - Strategy: Trading signal generation and processing
    - Monitoring: System health and performance tracking
    - Configuration: Dynamic config updates and validation
    - Runtime: State management and recovery

    Usage:
        config = BotConfig.from_profile(Profile.DEV.value)
        from bot_v2.orchestration.perps_bot_builder import create_perps_bot
        bot = create_perps_bot(config)
        await bot.run(single_cycle=False)  # Run continuously
    """

    def __init__(
        self,
        *,
        config_controller: ConfigController,
        registry: ServiceRegistry,
        event_store: EventStore,
        orders_store: OrdersStore,
        session_guard: TradingSessionGuard,
        baseline_snapshot: Any,
        configuration_guardian: ConfigurationGuardian | None = None,
    ) -> None:
        self.bot_id = "perps_bot"
        self.start_time = datetime.now(UTC)
        self.running = False

        self.config_controller = config_controller
        self.config = self.config_controller.current

        if registry.config is not self.config:
            registry = registry.with_updates(config=self.config)
        self.registry = registry

        self.event_store = event_store
        self.orders_store = orders_store
        self._session_guard = session_guard
        self.baseline_snapshot = baseline_snapshot
        self.configuration_guardian = configuration_guardian or ConfigurationGuardian(
            self.baseline_snapshot
        )

        self.symbols = list(self.config.symbols or [])
        if not self.symbols:
            logger.warning("No symbols configured; continuing with empty symbol list")
        self._derivatives_enabled = bool(getattr(self.config, "derivatives_enabled", False))

        self._state = PerpsBotRuntimeState(self.symbols)

    # ------------------------------------------------------------------
    @property
    def runtime_state(self) -> PerpsBotRuntimeState:
        return self._state

    @property
    def mark_windows(self) -> dict[str, list[Decimal]]:
        return self._state.mark_windows

    @property
    def last_decisions(self) -> dict[str, Any]:
        return self._state.last_decisions

    @property
    def _last_positions(self) -> dict[str, dict[str, Any]]:
        return self._state.last_positions

    @property
    def order_stats(self) -> dict[str, int]:
        return self._state.order_stats

    @property
    def _product_map(self) -> dict[str, Product]:
        return self._state.product_map

    @property
    def _order_lock(self) -> asyncio.Lock | None:
        return self._state.order_lock

    @property
    def _mark_lock(self) -> threading.RLock:
        return self._state.mark_lock

    @property
    def _symbol_strategies(self) -> dict[str, Any]:
        return self._state.symbol_strategies

    @property
    def strategy(self) -> Any | None:
        return self._state.strategy

    @property
    def _exec_engine(self) -> Any | None:
        return self._state.exec_engine

    @property
    def _process_symbol_dispatch(self) -> Any | None:
        return self._state.process_symbol_dispatch

    @property
    def _process_symbol_needs_context(self) -> bool | None:
        return self._state.process_symbol_needs_context

    @staticmethod
    def build_baseline_snapshot(config: BotConfig, derivatives_enabled: bool) -> Any:
        """Return a baseline snapshot for configuration drift detection."""

        config_dict = {
            "profile": config.profile,
            "dry_run": config.dry_run,
            "symbols": list(config.symbols) if config.symbols else [],
            "derivatives_enabled": derivatives_enabled,
            "update_interval": config.update_interval,
            "short_ma": config.short_ma,
            "long_ma": config.long_ma,
            "target_leverage": config.target_leverage,
            "trailing_stop_pct": config.trailing_stop_pct,
            "enable_shorts": config.enable_shorts,
            "max_position_size": config.max_position_size,
            "max_leverage": config.max_leverage,
            "reduce_only_mode": config.reduce_only_mode,
            "mock_broker": config.mock_broker,
            "mock_fills": config.mock_fills,
            "enable_order_preview": config.enable_order_preview,
            "account_telemetry_interval": config.account_telemetry_interval,
            "trading_window_start": config.trading_window_start,
            "trading_window_end": config.trading_window_end,
            "trading_days": config.trading_days,
            "daily_loss_limit": config.daily_loss_limit,
            "time_in_force": config.time_in_force,
            "perps_enable_streaming": getattr(config, "perps_enable_streaming", False),
            "perps_stream_level": getattr(config, "perps_stream_level", 1),
            "perps_paper_trading": getattr(config, "perps_paper_trading", False),
            "perps_force_mock": getattr(config, "perps_force_mock", False),
            "perps_position_fraction": getattr(config, "perps_position_fraction", None),
            "perps_skip_startup_reconcile": getattr(config, "perps_skip_startup_reconcile", False),
        }

        active_symbols = list(config.symbols) if config.symbols else []
        broker_type = "mock" if config.mock_broker else "live"

        return ConfigurationGuardian.create_baseline_snapshot(
            config_dict=config_dict,
            active_symbols=active_symbols,
            positions=[],
            account_equity=None,
            profile=config.profile,
            broker_type=broker_type,
        )

    def _construct_services(self) -> None:
        self.strategy_orchestrator = StrategyOrchestrator(self)
        self.execution_coordinator = ExecutionCoordinator(self)
        self.system_monitor = SystemMonitor(self)
        self.runtime_coordinator = RuntimeCoordinator(self)

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
                self._start_streaming_background()
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
        engine = self._state.exec_engine
        if engine is None:
            raise RuntimeError("Execution engine not initialized")
        return engine

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
        """Execute a single trading cycle with validation and trading logic."""
        logger.debug("Running update cycle")

        # Fetch current state for validation
        current_state = await self._fetch_current_state()

        # Validate configuration and handle any drift
        if not await self._validate_configuration_and_handle_drift(current_state):
            return

        # Update market data
        await self.update_marks()

        # Check if we should trade
        if not self._session_guard.should_trade():
            logger.info("Outside trading window; skipping trading actions this cycle")
            await self.system_monitor.log_status()
            return

        # Refresh state for trading
        trading_state = await self._fetch_current_state()

        # Execute trading logic for all symbols
        await self._execute_trading_cycle(trading_state)
        await self.system_monitor.log_status()

    async def _fetch_current_state(self) -> dict[str, Any]:
        """Fetch current broker state (balances, positions, equity)."""
        balances: Sequence[Balance] = []
        positions: Sequence[Position] = []
        account_equity: Decimal | None = None

        try:
            balances = await asyncio.to_thread(self.broker.list_balances)
        except Exception as exc:
            logger.warning("Unable to fetch balances: %s", exc)

        try:
            positions = await asyncio.to_thread(self.broker.list_positions)
        except Exception as exc:
            logger.warning("Unable to fetch positions: %s", exc)

        try:
            account_info = await asyncio.to_thread(self.broker.get_account_info)
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
        """Validate configuration and handle any detected drift.

        Returns:
            bool: True if trading should continue, False if cycle should be aborted
        """
        current_config_dict = {
            "profile": self.config.profile,
            "symbols": list(self.config.symbols) if self.config.symbols else [],
            "max_leverage": self.config.max_leverage,
            "max_position_size": self.config.max_position_size,
            "mock_broker": self.config.mock_broker,
        }

        validation_result = self.configuration_guardian.pre_cycle_check(
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

        # Check error severity and handle appropriately
        has_critical_errors = any(
            "critical" in error.lower() or "emergency_shutdown" in error.lower()
            for error in validation_result.errors
        )

        if has_critical_errors:
            logger.critical(
                "Critical configuration violations detected - initiating emergency shutdown"
            )
            self.running = False
            await self.shutdown()
            return False
        else:
            logger.warning(
                "High-severity configuration violations detected - switching to reduce-only mode"
            )
            self.set_reduce_only_mode(True, "Configuration drift detected")
            return False

    async def _execute_trading_cycle(self, trading_state: dict[str, Any]) -> None:
        """Execute trading logic for all configured symbols."""
        expects_context = self._process_symbol_expects_context()

        tasks = []
        for symbol in self.symbols:
            if expects_context:
                tasks.append(
                    self.process_symbol(
                        symbol, trading_state["balances"], trading_state["position_map"]
                    )
                )
            else:
                tasks.append(self.process_symbol(symbol))

        await asyncio.gather(*tasks)

    async def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> None:
        await self.strategy_orchestrator.process_symbol(symbol, balances, position_map)

    def _process_symbol_expects_context(self) -> bool:
        current_dispatch = getattr(self.process_symbol, "__func__", self.process_symbol)
        needs_context = self._state.process_symbol_needs_context
        if needs_context is None or current_dispatch is not self._state.process_symbol_dispatch:
            process_sig = inspect.signature(self.process_symbol)
            needs_context = len(process_sig.parameters) > 1
            self._state.process_symbol_needs_context = needs_context
            self._state.process_symbol_dispatch = current_dispatch
        return needs_context

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
        symbols = tuple(self.symbols)
        if not symbols:
            return

        quotes = await asyncio.gather(
            *(asyncio.to_thread(self.broker.get_quote, symbol) for symbol in symbols),
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
                self._update_mark_window(symbol, mark)
                try:
                    self.risk_manager.last_mark_update[symbol] = (
                        ts if isinstance(ts, datetime) else utc_now()
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
        mark_windows = self._state.mark_windows
        for symbol in self.symbols:
            mark_windows.setdefault(symbol, [])
        for symbol in list(mark_windows.keys()):
            if symbol not in self.symbols:
                del mark_windows[symbol]
        self._init_market_services()
        self.strategy_orchestrator.init_strategy()
        self._restart_streaming_if_needed(change.diff)

    # ------------------------------------------------------------------
    def _start_streaming_background(self) -> None:  # pragma: no cover - gated by env/profile
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
                self._stop_streaming_background()
            return

        # For enabled streaming ensure we refresh when toggle flips or level changes.
        if toggle_changed:
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
                    self.risk_manager.last_mark_update[sym] = utc_now()
                    emit_metric(
                        self.event_store,
                        self.bot_id,
                        {"event_type": "ws_mark_update", "symbol": sym, "mark": str(mark)},
                    )
                except Exception:
                    logger.exception("WS mark update bookkeeping failed for %s", sym)
        except Exception as exc:
            emit_metric(
                self.event_store,
                self.bot_id,
                {"event_type": "ws_stream_error", "message": str(exc)},
            )
        finally:
            emit_metric(
                self.event_store,
                self.bot_id,
                {"event_type": "ws_stream_exit"},
            )

    # ------------------------------------------------------------------
    def _update_mark_window(self, symbol: str, mark: Decimal) -> None:
        with self._state.mark_lock:
            window = self._state.mark_windows.setdefault(symbol, [])
            window.append(mark)
            max_size = max(self.config.short_ma, self.config.long_ma) + 5
            if len(window) > max_size:
                self._state.mark_windows[symbol] = window[-max_size:]

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
