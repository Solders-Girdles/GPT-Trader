from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    MarketType,
    Order,
    Position,
    Product,
)
from bot_v2.logging import (
    correlation_context,
    get_orchestration_logger,
    symbol_context,
)
from bot_v2.orchestration.config_controller import ConfigChange, ConfigController
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.coordinators import (
    CoordinatorContext,
    CoordinatorRegistry,
    ExecutionCoordinator,
    RuntimeCoordinator,
    StrategyCoordinator,
    TelemetryCoordinator,
)
from bot_v2.orchestration.lifecycle_manager import LifecycleManager
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
from bot_v2.orchestration.system_monitor import SystemMonitor
from bot_v2.utilities.config import ConfigBaselinePayload
from bot_v2.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from app.container import ApplicationContainer
    from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian
    from bot_v2.orchestration.account_telemetry import AccountTelemetryService
    from bot_v2.orchestration.intx_portfolio_service import IntxPortfolioService
    from bot_v2.orchestration.live_execution import LiveExecutionEngine
    from bot_v2.orchestration.market_monitor import MarketActivityMonitor
    from bot_v2.orchestration.symbol_processor import SymbolProcessor
    from bot_v2.persistence.event_store import EventStore
    from bot_v2.persistence.orders_store import OrdersStore

logger = get_logger(__name__, component="coinbase_trader")
json_logger = get_orchestration_logger("coinbase_trader")


class _CallableSymbolProcessor:
    """Adapter allowing bare callables to masquerade as ``SymbolProcessor`` implementations."""

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
    Lifecycle management is delegated to a ``CoordinatorRegistry`` which instantiates and tracks
    dedicated coordinators for:
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
        container: ApplicationContainer | None = None,
    ) -> None:
        # Basic attributes
        self.bot_id = "coinbase_trader"
        self.start_time = datetime.now(UTC)
        self.running = False

        # Store container reference if provided
        self._container = container

        # Core dependencies
        self.config_controller = config_controller
        self.config = self.config_controller.current
        self.registry = self._align_registry_with_config(registry)
        self.event_store = event_store
        self.orders_store = orders_store
        self._session_guard = session_guard
        self.baseline_snapshot = baseline_snapshot
        self.configuration_guardian = self._resolve_configuration_guardian(
            configuration_guardian, baseline_snapshot
        )

        # Initialize all components in a single, focused method
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all bot components in dependency order."""

        # Initialize runtime state
        self._state = self._initialize_symbols_state()
        self._symbol_processor_override: _CallableSymbolProcessor | None = None

        # Initialize orchestrators and coordinators
        self.strategy_orchestrator = StrategyOrchestrator(self)
        self._setup_coordinator_stack()

        # Initialize lifecycle management
        self.lifecycle_manager = LifecycleManager(self)
        self._initialize_service_placeholders()

    @classmethod
    def from_container(
        cls,
        container: ApplicationContainer,
        *,
        session_guard: TradingSessionGuard | None = None,
        baseline_snapshot: Any | None = None,
        configuration_guardian: ConfigurationGuardian | None = None,
    ) -> PerpsBot:
        """
        Create a PerpsBot instance from an ApplicationContainer.

        This class method provides a convenient way to create a PerpsBot
        using dependencies from an ApplicationContainer. This is the
        recommended way to create PerpsBot instances when using the
        composition root pattern.

        Args:
            container: The application container with dependencies
            session_guard: Optional session guard (created from config if not provided)
            baseline_snapshot: Optional baseline snapshot (created if not provided)
            configuration_guardian: Optional configuration guardian (created if not provided)

        Returns:
            Configured PerpsBot instance
        """
        # Get dependencies from container
        config_controller = container.config_controller
        config = config_controller.current
        registry = container.create_service_registry()
        event_store = container.event_store
        orders_store = container.orders_store

        # Create session guard if not provided
        if session_guard is None:
            session_guard = TradingSessionGuard(
                start=config.trading_window_start,
                end=config.trading_window_end,
                trading_days=config.trading_days,
            )

        # Create baseline snapshot if not provided
        if baseline_snapshot is None:
            baseline_snapshot = cls.build_baseline_snapshot(
                config,
                getattr(config, "derivatives_enabled", False),
            )

        # Create configuration guardian if not provided
        if configuration_guardian is None:
            from bot_v2.monitoring.configuration_guardian import (
                ConfigurationGuardian as _ConfigurationGuardian,
            )

            configuration_guardian = _ConfigurationGuardian(baseline_snapshot)

        # Create bot instance
        bot = cls(
            config_controller=config_controller,
            registry=registry,
            event_store=event_store,
            orders_store=orders_store,
            session_guard=session_guard,
            baseline_snapshot=baseline_snapshot,
            configuration_guardian=configuration_guardian,
            container=container,
        )

        # Bootstrap the bot lifecycle
        bot.lifecycle_manager.bootstrap()

        logger.info(
            "Created PerpsBot from container",
            operation="perps_bot_from_container",
            bot_id=bot.bot_id,
            symbol_count=len(bot.symbols),
            symbols=bot.symbols or ["<none>"],
        )

        return bot

    @property
    def container(self) -> ApplicationContainer | None:
        """Get the application container if available."""
        return self._container

    def _align_registry_with_config(self, registry: ServiceRegistry) -> ServiceRegistry:
        """Ensure the registry reflects the runtime configuration."""

        if registry.config is not self.config:
            return registry.with_updates(config=self.config)
        return registry

    def _resolve_configuration_guardian(
        self,
        configuration_guardian: ConfigurationGuardian | None,
        baseline_snapshot: Any,
    ) -> ConfigurationGuardian:
        """Return the configuration guardian instance backing drift detection."""

        if configuration_guardian is not None:
            return configuration_guardian

        from bot_v2.monitoring.configuration_guardian import (
            ConfigurationGuardian as _ConfigurationGuardian,
        )

        return _ConfigurationGuardian(baseline_snapshot)

    def _initialize_symbols_state(self) -> PerpsBotRuntimeState:
        """Initialise symbol configuration and backing runtime state."""

        symbols = list(self.config.symbols or [])
        if not symbols:
            logger.warning(
                "No symbols configured; continuing with empty symbol list",
                operation="coinbase_trader_init",
                stage="symbols_missing",
            )

        self.symbols = symbols
        self._derivatives_enabled = bool(getattr(self.config, "derivatives_enabled", False))
        return PerpsBotRuntimeState(symbols)

    def _setup_coordinator_stack(self) -> None:
        """Instantiate coordinator context, registry, and wiring."""

        self._coordinator_context = CoordinatorContext(
            config=self.config,
            registry=self.registry,
            event_store=self.event_store,
            orders_store=self.orders_store,
            symbols=tuple(self.symbols),
            bot_id=self.bot_id,
            runtime_state=self._state,
            config_controller=self.config_controller,
            strategy_orchestrator=self.strategy_orchestrator,
            execution_coordinator=None,
            strategy_coordinator=None,
            session_guard=self._session_guard,
            configuration_guardian=self.configuration_guardian,
            system_monitor=None,
            set_reduce_only_mode=self.set_reduce_only_mode,
            shutdown_hook=self.shutdown,
            set_running_flag=lambda value: setattr(self, "running", value),
        )

        self._coordinator_registry = CoordinatorRegistry(self._coordinator_context)
        self._register_coordinators()

        self.system_monitor = SystemMonitor(self)
        self._coordinator_context = self._coordinator_context.with_updates(
            system_monitor=self.system_monitor,
            execution_coordinator=self.execution_coordinator,
            strategy_coordinator=self.strategy_coordinator,
        )
        self._coordinator_registry._context = self._coordinator_context  # type: ignore[attr-defined]

        for coordinator in (
            self.runtime_coordinator,
            self.execution_coordinator,
            self.strategy_coordinator,
            self.telemetry_coordinator,
        ):
            if hasattr(coordinator, "update_context"):
                coordinator.update_context(self._coordinator_context)

    def _initialize_service_placeholders(self) -> None:
        """Reset service placeholders that are populated during telemetry startup."""

        self.account_manager = None
        self.account_telemetry = None
        self.market_monitor = None
        self.intx_portfolio_service = None

    def _register_coordinators(self) -> None:
        """Register orchestrator coordinators in dependency order."""

        runtime = RuntimeCoordinator(
            self._coordinator_context,
            config_controller=self.config_controller,
            strategy_orchestrator=self.strategy_orchestrator,
            execution_coordinator=None,
            product_cache=self._state.product_map,
        )
        execution = ExecutionCoordinator(self._coordinator_context)
        strategy = StrategyCoordinator(self._coordinator_context)
        telemetry = TelemetryCoordinator(self._coordinator_context)

        self._coordinator_context = self._coordinator_context.with_updates(
            execution_coordinator=execution,
            strategy_coordinator=strategy,
        )
        self._coordinator_registry._context = self._coordinator_context  # type: ignore[attr-defined]

        for coordinator in (runtime, execution, strategy, telemetry):
            coordinator.update_context(self._coordinator_context)

        self._coordinator_registry.register(runtime)
        self._coordinator_registry.register(execution)
        self._coordinator_registry.register(strategy)
        self._coordinator_registry.register(telemetry)

        # Inject the centralized state manager into the risk manager if available
        if (
            hasattr(self.registry, "reduce_only_state_manager")
            and self.registry.reduce_only_state_manager is not None
        ):
            if hasattr(self.registry.risk_manager, "_centralized_state_manager"):
                self.registry.risk_manager._centralized_state_manager = (
                    self.registry.reduce_only_state_manager
                )

    # ------------------------------------------------------------------
    @property
    def runtime_state(self) -> PerpsBotRuntimeState:
        return self._state

    @property
    def runtime_coordinator(self) -> RuntimeCoordinator:
        coordinator = self._coordinator_registry.get("runtime")
        if coordinator is None:
            raise RuntimeError("Runtime coordinator not registered")
        if not isinstance(coordinator, RuntimeCoordinator):
            raise RuntimeError("Runtime coordinator has unexpected type")
        return cast("RuntimeCoordinator", coordinator)

    @property
    def execution_coordinator(self) -> ExecutionCoordinator:
        coordinator = self._coordinator_context.execution_coordinator
        if coordinator is None:
            coordinator = self._coordinator_registry.get("execution")
        if coordinator is None:
            raise RuntimeError("Execution coordinator not registered")
        return cast("ExecutionCoordinator", coordinator)

    @property
    def strategy_coordinator(self) -> StrategyCoordinator:
        coordinator = self._coordinator_context.strategy_coordinator
        if coordinator is None:
            coordinator = self._coordinator_registry.get("strategy")
        if coordinator is None:
            raise RuntimeError("Strategy coordinator not registered")
        return cast("StrategyCoordinator", coordinator)

    @property
    def telemetry_coordinator(self) -> TelemetryCoordinator:
        coordinator = getattr(self._coordinator_context, "telemetry_coordinator", None)
        if coordinator is None:
            coordinator = self._coordinator_registry.get("telemetry")
        if coordinator is None:
            raise RuntimeError("Telemetry coordinator not registered")
        if not isinstance(coordinator, TelemetryCoordinator):
            raise RuntimeError("Telemetry coordinator has unexpected type")
        return cast("TelemetryCoordinator", coordinator)

    @property
    def settings(self) -> RuntimeSettings:
        settings = self.registry.runtime_settings
        if settings is not None:
            return settings
        settings = load_runtime_settings()
        self.registry = self.registry.with_updates(runtime_settings=settings)
        return settings

    @property
    def mark_windows(self) -> dict[str, list[Decimal]]:
        return cast(dict[str, list[Decimal]], self._state.mark_windows)

    @property
    def last_decisions(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._state.last_decisions)

    @property
    def _last_positions(self) -> dict[str, dict[str, Any]]:
        return cast(dict[str, dict[str, Any]], self._state.last_positions)

    @property
    def order_stats(self) -> dict[str, int]:
        return cast(dict[str, int], self._state.order_stats)

    @property
    def _product_map(self) -> dict[str, Product]:
        return cast(dict[str, Product], self._state.product_map)

    @property
    def _order_lock(self) -> asyncio.Lock | None:
        return cast(asyncio.Lock | None, self._state.order_lock)

    @property
    def _mark_lock(self) -> threading.RLock:
        return cast(threading.RLock, self._state.mark_lock)

    @property
    def _symbol_strategies(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._state.symbol_strategies)

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
        return cast(bool | None, self._state.process_symbol_needs_context)

    @staticmethod
    def build_baseline_snapshot(config: BotConfig, derivatives_enabled: bool) -> Any:
        """Return a baseline snapshot for configuration drift detection."""

        from bot_v2.monitoring.configuration_guardian import (
            ConfigurationGuardian as _ConfigurationGuardian,
        )

        payload = ConfigBaselinePayload.from_config(
            config,
            derivatives_enabled=derivatives_enabled,
        )

        payload_dict = payload.to_dict()
        active_symbols = list(payload_dict.get("symbols") or [])
        broker_type = "mock" if config.mock_broker else "live"
        runtime_settings = getattr(getattr(config, "state", None), "runtime_settings", None)

        return _ConfigurationGuardian.create_baseline_snapshot(
            config_dict=payload_dict,
            active_symbols=active_symbols,
            positions=[],
            account_equity=None,
            profile=config.profile,
            broker_type=broker_type,
            settings=runtime_settings,
        )

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
        return cast("LiveExecutionEngine | AdvancedExecutionEngine", engine)

    @property
    def account_manager(self) -> CoinbaseAccountManager | None:
        return self._state.account_manager

    @account_manager.setter
    def account_manager(self, value: CoinbaseAccountManager | None) -> None:
        self._state.account_manager = value

    @property
    def account_telemetry(self) -> AccountTelemetryService | None:
        return self._state.account_telemetry

    @account_telemetry.setter
    def account_telemetry(self, value: AccountTelemetryService | None) -> None:
        self._state.account_telemetry = value

    @property
    def market_monitor(self) -> MarketActivityMonitor | None:
        monitor = self._state.market_monitor
        if monitor is None:
            return self.__dict__.get("_market_monitor")
        return monitor

    @market_monitor.setter
    def market_monitor(self, value: MarketActivityMonitor | None) -> None:
        self._state.market_monitor = value
        if value is None:
            self.__dict__.pop("_market_monitor", None)
        else:
            self.__dict__["_market_monitor"] = value

    @property
    def intx_portfolio_service(self) -> IntxPortfolioService | None:
        return self._state.intx_portfolio_service

    @intx_portfolio_service.setter
    def intx_portfolio_service(self, value: IntxPortfolioService | None) -> None:
        self._state.intx_portfolio_service = value

    # ------------------------------------------------------------------
    async def run(self, single_cycle: bool = False) -> None:
        # Create correlation context for the entire bot run
        with correlation_context(operation="bot_run", bot_id=self.bot_id):
            await self.lifecycle_manager.run(single_cycle)

    async def run_cycle(self) -> None:
        # Create correlation context for each trading cycle
        with correlation_context(operation="trading_cycle", bot_id=self.bot_id):
            await self.strategy_coordinator.run_cycle()

    async def _fetch_current_state(self) -> dict[str, Any]:
        state = await self.strategy_coordinator._fetch_current_state()
        return cast(dict[str, Any], state)

    async def _validate_configuration_and_handle_drift(self, current_state: dict[str, Any]) -> bool:
        result = await self.strategy_coordinator._validate_configuration_and_handle_drift(
            current_state
        )
        return bool(result)

    async def _execute_trading_cycle(self, trading_state: dict[str, Any]) -> None:
        await self.strategy_coordinator._execute_trading_cycle(trading_state)

    @property
    def symbol_processor(self) -> SymbolProcessor:
        return self.strategy_coordinator.symbol_processor

    def set_symbol_processor(self, processor: SymbolProcessor | None) -> None:
        if isinstance(processor, _CallableSymbolProcessor):
            self._symbol_processor_override = processor
        else:
            self._symbol_processor_override = None
        self.strategy_coordinator.set_symbol_processor(processor)

    async def process_symbol(
        self,
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

    def _ensure_order_lock(self) -> asyncio.Lock:
        lock = self.strategy_coordinator.ensure_order_lock()
        return cast(asyncio.Lock, lock)

    async def _place_order(self, **kwargs: Any) -> Order | None:
        return await self.strategy_coordinator.place_order(**kwargs)

    async def _place_order_inner(self, **kwargs: Any) -> Order | None:
        return await self.strategy_coordinator.place_order_inner(**kwargs)

    # ------------------------------------------------------------------
    async def update_marks(self) -> None:
        await self.strategy_coordinator.update_marks()

    def get_product(self, symbol: str) -> Product:
        if symbol in self._product_map:
            return self._product_map[symbol]
        base, _, quote = symbol.partition("-")
        quote = quote or self.settings.coinbase_default_quote
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
        return bool(self.runtime_coordinator.is_reduce_only_mode())

    def set_reduce_only_mode(self, enabled: bool, reason: str) -> None:
        self.runtime_coordinator.set_reduce_only_mode(enabled, reason)

    def write_health_status(self, ok: bool, message: str = "", error: str = "") -> None:
        self.system_monitor.write_health_status(ok=ok, message=message, error=error)

    async def shutdown(self) -> None:
        # Create correlation context for shutdown
        with correlation_context(operation="bot_shutdown", bot_id=self.bot_id):
            await self.lifecycle_manager.shutdown()

    # ------------------------------------------------------------------
    def apply_config_change(self, change: ConfigChange) -> None:
        logger.info(
            "Applying configuration change",
            operation="config_change",
            stage="apply",
            diff=change.diff,
            changed_fields=sorted(change.diff.keys()),
        )
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
        self.telemetry_coordinator.init_market_services()
        self.strategy_orchestrator.init_strategy()
        self._restart_streaming_if_needed(change.diff)

        # Refresh configuration baseline to reflect the new runtime config
        new_baseline = self.build_baseline_snapshot(self.config, self._derivatives_enabled)
        self.baseline_snapshot = new_baseline
        if self.configuration_guardian is not None:
            self.configuration_guardian.reset_baseline(new_baseline)

    # ------------------------------------------------------------------
    def _start_streaming_background(self) -> None:  # pragma: no cover - gated by env/profile
        self.telemetry_coordinator.start_streaming_background()

    def _stop_streaming_background(self) -> None:
        self.telemetry_coordinator.stop_streaming_background()

    def _restart_streaming_if_needed(self, diff: dict[str, Any]) -> None:
        self.telemetry_coordinator.restart_streaming_if_needed(diff)

    def _run_stream_loop(self, symbols: list[str], level: int) -> None:
        self.telemetry_coordinator._run_stream_loop(symbols, level, stop_signal=None)

    # ------------------------------------------------------------------
    def _update_mark_window(self, symbol: str, mark: Decimal) -> None:
        self.strategy_coordinator.update_mark_window(symbol, mark)

    @staticmethod
    def _calculate_spread_bps(bid_price: Decimal, ask_price: Decimal) -> Decimal:
        return cast(Decimal, StrategyCoordinator.calculate_spread_bps(bid_price, ask_price))

    # ------------------------------------------------------------------
    async def _run_account_telemetry(self, interval_seconds: int = 300) -> None:
        await self.telemetry_coordinator.run_account_telemetry(interval_seconds)

    # ------------------------------------------------------------------
    def _wrap_symbol_processor(self, handler: Any) -> _CallableSymbolProcessor:
        """Convert a callable into a ``SymbolProcessor`` compatible adapter."""

        if not callable(handler):
            raise TypeError("process_symbol handler must be callable")

        try:
            parameters = inspect.signature(handler).parameters
        except (TypeError, ValueError):
            parameters = inspect.Signature(parameters=()).parameters
        requires_context = len(parameters) > 1
        return _CallableSymbolProcessor(handler, requires_context=requires_context)

    def _install_symbol_processor_override(self, handler: Any) -> None:
        """Install or remove a legacy ``process_symbol`` override."""

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


# Backwards-compatibility alias for the spot-first runtime name.
CoinbaseTrader = PerpsBot
