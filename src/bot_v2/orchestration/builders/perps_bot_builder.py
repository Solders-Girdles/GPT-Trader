"""Builder pattern for PerpsBot construction.

Separates object construction from runtime behavior by extracting initialization logic
from :class:`PerpsBot` into discrete, composable builder methods. The builder now
represents the canonical construction path — the ``USE_PERPS_BOT_BUILDER`` flag has
been retired in favor of always using this pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.features.market_data import MarketActivityMonitor, MarketDataService
from bot_v2.monitoring.metrics_server import MetricsServer
from bot_v2.monitoring.system import get_logger as _get_plog
from bot_v2.monitoring.telemetry import AccountTelemetryService
from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.execution_coordinator import ExecutionCoordinator
from bot_v2.orchestration.guardrails import GuardRailManager
from bot_v2.orchestration.runtime_coordinator import RuntimeCoordinator
from bot_v2.orchestration.service_rebinding import rebind_bot_services
from bot_v2.orchestration.service_registry import ServiceRegistry, empty_registry
from bot_v2.orchestration.session_guard import TradingSessionGuard
from bot_v2.orchestration.storage import StorageBootstrapper
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
from bot_v2.orchestration.streaming_service import StreamingService
from bot_v2.orchestration.system_monitor import SystemMonitor

if TYPE_CHECKING:  # pragma: no cover
    from datetime import datetime
    from decimal import Decimal

    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
    from bot_v2.orchestration.live_execution import LiveExecutionEngine

logger = logging.getLogger(__name__)


class PerpsBotBuilder:
    """Builder for PerpsBot instances.

    Provides a fluent API for constructing PerpsBot instances with explicit dependency injection
    and separated construction phases. Defers heavy initialization work until build() is called.

    Usage:
        bot = (
            PerpsBotBuilder(config)
            .with_registry(custom_registry)
            .with_symbols(["BTC-USD", "ETH-USD"])
            .build()
        )

    Default Behavior:
        - Creates ConfigController if not provided
        - Creates empty registry if not provided
        - Uses config.symbols if symbols not explicitly set
        - Auto-wires services based on feature flags
    """

    def __init__(self, config: BotConfig) -> None:
        """Initialize builder with required config.

        Args:
            config: Bot configuration (required)
        """
        self._config = config
        self._registry: ServiceRegistry | None = None
        self._symbols: list[str] | None = None
        self._config_controller: ConfigController | None = None

        # Built instance attributes (populated during build())
        self._bot_id: str | None = None
        self._start_time: datetime | None = None
        self._running: bool = False

    def with_registry(self, registry: ServiceRegistry) -> PerpsBotBuilder:
        """Set custom service registry.

        Args:
            registry: Pre-configured service registry

        Returns:
            Self for method chaining
        """
        self._registry = registry
        return self

    def with_symbols(self, symbols: list[str]) -> PerpsBotBuilder:
        """Set trading symbols.

        Args:
            symbols: List of trading symbols (e.g., ["BTC-USD", "ETH-USD"])

        Returns:
            Self for method chaining
        """
        self._symbols = symbols
        return self

    def with_config_controller(self, controller: ConfigController) -> PerpsBotBuilder:
        """Set custom config controller.

        Args:
            controller: Pre-configured config controller

        Returns:
            Self for method chaining
        """
        self._config_controller = controller
        return self

    def build(self) -> IBotRuntime:
        """Build and return a fully initialized :class:`PerpsBot` instance."""

        from bot_v2.orchestration.perps_bot import PerpsBot

        bot = PerpsBot.__new__(PerpsBot)
        self.build_into(bot)
        return bot

    def build_into(self, bot: IBotRuntime) -> None:
        """Populate an existing :class:`PerpsBot` instance via the builder pipeline."""

        self._build_configuration_state(bot)
        self._build_runtime_state(bot)
        self._build_storage(bot)
        self._build_core_services(bot)
        bot.guardrails.set_dry_run(bool(bot.config.dry_run))
        bot.metrics_server.register_guard_manager(bot.guardrails, profile=bot.config.profile.value)
        bot.runtime_coordinator.bootstrap()  # Initialize runtime coordinator
        self._build_market_data_service(bot)
        self._build_accounting_services(bot)
        self._build_market_services(bot)
        self._build_streaming_service(bot)
        self._start_streaming_if_configured(bot)

        # Ensure services that captured `_bot` during construction reference the
        # populated instance.
        rebind_bot_services(bot)

    # ------------------------------------------------------------------
    # Construction Phase Methods
    # ------------------------------------------------------------------

    def _build_configuration_state(self, bot: IBotRuntime) -> None:
        """Phase 1: Initialize configuration state.

        Creates:
            - bot.config_controller (or uses provided)
            - bot.config
            - bot.registry (synchronized with config)
            - bot._session_guard
            - bot.symbols
            - bot._derivatives_enabled
            - bot.bot_id
            - bot.start_time
            - bot.running
        """
        from datetime import UTC, datetime

        # Create or use provided ConfigController
        if self._config_controller is not None:
            bot.config_controller = self._config_controller
        else:
            bot.config_controller = ConfigController(self._config)

        bot.config = bot.config_controller.current

        # Create or sync registry
        base_registry = self._registry or empty_registry(bot.config)
        if base_registry.config is not bot.config:
            base_registry = base_registry.with_updates(config=bot.config)
        bot.registry = base_registry

        # Session guard
        bot._session_guard = TradingSessionGuard(
            start=bot.config.trading_window_start,
            end=bot.config.trading_window_end,
            trading_days=bot.config.trading_days,
        )

        # Symbols (use explicit symbols or fall back to config)
        bot.symbols = list(self._symbols or bot.config.symbols or [])
        if not bot.symbols:
            logger.warning("No symbols configured; continuing with empty symbol list")

        bot._derivatives_enabled = bool(getattr(bot.config, "derivatives_enabled", False))

        # Bot metadata
        bot.bot_id = "perps_bot"
        bot.start_time = datetime.now(UTC)
        bot.running = False

    def _build_runtime_state(self, bot: IBotRuntime) -> None:
        """Phase 2: Initialize runtime state.

        Creates:
            - bot.mark_windows
            - bot.last_decisions
            - bot._last_positions
            - bot.order_stats
            - bot._order_lock
            - bot._mark_lock
            - bot._symbol_strategies
            - bot.strategy
            - bot._exec_engine
            - bot._streaming_service
            - bot._product_map
        """
        from typing import Any

        from bot_v2.features.brokerages.core.interfaces import Product

        bot.mark_windows: dict[str, list[Decimal]] = {s: [] for s in bot.symbols}
        bot.last_decisions: dict[str, Any] = {}
        bot._last_positions: dict[str, dict[str, Any]] = {}
        bot.order_stats = {"attempted": 0, "successful": 0, "failed": 0}
        bot._order_lock: asyncio.Lock | None = None
        bot._mark_lock = threading.RLock()
        bot._symbol_strategies: dict[str, Any] = {}
        bot.strategy: Any | None = None
        bot._exec_engine: LiveExecutionEngine | AdvancedExecutionEngine | None = None
        bot._streaming_service: StreamingService | None = None
        bot._product_map: dict[str, Product] = {}

    def _build_storage(self, bot: IBotRuntime) -> None:
        """Phase 3: Bootstrap storage layer.

        Creates:
            - bot.event_store
            - bot.orders_store
            - Updates bot.registry with storage references
        """
        storage_ctx = StorageBootstrapper(bot.config, bot.registry).bootstrap()
        bot.event_store = storage_ctx.event_store
        bot.orders_store = storage_ctx.orders_store
        bot.registry = storage_ctx.registry

    def _build_core_services(self, bot: IBotRuntime) -> None:
        """Phase 4: Construct core orchestration services.

        Creates:
            - bot.strategy_orchestrator
            - bot.execution_coordinator
            - bot.system_monitor
            - bot.runtime_coordinator
            - bot.lifecycle_service

        IMPORTANT: Services that cache a `_bot` reference are rebound at the end of
        ``build_into`` so they point at the populated runtime instance. If you add a
        new service here that stores ``self._bot = bot``, make sure it survives the
        ``rebind_bot_services`` call.
        """
        from bot_v2.orchestration.lifecycle_service import LifecycleService

        monitoring_settings = getattr(bot.config, "monitoring_settings", {}) or {}
        bot.strategy_orchestrator = StrategyOrchestrator(bot)
        bot.execution_coordinator = ExecutionCoordinator(bot)
        bot.system_monitor = SystemMonitor(
            bot,
            monitor_settings=monitoring_settings,
            alert_settings=monitoring_settings.get("alerts") if monitoring_settings else None,
        )
        bot.runtime_coordinator = RuntimeCoordinator(bot)
        bot.lifecycle_service = LifecycleService(bot)
        bot.guardrails = GuardRailManager(
            max_trade_value=bot.config.max_trade_value,
            symbol_position_caps=bot.config.symbol_position_caps,
            daily_loss_limit=bot.config.daily_loss_limit,
        )

        # Initialize metrics server (configured from bot config)
        metrics_port = getattr(bot.config, "metrics_port", 9090)
        bot.metrics_server = MetricsServer(host="0.0.0.0", port=metrics_port)

    def _build_market_data_service(self, bot: IBotRuntime) -> None:
        """Phase 5: Initialize MarketDataService for mark/window maintenance."""

        bot._market_data_service = MarketDataService(
            symbols=bot.symbols,
            broker=bot.broker,
            risk_manager=bot.risk_manager,
            long_ma=bot.config.long_ma,
            short_ma=bot.config.short_ma,
            mark_lock=bot._mark_lock,
            mark_windows=bot.mark_windows,  # Share existing dict for backward compat
        )

    def _build_accounting_services(self, bot: IBotRuntime) -> None:
        """Phase 6: Initialize accounting and telemetry services.

        Creates:
            - bot.account_manager
            - bot.account_telemetry
            - Attaches telemetry to system_monitor
        """
        bot.account_manager = CoinbaseAccountManager(bot.broker, event_store=bot.event_store)
        bot.account_telemetry = AccountTelemetryService(
            broker=bot.broker,
            account_manager=bot.account_manager,
            event_store=bot.event_store,
            bot_id=bot.bot_id,
            profile=bot.config.profile.value,
        )
        if not bot.account_telemetry.supports_snapshots():
            logger.info("Account snapshot telemetry disabled; broker lacks required endpoints")
        bot.system_monitor.attach_account_telemetry(bot.account_telemetry)

    def _build_market_services(self, bot: IBotRuntime) -> None:
        """Phase 7: Initialize market monitoring services.

        Creates:
            - bot._market_monitor (with heartbeat logger)
        """
        from typing import Any

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

        bot._market_monitor = MarketActivityMonitor(
            tuple(bot.symbols),
            heartbeat_logger=_log_market_heartbeat,
        )

    def _build_streaming_service(self, bot: IBotRuntime) -> None:
        """Phase 8: Initialize StreamingService (Phase 2 refactoring).

        Creates:
            - bot._streaming_service (unconditionally, mirroring MarketDataService)
        """
        if bot._market_data_service is not None:
            stream_name = getattr(bot.broker, "stream_name", "coinbase_ws")
            bot._streaming_service = StreamingService(
                symbols=bot.symbols,
                broker=bot.broker,
                market_data_service=bot._market_data_service,
                risk_manager=bot.risk_manager,
                event_store=bot.event_store,
                market_monitor=bot._market_monitor,
                bot_id=bot.bot_id,
                metrics_server=bot.metrics_server,
                profile=bot.config.profile.value,
                stream_name=stream_name,
                rest_poll_interval=getattr(bot.config, "streaming_rest_poll_interval", 5.0),
            )
        else:
            bot._streaming_service = None

    def _start_streaming_if_configured(self, bot: IBotRuntime) -> None:
        """Phase 9: Start streaming if enabled in config.

        Starts streaming if:
            - config.perps_enable_streaming=true
            - config.profile in {CANARY, PROD}
        """
        if getattr(bot.config, "perps_enable_streaming", False) and bot.config.profile in {
            Profile.CANARY,
            Profile.PROD,
        }:
            try:
                if bot._streaming_service is not None:
                    configured_level = getattr(bot.config, "perps_stream_level", 1) or 1
                    bot._streaming_service.start(level=configured_level)
            except Exception:
                logger.exception("Failed to start streaming background worker")


# Type hint for bot interface (avoids circular import)
if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.orchestration.core import IBotRuntime
