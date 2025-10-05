from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from threading import RLock
from typing import TYPE_CHECKING, Any

from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    MarketType,
    Position,
    Product,
)
from bot_v2.monitoring.metrics_server import MetricsServer
from bot_v2.orchestration.config_controller import ConfigChange
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.account_telemetry import AccountTelemetryService
    from bot_v2.orchestration.builders import PerpsBotBuilder
    from bot_v2.orchestration.config_controller import ConfigController
    from bot_v2.orchestration.execution_coordinator import ExecutionCoordinator
    from bot_v2.orchestration.guardrails import GuardRailManager
    from bot_v2.orchestration.lifecycle_service import LifecycleService
    from bot_v2.orchestration.live_execution import LiveExecutionEngine
    from bot_v2.orchestration.market_data_service import MarketDataService
    from bot_v2.orchestration.market_monitor import MarketActivityMonitor
    from bot_v2.orchestration.runtime_coordinator import RuntimeCoordinator
    from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
    from bot_v2.orchestration.streaming_service import StreamingService
    from bot_v2.orchestration.system_monitor import SystemMonitor
    from bot_v2.persistence.event_store import EventStore
    from bot_v2.persistence.orders_store import OrdersStore

logger = logging.getLogger(__name__)


class PerpsBot:
    """Main Coinbase trading bot (spot by default, perps optional)."""

    registry: ServiceRegistry
    config_controller: ConfigController
    config: BotConfig
    _session_guard: TradingSessionGuard
    symbols: list[str]
    _derivatives_enabled: bool
    bot_id: str
    start_time: datetime
    running: bool

    mark_windows: dict[str, list[Decimal]]
    last_decisions: dict[str, Any]
    _last_positions: dict[str, dict[str, Any]]
    order_stats: dict[str, int]
    _order_lock: asyncio.Lock | None
    _mark_lock: RLock
    _symbol_strategies: dict[str, Any]
    strategy: Any | None
    _exec_engine: LiveExecutionEngine | AdvancedExecutionEngine | None
    _streaming_service: StreamingService | None
    _market_data_service: MarketDataService | None
    _market_monitor: MarketActivityMonitor | None
    _product_map: dict[str, Product]

    strategy_orchestrator: StrategyOrchestrator
    execution_coordinator: ExecutionCoordinator
    system_monitor: SystemMonitor
    runtime_coordinator: RuntimeCoordinator
    lifecycle_service: LifecycleService
    account_manager: CoinbaseAccountManager
    account_telemetry: AccountTelemetryService
    event_store: EventStore
    orders_store: OrdersStore
    metrics_server: MetricsServer
    guardrails: GuardRailManager

    def __init__(self, config: BotConfig, registry: ServiceRegistry | None = None) -> None:
        """Construct the bot using the builder pipeline."""
        from bot_v2.orchestration.builders import PerpsBotBuilder

        builder = PerpsBotBuilder(config)
        if registry is not None:
            builder = builder.with_registry(registry)

        builder.build_into(self)

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

    def _start_streaming_if_configured(self) -> None:
        if getattr(self.config, "perps_enable_streaming", False) and self.config.profile in {
            Profile.CANARY,
            Profile.PROD,
        }:
            try:
                if self._streaming_service is not None:
                    configured_level = getattr(self.config, "perps_stream_level", 1) or 1
                    self._streaming_service.start(level=configured_level)
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
        self.lifecycle_service.configure_background_tasks(single_cycle)
        await self.lifecycle_service.run(single_cycle)

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

        # Check cycle-level guards (e.g., daily loss limit)
        if hasattr(self, "guardrails"):
            cycle_context = {
                "balances": balances,
                "positions": positions,
                "position_map": position_map,
            }
            self.guardrails.check_cycle(cycle_context)

            # If daily loss guard triggered, activate reduce-only mode
            if self.guardrails.is_guard_active("daily_loss"):
                if not self.is_reduce_only_mode():
                    logger.warning("Daily loss guard active - entering reduce-only mode")
                    self.set_reduce_only_mode(True, "daily_loss_limit_reached")

        tasks = []
        for symbol in self.symbols:
            tasks.append(self.process_symbol(symbol, balances, position_map))
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

    # ------------------------------------------------------------------
    async def update_marks(self) -> None:
        """Update mark prices for all symbols via :class:`MarketDataService`."""
        if self._market_data_service is None:
            raise RuntimeError("MarketDataService is not initialized")
        await self._market_data_service.update_marks()

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
        except Exception as exc:
            logger.debug("Failed to stop streaming service cleanly: %s", exc, exc_info=True)

        try:
            if hasattr(self, "metrics_server") and self.metrics_server.is_running:
                self.metrics_server.stop()
        except Exception as exc:
            logger.debug("Failed to stop metrics server cleanly: %s", exc, exc_info=True)

    def _restart_streaming_if_needed(self, diff: dict[str, Any]) -> None:
        """Restart streaming service when config changes streaming settings."""
        streaming_enabled = bool(getattr(self.config, "perps_enable_streaming", False))
        toggle_changed = "perps_enable_streaming" in diff or "perps_stream_level" in diff

        if not streaming_enabled:
            if toggle_changed and self._streaming_service is not None:
                self._streaming_service.stop()
            return

        # For enabled streaming ensure we refresh when toggle flips or level changes.
        if toggle_changed and self._streaming_service is not None:
            self._streaming_service.stop()
        self._start_streaming_if_configured()

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
        if hasattr(self, "guardrails"):
            self.guardrails.set_dry_run(bool(self.config.dry_run))
            self.guardrails.update_limits(
                max_trade_value=self.config.max_trade_value,
                symbol_position_caps=self.config.symbol_position_caps,
                daily_loss_limit=self.config.daily_loss_limit,
            )
        for symbol in self.symbols:
            self.mark_windows.setdefault(symbol, [])
        for symbol in list(self.mark_windows.keys()):
            if symbol not in self.symbols:
                del self.mark_windows[symbol]

        # Update streaming service symbols if changed
        if "symbols" in change.diff and self._streaming_service is not None:
            self._streaming_service.update_symbols(self.symbols)

        if self._streaming_service is not None:
            self._streaming_service.set_rest_poll_interval(
                getattr(self.config, "streaming_rest_poll_interval", 5.0)
            )

        # Restart streaming if streaming config changed
        self._restart_streaming_if_needed(change.diff)

        self.strategy_orchestrator.init_strategy()

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

    async def _run_execution_metrics_export(self, interval_seconds: int = 60) -> None:
        """Export execution metrics to MetricsCollector periodically.

        Args:
            interval_seconds: Export interval in seconds (default: 60)
        """
        from bot_v2.monitoring.metrics_collector import get_metrics_collector

        collector = get_metrics_collector()

        while True:
            try:
                # Export AdvancedExecutionEngine metrics if available
                if hasattr(self, "exec_engine") and hasattr(self.exec_engine, "export_metrics"):
                    self.exec_engine.export_metrics(collector, prefix="execution")
            except Exception as exc:
                logger.debug("Failed to export execution metrics: %s", exc, exc_info=True)

            await asyncio.sleep(interval_seconds)
