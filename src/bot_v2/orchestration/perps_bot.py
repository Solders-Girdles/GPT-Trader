from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    MarketType,
    Order,
    Position,
    Product,
)
from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian
from bot_v2.orchestration.config_controller import ConfigChange, ConfigController
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.execution_coordinator import ExecutionCoordinator
from bot_v2.orchestration.lifecycle_manager import LifecycleManager
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.runtime_coordinator import RuntimeCoordinator
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard
from bot_v2.orchestration.strategy_coordinator import StrategyCoordinator
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
from bot_v2.orchestration.system_monitor import SystemMonitor
from bot_v2.orchestration.telemetry_coordinator import TelemetryCoordinator
from bot_v2.utilities.config import ConfigBaselinePayload

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
        self.strategy_orchestrator = StrategyOrchestrator(self)
        self.execution_coordinator = ExecutionCoordinator(self)
        self.system_monitor = SystemMonitor(self)
        self.runtime_coordinator = RuntimeCoordinator(self)
        self.lifecycle_manager = LifecycleManager(self)
        self.strategy_coordinator = StrategyCoordinator(self)
        self.telemetry_coordinator = TelemetryCoordinator(self)

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

        payload = ConfigBaselinePayload.from_config(
            config,
            derivatives_enabled=derivatives_enabled,
        )

        active_symbols = list(payload.symbols)
        broker_type = "mock" if config.mock_broker else "live"

        return ConfigurationGuardian.create_baseline_snapshot(
            config_dict=payload,
            active_symbols=active_symbols,
            positions=[],
            account_equity=None,
            profile=config.profile,
            broker_type=broker_type,
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
        return engine

    # ------------------------------------------------------------------
    async def run(self, single_cycle: bool = False) -> None:
        await self.lifecycle_manager.run(single_cycle)

    async def run_cycle(self) -> None:
        await self.strategy_coordinator.run_cycle()

    async def _fetch_current_state(self) -> dict[str, Any]:
        return await self.strategy_coordinator._fetch_current_state()

    async def _validate_configuration_and_handle_drift(self, current_state: dict[str, Any]) -> bool:
        return await self.strategy_coordinator._validate_configuration_and_handle_drift(
            current_state
        )

    async def _execute_trading_cycle(self, trading_state: dict[str, Any]) -> None:
        await self.strategy_coordinator._execute_trading_cycle(trading_state)

    async def process_symbol(
        self,
        symbol: str,
        balances: Sequence[Balance] | None = None,
        position_map: dict[str, Position] | None = None,
    ) -> None:
        await self.strategy_coordinator.process_symbol(symbol, balances, position_map)

    def _process_symbol_expects_context(self) -> bool:
        return self.strategy_coordinator._process_symbol_expects_context()

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        await self.strategy_coordinator.execute_decision(
            symbol, decision, mark, product, position_state
        )

    def _ensure_order_lock(self) -> asyncio.Lock:
        return self.strategy_coordinator.ensure_order_lock()

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
        await self.lifecycle_manager.shutdown()

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
        self.telemetry_coordinator._run_stream_loop(symbols, level)

    # ------------------------------------------------------------------
    def _update_mark_window(self, symbol: str, mark: Decimal) -> None:
        self.strategy_coordinator.update_mark_window(symbol, mark)

    @staticmethod
    def _calculate_spread_bps(bid_price: Decimal, ask_price: Decimal) -> Decimal:
        return StrategyCoordinator.calculate_spread_bps(bid_price, ask_price)

    # ------------------------------------------------------------------
    async def _run_account_telemetry(self, interval_seconds: int = 300) -> None:
        await self.telemetry_coordinator.run_account_telemetry(interval_seconds)
