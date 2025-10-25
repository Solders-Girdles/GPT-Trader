"""Runtime accessors and property helpers for the Perps bot."""

from __future__ import annotations

import asyncio
import threading
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.account_telemetry import AccountTelemetryService
    from bot_v2.orchestration.coordinators import (
        ExecutionCoordinator,
        RuntimeCoordinator,
        StrategyCoordinator,
        TelemetryCoordinator,
    )
    from bot_v2.orchestration.intx_portfolio_service import IntxPortfolioService
    from bot_v2.orchestration.live_execution import LiveExecutionEngine
    from bot_v2.orchestration.market_monitor import MarketActivityMonitor
    from bot_v2.orchestration.perps_bot import PerpsBot


class PerpsBotRuntimeAccessMixin:
    """Expose runtime state and service accessors."""

    @property
    def runtime_state(self: PerpsBot) -> Any:
        return self._state

    @property
    def runtime_coordinator(self) -> RuntimeCoordinator:
        from bot_v2.orchestration.coordinators import RuntimeCoordinator

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
        from bot_v2.orchestration.coordinators import TelemetryCoordinator

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
