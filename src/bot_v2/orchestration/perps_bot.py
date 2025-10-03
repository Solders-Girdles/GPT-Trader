from __future__ import annotations

import asyncio
import inspect
import logging
import os
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
from bot_v2.orchestration.config_controller import ConfigChange
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.service_rebinding import rebind_bot_services
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard

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
        """Construct the bot using the builder pipeline."""
        from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder

        builder = PerpsBotBuilder(config)
        if registry is not None:
            builder = builder.with_registry(registry)

        built = builder.build()
        self._apply_built_state(built)

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
        """Adopt the state from an instance produced by :class:`PerpsBotBuilder`.

        We instantiate the builder with a temporary object, then copy its finalized
        state onto ``self`` so caller code that expects ``PerpsBot()`` construction
        continues to work. After copying the attributes we invoke
        :func:`rebind_bot_services` so any coordinator that cached a ``_bot``
        reference during construction points at this real runtime instance.

        Args:
            built: Fully initialized :class:`PerpsBot` instance returned by the builder.
        """
        # Copy all attributes from built instance
        for key, value in built.__dict__.items():
            setattr(self, key, value)

        rebind_bot_services(self)

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

        # Note: Market data and streaming services are initialized by the builder
        # and handle config updates internally. No reinitialization needed here.
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
