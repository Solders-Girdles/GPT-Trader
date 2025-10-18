"""Context-driven telemetry coordinator implementation for PerpsBot."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.monitoring.system import get_logger as _get_plog
from bot_v2.orchestration.account_telemetry import AccountTelemetryService
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.intx_portfolio_service import IntxPortfolioService
from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.utilities import emit_metric, utc_now
from bot_v2.utilities.logging_patterns import get_logger

from .base import BaseCoordinator, CoordinatorContext, HealthStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage

logger = get_logger(__name__, component="telemetry_coordinator")


class TelemetryCoordinator(BaseCoordinator):
    """Manages account telemetry services and market streaming for the bot."""

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self._stream_task: asyncio.Task[Any] | None = None
        self._ws_stop: threading.Event | None = None
        self._market_monitor: MarketActivityMonitor | None = None
        self._pending_stream_config: tuple[list[str], int] | None = None

    @property
    def name(self) -> str:
        return "telemetry"

    def initialize(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self.context
        broker = ctx.broker
        if broker is None:
            logger.warning(
                "Telemetry initialization skipped: no broker available",
                operation="telemetry_init",
                stage="missing_broker",
            )
            return ctx

        try:
            from bot_v2.features.brokerages.coinbase.adapter import (
                CoinbaseBrokerage as _CoinbaseBrokerage,
            )
        except Exception:  # pragma: no cover - fallback
            logger.warning(
                "Coinbase adapter unavailable; telemetry coordinator skipping setup",
                operation="telemetry_init",
                stage="adapter_missing",
            )
            return ctx

        if not isinstance(broker, _CoinbaseBrokerage):
            logger.warning(
                "Telemetry coordinator requires a Coinbase brokerage; skipping setup",
                operation="telemetry_init",
                stage="adapter_mismatch",
            )
            return ctx

        account_manager = CoinbaseAccountManager(
            cast("CoinbaseBrokerage", broker), event_store=ctx.event_store
        )
        intx_service = IntxPortfolioService(
            account_manager=account_manager,
            runtime_settings=ctx.registry.runtime_settings if ctx.registry else None,
        )
        account_telemetry = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=ctx.event_store,
            bot_id=ctx.bot_id,
            profile=ctx.config.profile.value,
        )
        if not account_telemetry.supports_snapshots():
            logger.info(
                "Account snapshot telemetry disabled; broker lacks required endpoints",
                operation="telemetry_init",
                stage="snapshot_disabled",
            )

        def _log_market_heartbeat(**payload: Any) -> None:
            try:
                _get_plog().log_market_heartbeat(**payload)
            except Exception as exc:
                logger.debug(
                    "Failed to record market heartbeat",
                    symbol=payload.get("symbol") or payload.get("source"),
                    error=str(exc),
                    exc_info=True,
                    operation="market_monitor",
                    stage="heartbeat",
                )

        market_monitor = MarketActivityMonitor(ctx.symbols, heartbeat_logger=_log_market_heartbeat)
        self._market_monitor = market_monitor

        extras = dict(ctx.registry.extras)
        extras.update(
            {
                "account_manager": account_manager,
                "account_telemetry": account_telemetry,
                "intx_portfolio_service": intx_service,
                "market_monitor": market_monitor,
            }
        )
        updated_registry = ctx.registry.with_updates(extras=extras)
        return ctx.with_updates(registry=updated_registry)

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        tasks: list[asyncio.Task[Any]] = []

        account_telemetry = self.context.registry.extras.get("account_telemetry")
        if account_telemetry and account_telemetry.supports_snapshots():
            interval = self.context.config.account_telemetry_interval
            task = asyncio.create_task(self._run_account_telemetry(interval))
            self._register_background_task(task)
            tasks.append(task)
            logger.info(
                "Started account telemetry background task",
                interval=interval,
                operation="telemetry_tasks",
                stage="account_telemetry",
            )

        if self._should_enable_streaming():
            try:
                stream_task = await self._start_streaming()
                if stream_task is not None:
                    self._register_background_task(stream_task)
                    tasks.append(stream_task)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "Failed to start streaming background task",
                    error=str(exc),
                    exc_info=True,
                    operation="telemetry_tasks",
                    stage="stream_start",
                )

        return tasks

    async def shutdown(self) -> None:
        logger.info("Shutting down telemetry coordinator...")
        await self._stop_streaming()
        await super().shutdown()

    def health_check(self) -> HealthStatus:
        extras = self.context.registry.extras
        account_telemetry = extras.get("account_telemetry")
        healthy = account_telemetry is not None
        details = {
            "has_account_telemetry": account_telemetry is not None,
            "has_market_monitor": extras.get("market_monitor") is not None
            or self._market_monitor is not None,
            "streaming_active": self._stream_task is not None and not self._stream_task.done(),
            "background_tasks": len(self._background_tasks),
        }
        return HealthStatus(healthy=healthy, component=self.name, details=details)

    # ------------------------------------------------------------------ account telemetry helpers
    async def _run_account_telemetry(self, interval_seconds: int) -> None:
        account_telemetry = self.context.registry.extras.get("account_telemetry")
        if not account_telemetry or not account_telemetry.supports_snapshots():
            return
        await account_telemetry.run(interval_seconds)

    # ------------------------------------------------------------------ streaming helpers
    def _should_enable_streaming(self) -> bool:
        config = self.context.config
        return bool(config.perps_enable_streaming) and config.profile in {
            Profile.CANARY,
            Profile.PROD,
        }

    async def _start_streaming(self) -> asyncio.Task[Any] | None:
        symbols = list(self.context.symbols)
        if not symbols:
            logger.debug(
                "No symbols configured; skipping streaming",
                operation="telemetry_stream",
                stage="skip",
            )
            return None

        configured_level = self.context.config.perps_stream_level or 1
        try:
            level = max(int(configured_level), 1)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid streaming level; defaulting to 1",
                configured_level=configured_level,
                operation="telemetry_stream",
                stage="config",
            )
            level = 1

        self._ws_stop = threading.Event()
        self._pending_stream_config = (symbols, level)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                "No running event loop; streaming will be deferred",
                operation="telemetry_stream",
                stage="deferred",
            )
            return None

        task = loop.create_task(self._run_stream_loop_async(symbols, level, self._ws_stop))
        task.add_done_callback(self._handle_stream_task_completion)
        self._stream_task = task
        logger.info(
            "Started WS streaming task",
            symbols=symbols,
            level=level,
            operation="telemetry_stream",
            stage="start",
        )
        return task

    async def _stop_streaming(self) -> None:
        self._pending_stream_config = None
        if self._ws_stop:
            self._ws_stop.set()
            self._ws_stop = None

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                logger.info(
                    "WS streaming task cancelled",
                    operation="telemetry_stream",
                    stage="cancel",
                )
        self._stream_task = None
        logger.info(
            "Streaming halted",
            operation="telemetry_stream",
            stage="stop",
        )

    def _handle_stream_task_completion(self, task: asyncio.Task[Any]) -> None:
        self._stream_task = None
        self._ws_stop = None
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info(
                "WS streaming task cancelled",
                operation="telemetry_stream",
                stage="cancel",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "WS streaming task failed",
                error=str(exc),
                operation="telemetry_stream",
                stage="failed",
            )

    async def _run_stream_loop_async(
        self,
        symbols: list[str],
        level: int,
        stop_signal: threading.Event | None,
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                self._run_stream_loop,
                symbols,
                level,
                stop_signal,
            )
        except asyncio.CancelledError:
            if stop_signal:
                stop_signal.set()
            raise

    def _run_stream_loop(
        self,
        symbols: list[str],
        level: int,
        stop_signal: threading.Event | None,
    ) -> None:
        ctx = self.context
        broker = ctx.broker
        if broker is None:
            logger.error(
                "Cannot start streaming: no broker available",
                operation="telemetry_stream",
                stage="run",
            )
            return

        stream: Iterable[Any] | None = None
        try:
            stream = broker.stream_orderbook(symbols, level=level)
        except Exception as exc:  # pragma: no cover - dependent on broker impl
            logger.warning(
                "Orderbook stream unavailable, falling back to trades",
                error=str(exc),
                operation="telemetry_stream",
                stage="orderbook",
            )
            try:
                stream = broker.stream_trades(symbols)
            except Exception as trade_exc:
                logger.error(
                    "Failed to start streaming trades",
                    error=str(trade_exc),
                    operation="telemetry_stream",
                    stage="trades",
                )
                stream = None

        try:
            for msg in stream or []:
                if stop_signal and stop_signal.is_set():
                    break
                if not isinstance(msg, dict):
                    continue
                ctx = self.context
                sym = str(msg.get("product_id") or msg.get("symbol") or "")
                if not sym:
                    continue

                mark = self._extract_mark_from_message(msg)
                if mark is None or mark <= 0:
                    continue

                self._update_mark_and_metrics(ctx, sym, mark)
        except Exception as exc:  # pragma: no cover - defensive logging
            ctx = self.context
            emit_metric(
                ctx.event_store,
                ctx.bot_id,
                {"event_type": "ws_stream_error", "message": str(exc)},
            )
        finally:
            ctx = self.context
            emit_metric(
                ctx.event_store,
                ctx.bot_id,
                {"event_type": "ws_stream_exit"},
            )

    @staticmethod
    def _extract_mark_from_message(msg: dict[str, Any]) -> Decimal | None:
        bid = msg.get("best_bid") or msg.get("bid")
        ask = msg.get("best_ask") or msg.get("ask")
        try:
            if bid is not None and ask is not None:
                return (Decimal(str(bid)) + Decimal(str(ask))) / Decimal("2")
            raw_mark = msg.get("last") or msg.get("price")
            if raw_mark is not None:
                return Decimal(str(raw_mark))
        except Exception:
            return None
        return None

    def _update_mark_and_metrics(self, ctx: CoordinatorContext, symbol: str, mark: Decimal) -> None:
        strategy_coordinator = getattr(ctx, "strategy_coordinator", None)
        if strategy_coordinator and hasattr(strategy_coordinator, "update_mark_window"):
            try:
                strategy_coordinator.update_mark_window(symbol, mark)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug(
                    "Failed to update mark window",
                    error=str(exc),
                    symbol=symbol,
                    exc_info=True,
                    operation="telemetry_stream",
                    stage="mark_window",
                )
        else:
            runtime_state = ctx.runtime_state
            if runtime_state is not None:
                with runtime_state.mark_lock:
                    window = runtime_state.mark_windows.setdefault(symbol, [])
                    window.append(mark)
                    max_size = max(ctx.config.short_ma, ctx.config.long_ma) + 5
                    if len(window) > max_size:
                        runtime_state.mark_windows[symbol] = window[-max_size:]

        monitor = self._market_monitor or ctx.registry.extras.get("market_monitor")
        if monitor is not None:
            try:
                monitor.record_update(symbol)
            except Exception:  # pragma: no cover - defensive logging
                logger.debug(
                    "Failed to record market update",
                    symbol=symbol,
                    exc_info=True,
                    operation="telemetry_stream",
                    stage="market_monitor",
                )

        risk_manager = ctx.risk_manager
        if risk_manager is not None:
            try:
                timestamp = utc_now()
                record_fn = getattr(risk_manager, "record_mark_update", None)
                stored = record_fn(symbol, timestamp) if callable(record_fn) else timestamp
                risk_manager.last_mark_update[symbol] = stored
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "WS mark update bookkeeping failed",
                    symbol=symbol,
                    operation="telemetry_stream",
                    stage="risk_update",
                )

        emit_metric(
            ctx.event_store,
            ctx.bot_id,
            {"event_type": "ws_mark_update", "symbol": symbol, "mark": str(mark)},
        )


__all__ = ["TelemetryCoordinator"]
