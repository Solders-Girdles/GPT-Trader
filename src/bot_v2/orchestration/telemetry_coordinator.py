"""Telemetry and streaming coordination for :class:`PerpsBot`."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterable
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.features.brokerages.coinbase.account_manager import CoinbaseAccountManager
from bot_v2.monitoring.system import get_logger as _get_plog
from bot_v2.orchestration.account_telemetry import AccountTelemetryService
from bot_v2.orchestration.configuration import Profile
from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.utilities import emit_metric, utc_now

if TYPE_CHECKING:  # pragma: no cover - circular type import guard
    from bot_v2.orchestration.perps_bot import PerpsBot


logger = logging.getLogger(__name__)


class TelemetryCoordinator:
    """Manage account telemetry services and market streaming for ``PerpsBot``."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot
        self._stream_task: asyncio.Task[None] | None = None
        self._ws_stop: threading.Event | None = None
        self._pending_stream_config: tuple[list[str], int] | None = None
        self._market_monitor: MarketActivityMonitor | None = None

    # ------------------------------------------------------------------
    def bootstrap(self) -> None:
        """Initialise telemetry services and start streaming if eligible."""

        self.init_accounting_services()
        self.init_market_services()
        self.start_streaming_if_configured()

    def init_accounting_services(self) -> None:
        bot = self._bot
        account_manager = CoinbaseAccountManager(bot.broker, event_store=bot.event_store)
        bot.account_manager = account_manager
        account_telemetry = AccountTelemetryService(
            broker=bot.broker,
            account_manager=account_manager,
            event_store=bot.event_store,
            bot_id=bot.bot_id,
            profile=bot.config.profile.value,
        )
        bot.account_telemetry = account_telemetry
        if not account_telemetry.supports_snapshots():
            logger.info("Account snapshot telemetry disabled; broker lacks required endpoints")
        bot.system_monitor.attach_account_telemetry(account_telemetry)

    def init_market_services(self) -> None:
        bot = self._bot

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

        monitor = MarketActivityMonitor(tuple(bot.symbols), heartbeat_logger=_log_market_heartbeat)
        bot.market_monitor = monitor
        self._market_monitor = monitor

    # ------------------------------------------------------------------
    def start_streaming_if_configured(self) -> None:
        bot = self._bot
        if getattr(bot.config, "perps_enable_streaming", False) and bot.config.profile in {
            Profile.CANARY,
            Profile.PROD,
        }:
            try:
                self.start_streaming_background()
            except Exception:
                logger.exception("Failed to start streaming background worker")

    def start_streaming_background(self) -> None:  # pragma: no cover - gated by env/profile
        bot = self._bot
        symbols = list(bot.symbols)
        if not symbols:
            return
        configured_level = getattr(bot.config, "perps_stream_level", 1) or 1
        try:
            level = max(int(configured_level), 1)
        except (TypeError, ValueError):
            logger.warning("Invalid streaming level %s; defaulting to 1", configured_level)
            level = 1
        self._pending_stream_config = (symbols, level)
        scheduled = self._schedule_streaming_task()
        if not scheduled:
            logger.debug(
                "WS streaming task deferred; no running event loop yet (symbols=%s level=%s)",
                symbols,
                level,
            )

    def stop_streaming_background(self) -> None:
        self._pending_stream_config = None
        stop = self._ws_stop
        if stop:
            stop.set()
            self._ws_stop = None
        task = self._stream_task
        if task and not task.done():
            loop = task.get_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()
        self._stream_task = None

    def restart_streaming_if_needed(self, diff: dict[str, Any]) -> None:
        bot = self._bot
        streaming_enabled = bool(getattr(bot.config, "perps_enable_streaming", False))
        toggle_changed = "perps_enable_streaming" in diff or "perps_stream_level" in diff

        if not streaming_enabled:
            if toggle_changed:
                self.stop_streaming_background()
            return

        if toggle_changed:
            self.stop_streaming_background()
        self.start_streaming_if_configured()

    # ------------------------------------------------------------------
    async def run_account_telemetry(self, interval_seconds: int = 300) -> None:
        bot = self._bot
        account_telemetry = bot.account_telemetry
        if not account_telemetry or not account_telemetry.supports_snapshots():
            return
        await account_telemetry.run(interval_seconds)

    # ------------------------------------------------------------------
    def ensure_streaming_task(self) -> asyncio.Task[None] | None:
        """Ensure the WS stream task is scheduled when the event loop is running."""

        self._schedule_streaming_task()
        return self._stream_task

    def _schedule_streaming_task(self) -> bool:
        if not self._pending_stream_config:
            return False
        if self._stream_task and not self._stream_task.done():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        symbols, level = self._pending_stream_config
        self._ws_stop = threading.Event()
        task = loop.create_task(self._run_stream_loop_async(symbols, level, self._ws_stop))
        task.add_done_callback(self._handle_stream_task_completion)
        self._stream_task = task
        logger.info("Started WS streaming task for symbols=%s level=%s", symbols, level)
        return True

    def _handle_stream_task_completion(self, task: asyncio.Task[None]) -> None:
        self._stream_task = None
        self._ws_stop = None
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("WS streaming task cancelled")
        except Exception as exc:
            logger.exception("WS streaming task failed: %s", exc)

    async def _run_stream_loop_async(
        self, symbols: list[str], level: int, stop_signal: threading.Event | None
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._run_stream_loop, symbols, level, stop_signal)
        except asyncio.CancelledError:
            if stop_signal:
                stop_signal.set()
            raise

    def _run_stream_loop(
        self, symbols: list[str], level: int, stop_signal: threading.Event | None = None
    ) -> None:
        bot = self._bot
        try:
            stream: Iterable[Any] | None = None
            try:
                stream = bot.broker.stream_orderbook(symbols, level=level)
            except Exception as exc:
                logger.warning("Orderbook stream unavailable, falling back to trades: %s", exc)
                try:
                    stream = bot.broker.stream_trades(symbols)
                except Exception as trade_exc:
                    logger.error("Failed to start streaming trades: %s", trade_exc)
                    return

            for msg in stream or []:
                if stop_signal and stop_signal.is_set():
                    break
                if not isinstance(msg, dict):
                    continue
                sym = str(msg.get("product_id") or msg.get("symbol") or "")
                if not sym:
                    continue
                mark: Decimal | None = None
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

                bot.strategy_coordinator.update_mark_window(sym, mark)
                monitor = getattr(bot, "market_monitor", self._market_monitor)
                try:
                    if monitor:
                        monitor.record_update(sym)
                    timestamp = utc_now()
                    risk_manager = bot.risk_manager
                    record_fn = getattr(risk_manager, "record_mark_update", None)
                    stored = record_fn(sym, timestamp) if callable(record_fn) else timestamp
                    risk_manager.last_mark_update[sym] = stored
                    emit_metric(
                        bot.event_store,
                        bot.bot_id,
                        {"event_type": "ws_mark_update", "symbol": sym, "mark": str(mark)},
                    )
                except Exception:
                    logger.exception("WS mark update bookkeeping failed for %s", sym)
        except Exception as exc:  # pragma: no cover - defensive error metric
            emit_metric(
                bot.event_store,
                bot.bot_id,
                {"event_type": "ws_stream_error", "message": str(exc)},
            )
        finally:
            emit_metric(
                bot.event_store,
                bot.bot_id,
                {"event_type": "ws_stream_exit"},
            )
