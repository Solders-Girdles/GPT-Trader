"""Streaming service for real-time market data updates.

Extracted from PerpsBot to separate streaming responsibilities from orchestration.
Phase 2 of PerpsBot refactoring (2025-10-01).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.market_data_service import MarketDataService
    from bot_v2.orchestration.market_monitor import MarketActivityMonitor
    from bot_v2.persistence.event_store import EventStore

logger = logging.getLogger(__name__)


class StreamingService:
    """Manages WebSocket streaming for real-time mark price updates.

    This service encapsulates:
    - Background thread management for streaming
    - Orderbook/trades stream consumption
    - Real-time mark window updates (via MarketDataService)
    - Event logging and monitoring integration
    - Graceful shutdown and restart

    Design Notes:
    - Shares mark_lock and mark_windows with MarketDataService
    - Thread-safe start/stop operations
    - Fallback from orderbook to trades stream on error
    - All side effects preserved from original PerpsBot implementation
    """

    def __init__(
        self,
        symbols: list[str],
        broker: IBrokerage,
        market_data_service: MarketDataService,
        risk_manager: LiveRiskManager,
        event_store: EventStore,
        market_monitor: MarketActivityMonitor,
        bot_id: str = "perps_bot",
        stop_event: threading.Event | None = None,
        thread: threading.Thread | None = None,
    ) -> None:
        """Initialize streaming service.

        Args:
            symbols: Trading symbols to stream
            broker: Brokerage interface for streaming
            market_data_service: Market data service for mark window updates
            risk_manager: Risk manager to update timestamps
            event_store: Event store for metrics
            market_monitor: Market activity monitor for heartbeats
            bot_id: Bot identifier for event logging
            stop_event: Optional stop event (for testing)
            thread: Optional thread (for testing injection)
        """
        self.symbols = symbols
        self.broker = broker
        self.market_data_service = market_data_service
        self.risk_manager = risk_manager
        self.event_store = event_store
        self.market_monitor = market_monitor
        self.bot_id = bot_id

        # Thread management (allow injection for testing)
        self._ws_stop = stop_event
        self._ws_thread = thread

    def is_running(self) -> bool:
        """Check if streaming thread is running."""
        return (
            hasattr(self, "_ws_thread")
            and self._ws_thread is not None
            and self._ws_thread.is_alive()
        )

    def start(self, level: int | None = None) -> None:
        """Start background streaming thread.

        Args:
            level: Orderbook depth level (1-3), defaults to 1

        Side Effects:
        - Creates and starts daemon thread running _stream_loop
        - Creates stop event for shutdown signaling
        - Logs start message
        """
        if not self.symbols:
            logger.info("No symbols configured; skipping streaming start")
            return

        if self.is_running():
            logger.debug("Streaming already running; skipping duplicate start")
            return

        # Validate and default level
        if level is None:
            level = 1
        try:
            level = max(int(level), 1)
        except (TypeError, ValueError):
            logger.warning("Invalid streaming level %s; defaulting to 1", level)
            level = 1

        # Create stop event if not injected (testing)
        if self._ws_stop is None:
            try:
                self._ws_stop = threading.Event()
            except Exception:
                self._ws_stop = None

        # Create and start thread
        self._ws_thread = threading.Thread(
            target=self._stream_loop, args=(self.symbols, level), daemon=True
        )
        self._ws_thread.start()
        logger.info("Started WS streaming thread for symbols=%s level=%s", self.symbols, level)

    def stop(self) -> None:
        """Stop background streaming thread.

        Side Effects:
        - Sets stop event to signal thread shutdown
        - Joins thread with 2s timeout
        - Clears thread and event references
        - Logs errors but continues cleanup
        """
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

    def _stream_loop(self, symbols: list[str], level: int) -> None:
        """Main streaming loop (runs in background thread).

        Args:
            symbols: Symbols to stream
            level: Orderbook depth level

        Behavior:
        1. Try to open orderbook stream at specified level
        2. Fallback to trades stream if orderbook fails
        3. Process messages until stop event set
        4. Extract mark price from bid/ask or last trade
        5. Update mark window via MarketDataService
        6. Record metrics and update timestamps
        7. Log errors and exit events

        Side Effects (MUST preserve exactly):
        - Updates mark_windows via market_data_service._update_mark_window
        - Records market_monitor.record_update(symbol)
        - Updates risk_manager.last_mark_update[symbol]
        - Writes event_store metrics (ws_mark_update, ws_stream_error, ws_stream_exit)
        """
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
                # Check stop signal
                if hasattr(self, "_ws_stop") and self._ws_stop and self._ws_stop.is_set():
                    break

                if not isinstance(msg, dict):
                    continue

                # Extract symbol
                sym = str(msg.get("product_id") or msg.get("symbol") or "")
                if not sym:
                    continue

                # Extract mark price
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

                # Update mark window (thread-safe via MarketDataService)
                self.market_data_service._update_mark_window(sym, mark)

                # Update monitoring and metrics
                try:
                    self.market_monitor.record_update(sym)
                    self.risk_manager.last_mark_update[sym] = datetime.utcnow()
                    self.event_store.append_metric(
                        self.bot_id,
                        {"event_type": "ws_mark_update", "symbol": sym, "mark": str(mark)},
                    )
                except Exception:
                    logger.exception("WS mark update bookkeeping failed for %s", sym)

        except Exception as exc:
            # Log stream error
            try:
                self.event_store.append_metric(
                    self.bot_id,
                    {"event_type": "ws_stream_error", "message": str(exc)},
                )
            except Exception:
                logger.exception("Failed to record WS stream error metric")
        finally:
            # Log stream exit
            try:
                self.event_store.append_metric(self.bot_id, {"event_type": "ws_stream_exit"})
            except Exception:
                logger.exception("Failed to record WS stream exit metric")
