from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from bot_v2.features.brokerages.coinbase.market_data_features import RollingWindow
from bot_v2.features.brokerages.coinbase.utilities import MarkCache
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket, WSSubscription
from bot_v2.utilities.datetime_helpers import utc_now

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CoinbaseTicker:
    """Simple container for ticker snapshots."""

    symbol: str
    bid: Decimal | None
    ask: Decimal | None
    last: Decimal | None
    timestamp: datetime
    raw: dict[str, Any] | None = None


class TickerCache:
    """Thread-safe in-memory cache for Coinbase ticker snapshots."""

    def __init__(self, ttl_seconds: int = 5) -> None:
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, CoinbaseTicker] = {}
        self._lock = threading.RLock()

    def set(self, symbol: str, ticker: CoinbaseTicker) -> None:
        ticker.timestamp = utc_now()
        with self._lock:
            self._cache[symbol] = ticker
        logger.debug("TickerCache updated for %s", symbol)

    def get(self, symbol: str) -> CoinbaseTicker | None:
        with self._lock:
            return self._cache.get(symbol)

    def is_stale(self, symbol: str, ttl_seconds: int | None = None) -> bool:
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl_seconds
        snapshot = self.get(symbol)
        if snapshot is None:
            return True
        age = utc_now() - snapshot.timestamp
        return age.total_seconds() > ttl

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())


class CoinbaseTickerService:
    """Background WebSocket service that maintains ticker snapshots in memory."""

    def __init__(
        self,
        websocket_factory: Callable[[], CoinbaseWebSocket],
        symbols: Iterable[str] | None = None,
        cache: TickerCache | None = None,
        on_update: Callable[[str, CoinbaseTicker], None] | None = None,
        reconnect_delay: float = 5.0,
    ) -> None:
        self._websocket_factory = websocket_factory
        self._symbols: list[str] = list(symbols or [])
        self._cache = cache or TickerCache()
        self._on_update = on_update
        self._reconnect_delay = reconnect_delay
        self._stop_event = threading.Event()
        self._resubscribe_event = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._ws: CoinbaseWebSocket | None = None

    def set_symbols(self, symbols: Iterable[str]) -> None:
        with self._lock:
            self._symbols = list(symbols)
        self._resubscribe_event.set()
        logger.debug("TickerService symbols updated: %s", self._symbols)

    def ensure_started(self) -> None:
        if not self._thread or not self._thread.is_alive():
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="coinbase-ticker-service", daemon=True
            )
            self._thread.start()
            logger.info("CoinbaseTickerService thread started")

    def stop(self) -> None:
        self._stop_event.set()
        self._resubscribe_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._ws:
            try:
                self._ws.disconnect()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to disconnect Coinbase websocket", exc_info=True)
        logger.info("CoinbaseTickerService stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._ws = self._websocket_factory()
                if hasattr(self._ws, "connected") and not getattr(self._ws, "connected"):
                    self._ws.connect()
                self._subscribe()
                for message in self._stream_messages():
                    if self._stop_event.is_set():
                        break
                    self._handle_message(message)
                    if self._resubscribe_event.is_set():
                        self._resubscribe_event.clear()
                        self._subscribe()
            except Exception as exc:  # pragma: no cover - defensive reconnect
                logger.warning("CoinbaseTickerService stream error: %s", exc, exc_info=True)
                if self._stop_event.wait(self._reconnect_delay):
                    break
            finally:
                if self._ws:
                    try:
                        self._ws.disconnect()
                    except Exception:  # pragma: no cover - defensive
                        logger.debug("CoinbaseTickerService disconnect failed", exc_info=True)
                    self._ws = None

    def _stream_messages(self):
        if not self._ws:
            return []
        try:
            return self._ws.stream()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("CoinbaseTickerService stream() failed: %s", exc, exc_info=True)
            raise

    def _subscribe(self) -> None:
        if not self._ws:
            return
        with self._lock:
            symbols = [symbol for symbol in self._symbols]
        if not symbols:
            return
        try:
            self._ws.subscribe(WSSubscription(channels=["ticker"], product_ids=symbols))
            logger.info("Subscribed to Coinbase ticker stream for %d symbols", len(symbols))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to subscribe Coinbase ticker stream: %s", exc, exc_info=True)

    def _handle_message(self, message: dict[str, Any]) -> None:
        message_type = (message.get("type") or message.get("channel") or "").lower()
        if "ticker" not in message_type:
            return
        symbol = message.get("product_id") or message.get("symbol")
        if not symbol:
            return
        bid = self._safe_decimal(
            message.get("best_bid") or message.get("bid") or message.get("bid_price")
        )
        ask = self._safe_decimal(
            message.get("best_ask") or message.get("ask") or message.get("ask_price")
        )
        last = self._safe_decimal(
            message.get("price") or message.get("last") or message.get("close")
        )
        ticker = CoinbaseTicker(
            symbol=symbol, bid=bid, ask=ask, last=last, timestamp=utc_now(), raw=dict(message)
        )
        self._cache.set(symbol, ticker)
        if self._on_update:
            try:
                self._on_update(symbol, ticker)
            except Exception:  # pragma: no cover - callback defensive
                logger.debug("CoinbaseTickerService on_update callback failed", exc_info=True)

    @staticmethod
    def _safe_decimal(value: Any) -> Decimal | None:
        if value in (None, "", "null"):
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            logger.debug("Unable to parse decimal from value=%s", value)
            return None


class MarketDataService:
    """Maintains cached market data, rolling metrics, and mark prices."""

    def __init__(self) -> None:
        self._market_data: dict[str, dict[str, Any]] = {}
        self._rolling_windows: dict[str, dict[str, RollingWindow]] = {}
        self._mark_cache = MarkCache()

    @property
    def mark_cache(self) -> MarkCache:
        return self._mark_cache

    def initialise_symbols(self, symbols: Iterable[str]) -> None:
        for symbol in symbols:
            symbol = str(symbol)
            if symbol not in self._market_data:
                self._market_data[symbol] = {
                    "mid": Decimal("0"),
                    "spread_bps": 0.0,
                    "depth_l1": Decimal("0"),
                    "depth_l10": Decimal("0"),
                    "last_update": None,
                }
            if symbol not in self._rolling_windows:
                self._rolling_windows[symbol] = {
                    "vol_1m": RollingWindow(60),
                    "vol_5m": RollingWindow(300),
                }

    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._market_data

    def update_ticker(
        self,
        symbol: str,
        bid: Decimal | None,
        ask: Decimal | None,
        last: Decimal | None,
        timestamp: datetime,
    ) -> None:
        data = self._market_data.setdefault(symbol, {})
        if bid is not None and ask is not None:
            data["bid"] = bid
            data["ask"] = ask
            data["mid"] = (bid + ask) / 2
            if bid > 0:
                data["spread_bps"] = float((ask - bid) / bid * 10000)
        if last is not None:
            data["last"] = last
        data["last_update"] = timestamp

    def record_trade(self, symbol: str, size: Decimal, timestamp: datetime) -> None:
        windows = self._rolling_windows.get(symbol)
        if not windows:
            return
        for window in windows.values():
            window.add(float(size), timestamp)

    def update_depth(self, symbol: str, changes: Iterable[Iterable[str]]) -> None:
        bid_depth_usd = Decimal("0")
        ask_depth_usd = Decimal("0")
        bid_depth_l1_usd = Decimal("0")
        ask_depth_l1_usd = Decimal("0")

        bid_count = 0
        ask_count = 0

        for change in list(changes)[:10]:
            if len(change) < 3:
                continue
            side, price_str, size_str = change
            price = Decimal(price_str) if price_str else Decimal("0")
            size = Decimal(size_str) if size_str and size_str != "0" else Decimal("0")
            notional = price * size
            if side == "buy":
                bid_depth_usd += notional
                if bid_count == 0:
                    bid_depth_l1_usd = notional
                bid_count += 1
            elif side == "sell":
                ask_depth_usd += notional
                if ask_count == 0:
                    ask_depth_l1_usd = notional
                ask_count += 1

        data = self._market_data.setdefault(symbol, {})
        data["depth_l1"] = bid_depth_l1_usd + ask_depth_l1_usd
        data["depth_l10"] = bid_depth_usd + ask_depth_usd

    def is_stale(self, symbol: str, threshold_seconds: int = 10) -> bool:
        data = self._market_data.get(symbol)
        if not data:
            return True
        last_update = data.get("last_update")
        if not last_update:
            return True
        return (datetime.utcnow() - last_update).total_seconds() > threshold_seconds

    def get_cached_quote(self, symbol: str) -> dict[str, Any] | None:
        data = self._market_data.get(symbol)
        if not data:
            return None
        if not data.get("last_update"):
            return None
        return data

    def get_snapshot(self, symbol: str) -> dict[str, Any]:
        data = self._market_data.get(symbol)
        if not data:
            return {}
        snapshot = dict(data)
        windows = self._rolling_windows.get(symbol)
        if windows:
            snapshot.update(
                {
                    "vol_1m": windows.get("vol_1m", RollingWindow(60)).sum,
                    "vol_5m": windows.get("vol_5m", RollingWindow(300)).sum,
                }
            )
        return snapshot

    def set_mark(self, symbol: str, price: Decimal) -> None:
        self._mark_cache.set_mark(symbol, price)

    def get_mark(self, symbol: str) -> Decimal | None:
        return self._mark_cache.get_mark(symbol)

    def rolling_windows(self, symbol: str) -> dict[str, RollingWindow]:
        return self._rolling_windows.setdefault(symbol, {})
