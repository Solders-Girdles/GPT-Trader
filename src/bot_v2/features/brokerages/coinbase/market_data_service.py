from __future__ import annotations

"""
Market data utilities for Coinbase: ticker cache and WS-backed service.

This module is optional and only used when explicitly enabled by the adapter.
It keeps an in-memory cache of bid/ask/last with timestamps and provides
basic staleness checks and a background streaming loop with reconnection
inherited from CoinbaseWebSocket.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Callable
import threading
import logging

from .ws import CoinbaseWebSocket, WSSubscription, normalize_market_message

logger = logging.getLogger(__name__)


@dataclass
class Ticker:
    bid: Decimal
    ask: Decimal
    last: Decimal
    ts: datetime


class TickerCache:
    """In-memory cache of latest tickers by symbol."""

    def __init__(self):
        self._data: Dict[str, Ticker] = {}
        self._lock = threading.RLock()

    def update(self, symbol: str, bid: Decimal, ask: Decimal, last: Decimal, ts: datetime) -> None:
        with self._lock:
            self._data[symbol] = Ticker(bid=bid, ask=ask, last=last, ts=ts)

    def get(self, symbol: str) -> Optional[Ticker]:
        with self._lock:
            return self._data.get(symbol)

    def is_stale(self, symbol: str, threshold_seconds: int = 5) -> bool:
        with self._lock:
            t = self._data.get(symbol)
            if not t:
                return True
            return (datetime.utcnow() - t.ts) > timedelta(seconds=threshold_seconds)


class CoinbaseTickerService:
    """WebSocket-backed ticker service with simple background loop.

    - Subscribes to ticker channel for provided symbols
    - Updates TickerCache on each message
    - Reconnect/backoff handled by CoinbaseWebSocket
    """

    def __init__(
        self,
        websocket_factory: Callable[[], CoinbaseWebSocket],
        symbols: List[str],
        cache: Optional[TickerCache] = None,
        on_update: Optional[Callable[[str, Ticker], None]] = None,
    ):
        self._ws_factory = websocket_factory
        self._symbols = symbols
        self.cache = cache or TickerCache()
        self.on_update = on_update
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _loop(self) -> None:
        try:
            ws = self._ws_factory()
            if hasattr(ws, 'connect'):
                ws.connect()
            ws.subscribe(WSSubscription(channels=["ticker"], product_ids=list(self._symbols)))
            for msg in ws.stream_messages():
                if self._stop.is_set():
                    break
                try:
                    norm = normalize_market_message(msg)
                    pid = norm.get('product_id') or norm.get('symbol')
                    if not pid:
                        continue
                    # Accept various field names for price fields
                    bid = norm.get('best_bid') or norm.get('bid') or norm.get('price')
                    ask = norm.get('best_ask') or norm.get('ask') or norm.get('price')
                    last = norm.get('last') or norm.get('price') or bid or ask
                    ts_str = norm.get('time') or norm.get('timestamp')
                    ts = datetime.fromisoformat(ts_str) if isinstance(ts_str, str) else datetime.utcnow()
                    if bid and ask and last:
                        self.cache.update(pid, bid, ask, last, ts)
                        if self.on_update:
                            try:
                                self.on_update(pid, self.cache.get(pid))  # type: ignore[arg-type]
                            except Exception:
                                pass
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Ticker loop ended: {e}")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="coinbase_ticker_ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

