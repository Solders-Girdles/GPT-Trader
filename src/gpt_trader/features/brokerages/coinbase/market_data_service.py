import threading
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Ticker:
    symbol: str
    bid: float
    ask: float
    last: float
    ts: datetime


class TickerCache:
    """Thread-safe cache for ticker data.

    WebSocket threads update this cache while trading threads read from it.
    All operations are protected by an RLock for safe concurrent access.
    """

    def __init__(self, ttl_seconds: int = 5):
        self.ttl = ttl_seconds
        self._lock = threading.RLock()
        self._cache: dict[str, Ticker] = {}

    def update(self, ticker: Ticker) -> None:
        """Update ticker data (called by WebSocket thread)."""
        with self._lock:
            self._cache[ticker.symbol] = ticker

    def get(self, symbol: str) -> Ticker | None:
        """Get ticker data (called by trading thread)."""
        with self._lock:
            return self._cache.get(symbol)

    def is_stale(self, symbol: str) -> bool:
        """Check if ticker data is stale (thread-safe)."""
        with self._lock:
            ticker = self._cache.get(symbol)
            if not ticker:
                return True
            return (datetime.utcnow() - ticker.ts).total_seconds() > self.ttl


class CoinbaseTickerService:
    def __init__(self, symbols: list[str] | None = None):
        self._symbols = symbols or []
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        # Mock thread start
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def set_symbols(self, symbols: list[str]) -> None:
        self._symbols = symbols

    def _run(self) -> None:
        pass

    def get_mark(self, symbol: str) -> float | None:
        # Assuming 'last' price is the mark for now, or mocked
        # Since this is mostly a stub/mock service in this context
        return None


MarketDataService = CoinbaseTickerService
