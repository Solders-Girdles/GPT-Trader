from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

@dataclass
class Ticker:
    symbol: str
    bid: float
    ask: float
    last: float
    ts: datetime

class TickerCache:
    def __init__(self, ttl_seconds: int = 5):
        self.ttl = ttl_seconds
        self._cache: Dict[str, Ticker] = {}

    def update(self, ticker: Ticker):
        self._cache[ticker.symbol] = ticker

    def get(self, symbol: str) -> Optional[Ticker]:
        return self._cache.get(symbol)

    def is_stale(self, symbol: str) -> bool:
        ticker = self.get(symbol)
        if not ticker:
            return True
        return (datetime.utcnow() - ticker.ts).total_seconds() > self.ttl

class CoinbaseTickerService:
    def __init__(self, symbols: List[str] = None):
        self._symbols = symbols or []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        # Mock thread start
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def set_symbols(self, symbols: List[str]):
        self._symbols = symbols

    def _run(self):
        pass

MarketDataService = CoinbaseTickerService
