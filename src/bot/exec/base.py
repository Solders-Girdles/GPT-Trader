from __future__ import annotations

from abc import ABC, abstractmethod


class Broker(ABC):
    @abstractmethod
    def submit_market_order(self, symbol: str, side: str, qty: int) -> None: ...
