from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd


class HistoricalDataSource(ABC):
    @abstractmethod
    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Return DataFrame with at least ['Open','High','Low','Close','Volume'] indexed by date."""


class LiveDataStream(Protocol):
    def subscribe(self, symbols: list[str]) -> None: ...
    def next_tick(self) -> dict: ...  # placeholder for future real-time
