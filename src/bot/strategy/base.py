from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str
    supports_short: bool = False

    @abstractmethod
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Produce model signals for downstream engines.

        Requirements
        - Return a DataFrame indexed like `bars` with at least a 'signal' column.
        - 'signal' semantics: 1 for buy/long, 0 for flat, -1 for sell/short.
        - If the strategy is long-only, set `supports_short=False` and never emit -1.
        - Indicators used by the engine (e.g., 'atr', 'donchian_upper') should be included
          when applicable to enable sizing and stop logic.
        """
