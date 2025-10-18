from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd

from gpt_trader.data.yahoo import YahooMarketData


class _StubProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_historical_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        return self._frame


def test_bars_converts_dataframe_to_domain_bars() -> None:
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.5],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.5, 100.5, 101.5],
            "Close": [100.5, 101.5, 102.0],
            "Volume": [1_000, 2_000, 3_000],
        },
        index=pd.to_datetime(
            [
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 2, tzinfo=timezone.utc),
                datetime(2024, 1, 3, tzinfo=timezone.utc),
            ]
        ),
    )

    market_data = YahooMarketData(provider_factory=lambda: _StubProvider(frame))

    bars = list(market_data.bars("AAPL", lookback=2, interval="1d"))

    assert len(bars) == 2
    assert bars[0].symbol == "AAPL"
    assert bars[0].timestamp == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert bars[0].close == Decimal("101.5")
    assert bars[1].timestamp == datetime(2024, 1, 3, tzinfo=timezone.utc)
    assert bars[1].volume == Decimal("3000")


def test_bars_returns_empty_for_empty_frame() -> None:
    frame = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.to_datetime([]),
    )

    market_data = YahooMarketData(provider_factory=lambda: _StubProvider(frame))

    assert list(market_data.bars("AAPL", lookback=5, interval="1d")) == []
