from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from gpt_trader.app import run
from gpt_trader.domain import Bar
from gpt_trader.settings import Settings


class _RecorderMarketData:
    def __init__(self, bars: list[Bar]) -> None:
        self._bars = bars
        self.calls: list[tuple[str, int, str]] = []

    def bars(self, symbol: str, lookback: int, interval: str):
        self.calls.append((symbol, lookback, interval))
        return self._bars


def test_run_uses_supplied_market_data_parameters() -> None:
    stub_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 5, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("99"),
            close=Decimal("104"),
            volume=Decimal("1200"),
        )
    ]

    recorder = _RecorderMarketData(stub_bars)
    cfg = Settings(openai_api_key="sk-test")

    run(symbols=["AAPL"], cfg=cfg, lookback=42, interval="1h", market_data=recorder)

    assert recorder.calls == [("AAPL", 42, "1h")]
