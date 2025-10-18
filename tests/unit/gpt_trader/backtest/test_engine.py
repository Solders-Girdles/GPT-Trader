from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from gpt_trader.backtest import Backtester
from gpt_trader.domain import Bar, Signal


def _bar(close: str, day: int) -> Bar:
    price = Decimal(close)
    timestamp = datetime(2024, 1, day, tzinfo=timezone.utc)
    return Bar(
        symbol="AAPL",
        timestamp=timestamp,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1000"),
    )


class _ToggleStrategy:
    def decide(self, bars: list[Bar]) -> Signal:
        symbol = bars[-1].symbol
        if len(bars) == 1:
            return Signal(symbol=symbol, action="BUY")
        if len(bars) < 3:
            return Signal(symbol=symbol, action="HOLD")
        return Signal(symbol=symbol, action="SELL")


def test_backtester_generates_trades() -> None:
    bars = [_bar("100", 1), _bar("105", 2), _bar("102", 3)]
    strategy = _ToggleStrategy()

    result = Backtester(strategy).run("AAPL", bars)

    assert result.symbol == "AAPL"
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry.close == Decimal("100")
    assert trade.exit.close == Decimal("102")
    assert trade.return_pct == Decimal("0.02")
    assert trade.hold_duration.total_seconds() == 172800.0
    assert result.cumulative_return == Decimal("0.02")
    assert result.average_trade_return == Decimal("0.02")
    assert result.total_trades == 1
    assert result.win_rate == Decimal("1")
    assert result.best_trade_return == Decimal("0.02")
    assert result.worst_trade_return == Decimal("0.02")
    assert result.max_drawdown == Decimal("0")
    assert result.equity_curve == [Decimal("1"), Decimal("1.02")]


def test_backtester_requires_bars() -> None:
    strategy = _ToggleStrategy()

    with pytest.raises(ValueError):
        Backtester(strategy).run("AAPL", [])
