from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from click.testing import CliRunner

import gpt_trader.cli as cli
from gpt_trader.backtest import BacktestResult, Trade
from gpt_trader.domain import Bar, Signal


def _bar(price: float, ts: datetime) -> Bar:
    value = Decimal(str(price))
    return Bar(
        symbol="AAPL",
        timestamp=ts,
        open=value,
        high=value,
        low=value,
        close=value,
        volume=Decimal("1000"),
    )


class _FakeMarketData:
    def __init__(self, bars: list[Bar]) -> None:
        self._bars = bars

    def bars(self, symbol: str, lookback: int, interval: str) -> list[Bar]:
        return self._bars


class _ConstantSignalStrategy:
    def __init__(self, action: str = "HOLD") -> None:
        self.action = action

    def decide(self, bars: list[Bar]) -> Signal:
        return Signal(symbol=bars[-1].symbol, action=self.action)


class _FakeBacktester:
    def __init__(self, trade: Trade) -> None:
        self._trade = trade

    def run(self, symbol: str, bars: list[Bar]) -> BacktestResult:
        return BacktestResult(
            symbol=symbol,
            trades=[self._trade],
            cumulative_return=self._trade.return_pct,
            average_trade_return=self._trade.return_pct,
            win_rate=Decimal("1"),
            total_trades=1,
            best_trade_return=self._trade.return_pct,
            worst_trade_return=self._trade.return_pct,
            max_drawdown=Decimal("0"),
            equity_curve=[Decimal("1"), Decimal("1") + self._trade.return_pct],
        )


def _basic_trade() -> Trade:
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    entry = _bar(100, ts)
    exit_bar = _bar(103, ts + timedelta(days=1))  # type: ignore[name-defined]
    return Trade(
        entry=entry,
        exit=exit_bar,
        return_pct=Decimal("0.03"),
        hold_duration=exit_bar.timestamp - entry.timestamp,
        metadata={
            "entry": {"reason": "short_ma_cross_above_long_ma", "confidence": 0.75},
            "exit": {"reason": "profit_target", "confidence": 0.9},
        },
    )


def test_backtest_quiet_and_json(tmp_path, monkeypatch) -> None:
    trade = _basic_trade()
    bars = [
        _bar(100, datetime(2024, 1, 1, tzinfo=timezone.utc)),
        _bar(103, datetime(2024, 1, 2, tzinfo=timezone.utc)),
    ]
    runner = CliRunner()

    monkeypatch.setattr(cli, "YahooMarketData", lambda: _FakeMarketData(bars))
    monkeypatch.setattr(cli, "get_strategy", lambda name: _ConstantSignalStrategy("HOLD"))
    monkeypatch.setattr(cli, "Backtester", lambda base: _FakeBacktester(trade))

    output = tmp_path / "report.json"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--strategy",
            "ma-crossover",
            "--quiet",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "backtest |" not in result.stdout
    data = json.loads(output.read_text())
    assert data["summary"]["average_entry_confidence"] == 0.75
    assert data["summary"]["last_entry_reason"] == "short_ma_cross_above_long_ma"
    assert data["summary"]["last_exit_reason"] == "profit_target"


def test_backtest_trades_csv(tmp_path, monkeypatch) -> None:
    trade = _basic_trade()
    bars = [
        _bar(100, datetime(2024, 1, 1, tzinfo=timezone.utc)),
        _bar(103, datetime(2024, 1, 2, tzinfo=timezone.utc)),
    ]
    runner = CliRunner()

    monkeypatch.setattr(cli, "YahooMarketData", lambda: _FakeMarketData(bars))
    monkeypatch.setattr(cli, "get_strategy", lambda name: _ConstantSignalStrategy("HOLD"))
    monkeypatch.setattr(cli, "Backtester", lambda base: _FakeBacktester(trade))

    trades_csv = tmp_path / "trades.csv"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--strategy",
            "ma-crossover",
            "--trades-csv",
            str(trades_csv),
        ],
    )

    assert result.exit_code == 0
    rows = trades_csv.read_text().strip().splitlines()
    assert rows[0].split(",")[-1] == "entry_confidence"
    assert rows[1].split(",")[-1] == "0.75"


def test_backtest_plot_with_start(tmp_path, monkeypatch) -> None:
    trade = _basic_trade()
    bars = [
        _bar(100, datetime(2024, 1, 1, tzinfo=timezone.utc)),
        _bar(103, datetime(2024, 1, 2, tzinfo=timezone.utc)),
    ]
    runner = CliRunner()

    called: dict[str, object] = {}

    def fake_plot(levels, timestamps, path, *, log_scale, benchmarks):
        called["levels"] = levels
        called["timestamps"] = timestamps
        called["benchmarks"] = benchmarks

    monkeypatch.setattr(cli, "YahooMarketData", lambda: _FakeMarketData(bars))
    monkeypatch.setattr(cli, "get_strategy", lambda name: _ConstantSignalStrategy("HOLD"))
    monkeypatch.setattr(cli, "Backtester", lambda base: _FakeBacktester(trade))
    monkeypatch.setattr(cli, "_plot_equity_curve", fake_plot)

    plot_path = tmp_path / "plot.png"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--strategy",
            "ma-crossover",
            "--plot",
            str(plot_path),
            "--benchmark-start",
            "2024-01-02T00:00:00",
        ],
    )

    assert result.exit_code == 0
    assert called["levels"] == [Decimal("1.03")]
    assert len(called["timestamps"]) == 1
