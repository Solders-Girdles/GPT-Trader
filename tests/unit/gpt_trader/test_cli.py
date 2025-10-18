from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from click.testing import CliRunner

import gpt_trader.cli as cli
from gpt_trader.backtest import BacktestResult, Trade
from gpt_trader.domain import Bar, Signal
from gpt_trader.settings import Settings

runner = CliRunner()


def test_trade_reads_symbols_file(tmp_path, monkeypatch) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\n# comment\n\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run(*, symbols, cfg, lookback, interval, log_dir, **_kwargs):
        captured["symbols"] = symbols
        captured["cfg"] = cfg
        captured["lookback"] = lookback
        captured["interval"] = interval
        captured["log_dir"] = log_dir

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "_resolve_settings", lambda: Settings(openai_api_key="sk-test"))

    result = runner.invoke(
        cli.app,
        [
            "trade",
            "--symbols-file",
            str(symbols_file),
            "--lookback",
            "42",
            "--interval",
            "1h",
        ],
    )

    assert result.exit_code == 0
    assert captured["symbols"] == ["AAPL", "MSFT"]
    assert captured["lookback"] == 42
    assert captured["interval"] == "1h"
    assert captured["log_dir"] is None


def test_trade_accepts_log_dir_override(tmp_path, monkeypatch) -> None:
    log_dir = tmp_path / "logs"

    called: dict[str, object] = {}

    def fake_run(*, symbols, cfg, lookback, interval, log_dir, **_kwargs):
        called["log_dir"] = log_dir
        called["symbols"] = symbols

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "_resolve_settings", lambda: Settings(openai_api_key="sk-test"))

    result = runner.invoke(
        cli.app,
        [
            "trade",
            "--symbol",
            "BTC-USD",
            "--log-dir",
            str(log_dir),
        ],
    )

    assert result.exit_code == 0
    assert called["symbols"] == ["BTC-USD"]
    assert Path(called["log_dir"]) == log_dir


def test_backtest_command_runs(monkeypatch) -> None:
    runner = CliRunner()

    fake_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1100"),
        ),
    ]

    class FakeMarketData:
        def bars(self, symbol, lookback, interval):
            return fake_bars

    class FakeStrategy:
        def decide(self, bars):
            return Signal(symbol=bars[-1].symbol, action="HOLD")

    class FakeBacktester:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, bars):
            hold_duration = fake_bars[1].timestamp - fake_bars[0].timestamp
            return BacktestResult(
                symbol=symbol,
                trades=[
                    Trade(
                        entry=fake_bars[0],
                        exit=fake_bars[1],
                        return_pct=Decimal("0.05"),
                        hold_duration=hold_duration,
                        metadata={
                            "entry": {
                                "reason": "initial_position",
                                "confidence": 1.0,
                                "metadata": {"reason": "initial_position"},
                            },
                            "exit": {
                                "reason": "profit_target",
                                "confidence": 0.8,
                                "metadata": {"reason": "profit_target"},
                            },
                        },
                    )
                ],
                cumulative_return=Decimal("0.05"),
                average_trade_return=Decimal("0.05"),
                win_rate=Decimal("1"),
                total_trades=1,
                best_trade_return=Decimal("0.05"),
                worst_trade_return=Decimal("0.05"),
                max_drawdown=Decimal("0"),
                equity_curve=[Decimal("1"), Decimal("1.05")],
            )

    monkeypatch.setattr(cli, "YahooMarketData", lambda: FakeMarketData())
    monkeypatch.setattr(cli, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(cli, "Backtester", FakeBacktester)

    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--lookback",
            "10",
            "--interval",
            "1d",
            "--strategy",
            "ma",
        ],
    )

    assert result.exit_code == 0
    assert "AAPL backtest" in result.stdout
    assert "cum_return=5.00%" in result.stdout
    assert "avg_per_trade=5.00%" in result.stdout
    assert "win_rate=100.0%" in result.stdout
    assert "best=5.00%" in result.stdout
    assert "worst=5.00%" in result.stdout
    assert "max_drawdown=0.00%" in result.stdout
    assert "avg_hold=24.00h" in result.stdout
    assert "avg_conf=1.00" in result.stdout
    assert "last_reason=initial_position" in result.stdout
    assert "last_exit_reason=profit_target" in result.stdout


def test_backtest_command_writes_report(tmp_path, monkeypatch) -> None:
    fake_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1100"),
        ),
    ]

    class FakeMarketData:
        def bars(self, symbol, lookback, interval):
            return fake_bars

    class FakeStrategy:
        def decide(self, bars):
            return Signal(symbol=bars[-1].symbol, action="HOLD")

    class FakeBacktester:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, bars):
            hold_duration = fake_bars[1].timestamp - fake_bars[0].timestamp
            return BacktestResult(
                symbol=symbol,
                trades=[
                    Trade(
                        entry=fake_bars[0],
                        exit=fake_bars[1],
                        return_pct=Decimal("0.05"),
                        hold_duration=hold_duration,
                        metadata={
                            "entry": {
                                "reason": "initial_position",
                                "confidence": 1.0,
                                "metadata": {"reason": "initial_position"},
                            },
                            "exit": {
                                "reason": "profit_target",
                                "confidence": 0.8,
                                "metadata": {"reason": "profit_target"},
                            },
                        },
                    )
                ],
                cumulative_return=Decimal("0.05"),
                average_trade_return=Decimal("0.05"),
                win_rate=Decimal("1"),
                total_trades=1,
                best_trade_return=Decimal("0.05"),
                worst_trade_return=Decimal("0.05"),
                max_drawdown=Decimal("0"),
                equity_curve=[Decimal("1"), Decimal("1.05")],
            )

    monkeypatch.setattr(cli, "YahooMarketData", lambda: FakeMarketData())
    monkeypatch.setattr(cli, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(cli, "Backtester", FakeBacktester)

    output_path = tmp_path / "report.json"

    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--lookback",
            "10",
            "--interval",
            "1d",
            "--strategy",
            "ma",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    data = json.loads(output_path.read_text())
    assert data["symbol"] == "AAPL"
    assert data["total_trades"] == 1
    assert data["cumulative_return_pct"] == 5.0
    assert data["win_rate_pct"] == 100.0
    assert data["average_trade_return_pct"] == 5.0
    assert data["best_trade_return_pct"] == 5.0
    assert data["worst_trade_return_pct"] == 5.0
    assert data["max_drawdown_pct"] == 0.0
    assert data["equity_curve"] == [1.0, 1.05]
    assert data["equity_curve_timestamps"] == [
        fake_bars[0].timestamp.isoformat(),
        fake_bars[1].timestamp.isoformat(),
    ]
    assert len(data["trades"]) == 1
    trade = data["trades"][0]
    assert trade["hold_duration_seconds"] == 86400.0
    assert trade["metadata"]["entry"]["confidence"] == 1.0
    assert trade["metadata"]["exit"]["reason"] == "profit_target"
    assert data["summary"]["average_entry_confidence"] == 1.0
    assert data["summary"]["last_entry_reason"] == "initial_position"
    assert data["summary"]["last_exit_reason"] == "profit_target"


def test_backtest_plot_option_invokes_helper(tmp_path, monkeypatch) -> None:
    fake_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1100"),
        ),
    ]

    class FakeMarketData:
        def bars(self, symbol, lookback, interval):
            return fake_bars

    class FakeStrategy:
        def decide(self, bars):
            return Signal(symbol=bars[-1].symbol, action="HOLD")

    class FakeBacktester:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, bars):
            hold_duration = fake_bars[1].timestamp - fake_bars[0].timestamp
            return BacktestResult(
                symbol=symbol,
                trades=[
                    Trade(
                        entry=fake_bars[0],
                        exit=fake_bars[1],
                        return_pct=Decimal("0.05"),
                        hold_duration=hold_duration,
                        metadata={
                            "entry": {
                                "reason": "initial_position",
                                "confidence": 1.0,
                                "metadata": {"reason": "initial_position"},
                            },
                            "exit": {
                                "reason": "profit_target",
                                "confidence": 0.8,
                                "metadata": {"reason": "profit_target"},
                            },
                        },
                    )
                ],
                cumulative_return=Decimal("0.05"),
                average_trade_return=Decimal("0.05"),
                win_rate=Decimal("1"),
                total_trades=1,
                best_trade_return=Decimal("0.05"),
                worst_trade_return=Decimal("0.05"),
                max_drawdown=Decimal("0"),
                equity_curve=[Decimal("1"), Decimal("1.05")],
            )

    called: dict[str, object] = {}

    def fake_plot(levels, timestamps, path, *, log_scale, benchmarks):
        called["levels"] = levels
        called["timestamps"] = timestamps
        called["path"] = path
        called["log"] = log_scale
        called["benchmark"] = benchmarks

    monkeypatch.setattr(cli, "YahooMarketData", lambda: FakeMarketData())
    monkeypatch.setattr(cli, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(cli, "Backtester", FakeBacktester)
    monkeypatch.setattr(cli, "_plot_equity_curve", fake_plot)

    plot_path = tmp_path / "curve.png"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--lookback",
            "10",
            "--interval",
            "1d",
            "--strategy",
            "ma",
            "--plot",
            str(plot_path),
        ],
    )

    assert result.exit_code == 0
    assert called["path"] == plot_path
    assert called["levels"] == [Decimal("1"), Decimal("1.05")]
    assert called["log"] is False
    assert called["benchmark"] == []
    assert called["timestamps"] == [fake_bars[0].timestamp, fake_bars[1].timestamp]


def test_backtest_plot_with_log_and_benchmark(tmp_path, monkeypatch) -> None:
    fake_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1100"),
        ),
    ]

    class FakeMarketData:
        def bars(self, symbol, lookback, interval):
            return fake_bars

    class FakeStrategy:
        def decide(self, bars):
            return Signal(symbol=bars[-1].symbol, action="HOLD")

    class FakeBacktester:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, bars):
            hold_duration = fake_bars[1].timestamp - fake_bars[0].timestamp
            return BacktestResult(
                symbol=symbol,
                trades=[
                    Trade(
                        entry=fake_bars[0],
                        exit=fake_bars[1],
                        return_pct=Decimal("0.05"),
                        hold_duration=hold_duration,
                        metadata={
                            "entry": {
                                "reason": "initial_position",
                                "confidence": 1.0,
                                "metadata": {"reason": "initial_position"},
                            },
                            "exit": {
                                "reason": "profit_target",
                                "confidence": 0.8,
                                "metadata": {"reason": "profit_target"},
                            },
                        },
                    )
                ],
                cumulative_return=Decimal("0.05"),
                average_trade_return=Decimal("0.05"),
                win_rate=Decimal("1"),
                total_trades=1,
                best_trade_return=Decimal("0.05"),
                worst_trade_return=Decimal("0.05"),
                max_drawdown=Decimal("0"),
                equity_curve=[Decimal("1"), Decimal("1.05")],
            )

    bench_file = tmp_path / "bench.csv"
    bench_file.write_text(
        "2024-01-01T00:00:00,1\n2024-01-02T00:00:00,1.02\n",
        encoding="utf-8",
    )

    called: dict[str, object] = {}

    def fake_plot(levels, timestamps, path, *, log_scale, benchmarks):
        called["levels"] = levels
        called["timestamps"] = timestamps
        called["path"] = path
        called["log"] = log_scale
        called["benchmark"] = benchmarks

    monkeypatch.setattr(cli, "YahooMarketData", lambda: FakeMarketData())
    monkeypatch.setattr(cli, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(cli, "Backtester", FakeBacktester)
    monkeypatch.setattr(cli, "_plot_equity_curve", fake_plot)

    plot_path = tmp_path / "curve.png"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--lookback",
            "10",
            "--interval",
            "1d",
            "--strategy",
            "ma",
            "--plot",
            str(plot_path),
            "--plot-log",
            "--benchmark-csv",
            str(bench_file),
            "--benchmark-start",
            "2023-12-31T00:00:00",
        ],
    )

    assert result.exit_code == 0
    assert called["path"] == plot_path
    assert called["levels"] == [Decimal("1"), Decimal("1.05")]
    assert called["log"] is True
    assert called["benchmark"] == [[1.0, 1.02]]
    assert called["timestamps"] == [fake_bars[0].timestamp, fake_bars[1].timestamp]


def test_backtest_plot_with_benchmark_start_trims(tmp_path, monkeypatch) -> None:
    fake_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1100"),
        ),
    ]

    class FakeMarketData:
        def bars(self, symbol, lookback, interval):
            return fake_bars

    class FakeStrategy:
        def decide(self, bars):
            return Signal(symbol=bars[-1].symbol, action="HOLD")

    class FakeBacktester:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, bars):
            hold_duration = fake_bars[1].timestamp - fake_bars[0].timestamp
            return BacktestResult(
                symbol=symbol,
                trades=[
                    Trade(
                        entry=fake_bars[0],
                        exit=fake_bars[1],
                        return_pct=Decimal("0.05"),
                        hold_duration=hold_duration,
                        metadata={
                            "entry": {
                                "reason": "initial_position",
                                "confidence": 1.0,
                                "metadata": {"reason": "initial_position"},
                            },
                            "exit": {
                                "reason": "profit_target",
                                "confidence": 0.8,
                                "metadata": {"reason": "profit_target"},
                            },
                        },
                    )
                ],
                cumulative_return=Decimal("0.05"),
                average_trade_return=Decimal("0.05"),
                win_rate=Decimal("1"),
                total_trades=1,
                best_trade_return=Decimal("0.05"),
                worst_trade_return=Decimal("0.05"),
                max_drawdown=Decimal("0"),
                equity_curve=[Decimal("1"), Decimal("1.05")],
            )

    called: dict[str, object] = {}

    def fake_plot(levels, timestamps, path, *, log_scale, benchmarks):
        called["levels"] = levels
        called["timestamps"] = timestamps
        called["benchmarks"] = benchmarks

    monkeypatch.setattr(cli, "YahooMarketData", lambda: FakeMarketData())
    monkeypatch.setattr(cli, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(cli, "Backtester", FakeBacktester)
    monkeypatch.setattr(cli, "_plot_equity_curve", fake_plot)

    plot_path = tmp_path / "curve.png"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--lookback",
            "10",
            "--interval",
            "1d",
            "--strategy",
            "ma",
            "--plot",
            str(plot_path),
            "--benchmark-start",
            "2024-01-02T00:00:00",
        ],
    )

    assert result.exit_code == 0
    assert called["levels"] == [Decimal("1.05")]
    assert called["timestamps"] == [fake_bars[1].timestamp]


def test_backtest_equity_csv(tmp_path, monkeypatch) -> None:
    fake_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1100"),
        ),
    ]

    class FakeMarketData:
        def bars(self, symbol, lookback, interval):
            return fake_bars

    class FakeStrategy:
        def decide(self, bars):
            return Signal(symbol=bars[-1].symbol, action="HOLD")

    class FakeBacktester:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, bars):
            hold_duration = fake_bars[1].timestamp - fake_bars[0].timestamp
            return BacktestResult(
                symbol=symbol,
                trades=[
                    Trade(
                        entry=fake_bars[0],
                        exit=fake_bars[1],
                        return_pct=Decimal("0.05"),
                        hold_duration=hold_duration,
                        metadata={
                            "entry": {
                                "reason": "initial_position",
                                "confidence": 1.0,
                                "metadata": {"reason": "initial_position"},
                            },
                            "exit": {
                                "reason": "profit_target",
                                "confidence": 0.8,
                                "metadata": {"reason": "profit_target"},
                            },
                        },
                    )
                ],
                cumulative_return=Decimal("0.05"),
                average_trade_return=Decimal("0.05"),
                win_rate=Decimal("1"),
                total_trades=1,
                best_trade_return=Decimal("0.05"),
                worst_trade_return=Decimal("0.05"),
                max_drawdown=Decimal("0"),
                equity_curve=[Decimal("1"), Decimal("1.05")],
            )

    monkeypatch.setattr(cli, "YahooMarketData", lambda: FakeMarketData())
    monkeypatch.setattr(cli, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(cli, "Backtester", FakeBacktester)

    equity_path = tmp_path / "curve.csv"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--lookback",
            "10",
            "--interval",
            "1d",
            "--strategy",
            "ma",
            "--equity-csv",
            str(equity_path),
        ],
    )

    assert result.exit_code == 0
    lines = equity_path.read_text().strip().splitlines()
    assert lines[0] == "index,equity"
    assert lines[1] == "0,1.0"
    assert lines[2] == "1,1.05"


def test_backtest_trades_csv(tmp_path, monkeypatch) -> None:
    fake_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1100"),
        ),
    ]

    class FakeMarketData:
        def bars(self, symbol, lookback, interval):
            return fake_bars

    class FakeStrategy:
        def decide(self, bars):
            return Signal(symbol=bars[-1].symbol, action="HOLD")

    class FakeBacktester:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, bars):
            hold_duration = fake_bars[1].timestamp - fake_bars[0].timestamp
            return BacktestResult(
                symbol=symbol,
                trades=[
                    Trade(
                        entry=fake_bars[0],
                        exit=fake_bars[1],
                        return_pct=Decimal("0.05"),
                        hold_duration=hold_duration,
                        metadata={
                            "entry": {
                                "reason": "initial_position",
                                "confidence": 0.75,
                                "metadata": {"reason": "initial_position"},
                            },
                            "exit": {
                                "reason": "profit_target",
                                "confidence": 0.9,
                                "metadata": {"reason": "profit_target"},
                            },
                        },
                    )
                ],
                cumulative_return=Decimal("0.05"),
                average_trade_return=Decimal("0.05"),
                win_rate=Decimal("1"),
                total_trades=1,
                best_trade_return=Decimal("0.05"),
                worst_trade_return=Decimal("0.05"),
                max_drawdown=Decimal("0"),
                equity_curve=[Decimal("1"), Decimal("1.05")],
            )

    monkeypatch.setattr(cli, "YahooMarketData", lambda: FakeMarketData())
    monkeypatch.setattr(cli, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(cli, "Backtester", FakeBacktester)

    trades_path = tmp_path / "trades.csv"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--lookback",
            "10",
            "--interval",
            "1d",
            "--strategy",
            "ma",
            "--trades-csv",
            str(trades_path),
        ],
    )

    assert result.exit_code == 0
    rows = trades_path.read_text().strip().splitlines()
    assert (
        rows[0]
        == "entry_timestamp,exit_timestamp,entry_close,exit_close,return_pct,hold_duration_seconds,entry_reason,exit_reason,entry_confidence"
    )
    assert rows[1].endswith(",0.75")


def test_backtest_quiet_suppresses_summary(tmp_path, monkeypatch) -> None:
    fake_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1100"),
        ),
    ]

    class FakeMarketData:
        def bars(self, symbol, lookback, interval):
            return fake_bars

    class FakeStrategy:
        def decide(self, bars):
            return Signal(symbol=bars[-1].symbol, action="HOLD")

    class FakeBacktester:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, bars):
            hold_duration = fake_bars[1].timestamp - fake_bars[0].timestamp
            return BacktestResult(
                symbol=symbol,
                trades=[
                    Trade(
                        entry=fake_bars[0],
                        exit=fake_bars[1],
                        return_pct=Decimal("0.05"),
                        hold_duration=hold_duration,
                        metadata={
                            "entry": {
                                "reason": "initial_position",
                                "confidence": 0.75,
                                "metadata": {"reason": "initial_position"},
                            },
                            "exit": {
                                "reason": "profit_target",
                                "confidence": 0.9,
                                "metadata": {"reason": "profit_target"},
                            },
                        },
                    )
                ],
                cumulative_return=Decimal("0.05"),
                average_trade_return=Decimal("0.05"),
                win_rate=Decimal("1"),
                total_trades=1,
                best_trade_return=Decimal("0.05"),
                worst_trade_return=Decimal("0.05"),
                max_drawdown=Decimal("0"),
                equity_curve=[Decimal("1"), Decimal("1.05")],
            )

    monkeypatch.setattr(cli, "YahooMarketData", lambda: FakeMarketData())
    monkeypatch.setattr(cli, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(cli, "Backtester", FakeBacktester)

    output_path = tmp_path / "report.json"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--lookback",
            "10",
            "--interval",
            "1d",
            "--strategy",
            "ma",
            "--output",
            str(output_path),
            "--quiet",
        ],
    )

    assert result.exit_code == 0
    assert "backtest |" not in result.stdout
    assert "Wrote backtest report" in result.stdout


def test_backtest_benchmark_align_pad(tmp_path, monkeypatch) -> None:
    fake_bars = [
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
        Bar(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
            open=Decimal("101"),
            high=Decimal("102"),
            low=Decimal("100"),
            close=Decimal("105"),
            volume=Decimal("1100"),
        ),
    ]

    class FakeMarketData:
        def bars(self, symbol, lookback, interval):
            return fake_bars

    class FakeStrategy:
        def decide(self, bars):
            return Signal(symbol=bars[-1].symbol, action="HOLD")

    class FakeBacktester:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, bars):
            hold_duration = fake_bars[1].timestamp - fake_bars[0].timestamp
            return BacktestResult(
                symbol=symbol,
                trades=[
                    Trade(
                        entry=fake_bars[0],
                        exit=fake_bars[1],
                        return_pct=Decimal("0.05"),
                        hold_duration=hold_duration,
                        metadata={
                            "entry": {
                                "reason": "initial_position",
                                "confidence": 1.0,
                                "metadata": {"reason": "initial_position"},
                            },
                            "exit": {
                                "reason": "profit_target",
                                "confidence": 0.8,
                                "metadata": {"reason": "profit_target"},
                            },
                        },
                    )
                ],
                cumulative_return=Decimal("0.05"),
                average_trade_return=Decimal("0.05"),
                win_rate=Decimal("1"),
                total_trades=1,
                best_trade_return=Decimal("0.05"),
                worst_trade_return=Decimal("0.05"),
                max_drawdown=Decimal("0"),
                equity_curve=[Decimal("1"), Decimal("1.05")],
            )

    bench = tmp_path / "bench.csv"
    bench.write_text("2024-01-01T00:00:00,1\n", encoding="utf-8")

    called: dict[str, object] = {}

    def fake_plot(levels, timestamps, path, *, log_scale, benchmarks):
        called["benchmarks"] = benchmarks

    monkeypatch.setattr(cli, "YahooMarketData", lambda: FakeMarketData())
    monkeypatch.setattr(cli, "get_strategy", lambda name: FakeStrategy())
    monkeypatch.setattr(cli, "Backtester", FakeBacktester)
    monkeypatch.setattr(cli, "_plot_equity_curve", fake_plot)

    plot_path = tmp_path / "plot.png"
    result = runner.invoke(
        cli.app,
        [
            "backtest",
            "--symbol",
            "AAPL",
            "--lookback",
            "10",
            "--interval",
            "1d",
            "--strategy",
            "ma",
            "--plot",
            str(plot_path),
            "--benchmark-csv",
            str(bench),
            "--benchmark-align",
            "pad",
        ],
    )

    assert result.exit_code == 0
    assert called["benchmarks"] == [[1.0, 1.0]]
