"""Click-based CLI entry point for the ``gpt_trader`` package."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import click

from gpt_trader.backtest import Backtester, BacktestResult, Trade
from gpt_trader.data import YahooMarketData
from gpt_trader.strategy import get_strategy

from .app import run
from .logging import configure as configure_logging
from .settings import Settings, get_settings

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(
    context_settings=CONTEXT_SETTINGS,
    help="Command-line interface for the next-generation GPT-Trader stack.",
)
def app() -> None:
    """CLI root group."""


@app.command("trade")
@click.option(
    "--symbol",
    "-s",
    multiple=True,
    metavar="SYMBOL",
    help="Symbol to trade; repeat the option to target multiple symbols.",
)
@click.option(
    "--symbols-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a file containing newline-delimited symbols.",
)
@click.option(
    "--lookback",
    "-l",
    default=120,
    show_default=True,
    type=click.IntRange(1),
    help="Number of trailing bars to retrieve per symbol.",
)
@click.option(
    "--interval",
    "-i",
    default="1d",
    show_default=True,
    help="Market data interval (e.g. 1m, 15m, 1h, 1d).",
)
@click.option(
    "--log-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Override the directory used for logs and reports.",
)
def trade(
    symbol: tuple[str, ...],
    symbols_file: Path | None,
    lookback: int,
    interval: str,
    log_dir: Path | None,
) -> None:
    """Execute a trading run for the selected symbols."""
    targets = _compose_symbols(symbol, symbols_file)
    run(
        symbols=targets,
        cfg=_resolve_settings(),
        lookback=lookback,
        interval=interval,
        log_dir=log_dir,
    )


@app.command("backtest")
@click.option(
    "--symbol",
    required=True,
    metavar="SYMBOL",
    help="Symbol to backtest.",
)
@click.option(
    "--lookback",
    "-l",
    default=300,
    show_default=True,
    type=click.IntRange(1),
    help="Number of trailing bars to retrieve.",
)
@click.option(
    "--interval",
    "-i",
    default="1d",
    show_default=True,
    help="Market data interval (e.g. 1m, 1h, 1d).",
)
@click.option(
    "--strategy",
    "-S",
    "strategy_name",
    default="ma-crossover",
    show_default=True,
    help="Strategy identifier (built-ins: ma-crossover, buy-and-hold).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Optional path to write a JSON report.",
)
@click.option(
    "--plot",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Optional path to write an equity curve plot (requires matplotlib).",
)
@click.option(
    "--plot-log/--plot-linear",
    default=False,
    show_default=True,
    help="Render equity curve on a logarithmic scale when plotting.",
)
@click.option(
    "--benchmark-csv",
    "benchmark_csvs",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    multiple=True,
    help="Optional CSV(s) with benchmark equity values (index,value) to overlay on the plot.",
)
@click.option(
    "--benchmark-start",
    type=str,
    help="ISO timestamp to trim equity curve and benchmarks when plotting.",
)
@click.option(
    "--benchmark-align",
    type=click.Choice(["trim", "pad"], case_sensitive=False),
    default="trim",
    show_default=True,
    help="How to align benchmarks when lengths differ (trim to shortest or pad with last value).",
)
@click.option(
    "--equity-csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Optional path to write equity curve samples as CSV (index,value).",
)
@click.option(
    "--trades-csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Optional path to write trade summary CSV.",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress console summary output (artifact exports still run).",
)
def backtest(
    symbol: str,
    lookback: int,
    interval: str,
    strategy_name: str,
    output: Path | None,
    plot: Path | None,
    plot_log: bool,
    benchmark_csvs: tuple[Path, ...],
    benchmark_start: str | None,
    equity_csv: Path | None,
    trades_csv: Path | None,
    quiet: bool,
    benchmark_align: str,
) -> None:
    """Run a deterministic backtest for ``symbol``."""
    market_data = YahooMarketData()
    bars = list(market_data.bars(symbol, lookback=lookback, interval=interval))
    if not bars:
        raise click.ClickException(
            f"No market data returned for symbol={symbol} interval={interval} lookback={lookback}"
        )

    try:
        strategy = get_strategy(strategy_name)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param="strategy_name", param_hint="--strategy") from exc

    result = Backtester(strategy).run(symbol, bars)
    cumulative_percent = float(result.cumulative_return * 100)
    average_percent = float(result.average_trade_return * 100)
    win_rate_percent = float(result.win_rate * 100)
    best_trade_percent = float(result.best_trade_return * 100)
    worst_trade_percent = float(result.worst_trade_return * 100)
    max_drawdown_percent = float(result.max_drawdown * 100)
    if result.trades:
        avg_hold_seconds = (
            sum(trade.hold_duration.total_seconds() for trade in result.trades)
            / result.total_trades
        )
    else:
        avg_hold_seconds = 0.0
    avg_hold_hours = avg_hold_seconds / 3600 if avg_hold_seconds else 0.0

    eq_timestamps = _equity_timestamps(result.trades)
    avg_confidence, latest_entry_reason, latest_exit_reason = _aggregate_trade_context(
        result.trades
    )

    summary = (
        f"{symbol} backtest | trades={result.total_trades} | "
        f"cum_return={cumulative_percent:.2f}% | "
        f"avg_per_trade={average_percent:.2f}% | "
        f"win_rate={win_rate_percent:.1f}% | "
        f"best={best_trade_percent:.2f}% | "
        f"worst={worst_trade_percent:.2f}% | "
        f"max_drawdown={max_drawdown_percent:.2f}% | "
        f"avg_hold={avg_hold_hours:.2f}h"
    )
    if avg_confidence is not None:
        summary += f" | avg_conf={avg_confidence:.2f}"
    if latest_entry_reason:
        summary += f" | last_reason={latest_entry_reason}"
    if latest_exit_reason:
        summary += f" | last_exit_reason={latest_exit_reason}"

    if not quiet:
        click.echo(summary)

    eq_timestamps_serial = eq_timestamps if eq_timestamps else None

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        report_dict = _serialise_backtest(
            result,
            eq_timestamps_serial,
            avg_confidence=avg_confidence,
            last_entry_reason=latest_entry_reason,
            last_exit_reason=latest_exit_reason,
        )
        output.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        click.echo(f"Wrote backtest report to {output}")

    if equity_csv:
        equity_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_equity_csv(result.equity_curve, equity_csv)
        click.echo(f"Wrote equity curve CSV to {equity_csv}")

    if trades_csv:
        trades_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_trades_csv(result.trades, trades_csv)
        click.echo(f"Wrote trades CSV to {trades_csv}")

    plot_levels = list(result.equity_curve)
    plot_timestamps = list(eq_timestamps) if eq_timestamps else None

    benchmark_start_dt: datetime | None = None
    if benchmark_start:
        benchmark_start_dt = _parse_timestamp(benchmark_start)
        if benchmark_start_dt is None:
            raise click.ClickException(f"Could not parse --benchmark-start: {benchmark_start}")

    if benchmark_start_dt and plot_timestamps:
        start_epoch = _to_epoch(benchmark_start_dt)
        eq_epochs = [_to_epoch(ts) for ts in plot_timestamps]
        start_idx = next((i for i, epoch in enumerate(eq_epochs) if epoch >= start_epoch), None)
        if start_idx is not None and start_idx > 0:
            plot_timestamps = plot_timestamps[start_idx:]
            plot_levels = plot_levels[start_idx:]
        if not plot_levels:
            plot_levels = [result.equity_curve[-1]]
            plot_timestamps = [eq_timestamps[-1]] if eq_timestamps else None

    if plot:
        plot.parent.mkdir(parents=True, exist_ok=True)
        benchmarks = [_load_benchmark(path) for path in benchmark_csvs]
        aligned = _align_benchmarks(
            benchmarks, len(plot_levels), plot_timestamps, mode=benchmark_align
        )
        _plot_equity_curve(
            plot_levels, plot_timestamps, plot, log_scale=plot_log, benchmarks=aligned
        )
        click.echo(f"Wrote equity curve plot to {plot}")


def _compose_symbols(cli_symbols: Sequence[str], symbols_file: Path | None) -> Sequence[str] | None:
    if symbols_file is None:
        return list(cli_symbols) if cli_symbols else None

    contents = symbols_file.read_text(encoding="utf-8").splitlines()
    parsed = [
        line.strip() for line in contents if line.strip() and not line.strip().startswith("#")
    ]
    parsed.extend(cli_symbols)
    return parsed or None


def _resolve_settings() -> Settings:
    """Allow dependency injection from tests without global mutation."""
    return get_settings()


def _aggregate_trade_context(
    trades: Sequence[Trade],
) -> tuple[float | None, str | None, str | None]:
    if not trades:
        return None, None, None

    confidences: list[float] = []
    entry_reasons: list[str] = []
    exit_reasons: list[str] = []

    for trade in trades:
        meta = trade.metadata or {}
        entry_meta = meta.get("entry", {}) if isinstance(meta.get("entry"), dict) else {}
        exit_meta = meta.get("exit", {}) if isinstance(meta.get("exit"), dict) else {}

        conf = entry_meta.get("confidence")
        if isinstance(conf, (float, int)):
            confidences.append(float(conf))

        entry_reason = entry_meta.get("reason")
        if isinstance(entry_reason, str) and entry_reason:
            entry_reasons.append(entry_reason)

        exit_reason = exit_meta.get("reason")
        if isinstance(exit_reason, str) and exit_reason:
            exit_reasons.append(exit_reason)

    avg_conf = sum(confidences) / len(confidences) if confidences else None
    last_entry = entry_reasons[-1] if entry_reasons else None
    last_exit = exit_reasons[-1] if exit_reasons else None
    return avg_conf, last_entry, last_exit


def _serialise_backtest(
    result: BacktestResult,
    timestamps: Sequence[datetime] | None,
    *,
    avg_confidence: float | None,
    last_entry_reason: str | None,
    last_exit_reason: str | None,
) -> dict[str, object]:
    def serialise_trade(trade: Trade) -> dict[str, object]:
        return {
            "entry_timestamp": trade.entry.timestamp.isoformat(),
            "exit_timestamp": trade.exit.timestamp.isoformat(),
            "entry_close": float(trade.entry.close),
            "exit_close": float(trade.exit.close),
            "return_pct": float(trade.return_pct * 100),
            "hold_duration_seconds": trade.hold_duration.total_seconds(),
            "metadata": trade.metadata,
        }

    return {
        "symbol": result.symbol,
        "total_trades": result.total_trades,
        "cumulative_return_pct": float(result.cumulative_return * 100),
        "average_trade_return_pct": float(result.average_trade_return * 100),
        "win_rate_pct": float(result.win_rate * 100),
        "best_trade_return_pct": float(result.best_trade_return * 100),
        "worst_trade_return_pct": float(result.worst_trade_return * 100),
        "max_drawdown_pct": float(result.max_drawdown * 100),
        "equity_curve": [float(level) for level in result.equity_curve],
        "equity_curve_timestamps": [ts.isoformat() for ts in timestamps] if timestamps else None,
        "trades": [serialise_trade(trade) for trade in result.trades],
        "summary": {
            "average_entry_confidence": avg_confidence,
            "last_entry_reason": last_entry_reason,
            "last_exit_reason": last_exit_reason,
        },
    }


def main() -> None:
    """Entry point compatible with setuptools-style script loading."""
    configure_logging()
    app(prog_name="gpt-trader-next")


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    main()


def _plot_equity_curve(
    levels: Sequence[Decimal],
    timestamps: Sequence[datetime] | None,
    output: Path,
    *,
    log_scale: bool,
    benchmarks: list[list[float]],
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
        from matplotlib import dates as mdates  # type: ignore[import]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise click.ClickException(
            "Plotting requires matplotlib. Install it with 'poetry add matplotlib' or omit --plot."
        ) from exc

    numeric_levels = [float(level) for level in levels]
    fig, ax = plt.subplots(figsize=(6, 4))
    if timestamps:
        date_numbers = mdates.date2num(timestamps)
        ax.plot_date(date_numbers, numeric_levels, marker="o", linestyle="-", label="equity")
        for idx, benchmark in enumerate(benchmarks):
            label = f"benchmark-{idx + 1}" if len(benchmarks) > 1 else "benchmark"
            ax.plot_date(date_numbers[: len(benchmark)], benchmark, linestyle="--", label=label)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        fig.autofmt_xdate()
    else:
        x_axis = range(len(numeric_levels))
        ax.plot(x_axis, numeric_levels, marker="o", label="equity")
        for idx, benchmark in enumerate(benchmarks):
            label = f"benchmark-{idx + 1}" if len(benchmarks) > 1 else "benchmark"
            ax.plot(x_axis[: len(benchmark)], benchmark, linestyle="--", label=label)
    if benchmarks:
        ax.legend()
    if log_scale:
        ax.set_yscale("log")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Timestamp" if timestamps else "Trade Index")
    ax.set_ylabel("Relative Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _write_equity_csv(levels: Sequence[Decimal], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["index", "equity"])
        for idx, level in enumerate(levels):
            writer.writerow([idx, float(level)])


def _write_trades_csv(trades: Sequence[Trade], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "entry_timestamp",
                "exit_timestamp",
                "entry_close",
                "exit_close",
                "return_pct",
                "hold_duration_seconds",
                "entry_reason",
                "exit_reason",
                "entry_confidence",
            ]
        )
        for trade in trades:
            meta = trade.metadata or {}
            entry_meta = meta.get("entry", {}) if isinstance(meta.get("entry"), dict) else {}
            exit_meta = meta.get("exit", {}) if isinstance(meta.get("exit"), dict) else {}
            entry_reason = (
                entry_meta.get("reason") if isinstance(entry_meta.get("reason"), str) else None
            )
            exit_reason = (
                exit_meta.get("reason") if isinstance(exit_meta.get("reason"), str) else None
            )
            entry_confidence = (
                entry_meta.get("confidence")
                if isinstance(entry_meta.get("confidence"), (float, int))
                else None
            )

            writer.writerow(
                [
                    trade.entry.timestamp.isoformat(),
                    trade.exit.timestamp.isoformat(),
                    float(trade.entry.close),
                    float(trade.exit.close),
                    float(trade.return_pct * 100),
                    trade.hold_duration.total_seconds(),
                    entry_reason,
                    exit_reason,
                    entry_confidence,
                ]
            )


def _load_benchmark(path: Path) -> dict[str, list]:
    timestamps: list[datetime | None] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) == 1:
                timestamps.append(None)
                values.append(float(row[0]))
            else:
                ts = _parse_timestamp(row[0])
                timestamps.append(ts)
                values.append(float(row[1]))
    return {"timestamps": timestamps if any(timestamps) else None, "values": values}


def _parse_timestamp(raw: str) -> datetime | None:
    if raw.endswith("Z"):
        raw = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _align_benchmarks(
    benchmarks: list[dict[str, list]],
    target_length: int,
    eq_timestamps: Sequence[datetime] | None,
    *,
    mode: str,
) -> list[list[float]]:
    aligned: list[list[float]] = []
    for bench in benchmarks:
        values = bench.get("values", [])
        times = bench.get("timestamps")
        if times and eq_timestamps and any(times):
            bench_pairs = []
            for idx, ts in enumerate(times):
                if ts is None:
                    continue
                bench_pairs.append((_to_epoch(ts), values[idx]))
            if bench_pairs:
                eq_numeric = [_to_epoch(ts) for ts in eq_timestamps]
                bench_numeric, bench_values = zip(*bench_pairs)
                bench_numeric = list(bench_numeric)
                bench_values = list(bench_values)
                series: list[float] = []
                idx = 0
                last = bench_values[0]
                for eq_ts in eq_numeric:
                    while idx < len(bench_numeric) and bench_numeric[idx] <= eq_ts:
                        last = bench_values[idx]
                        idx += 1
                    series.append(last)
                aligned.append(series)
                continue
        aligned.append(_align_values(values, target_length, mode))
    return aligned


def _align_values(values: list[float], target_length: int, mode: str) -> list[float]:
    if mode == "trim":
        trimmed = values[:target_length]
        if len(trimmed) < target_length:
            if values:
                trimmed.extend([values[-1]] * (target_length - len(trimmed)))
            else:
                trimmed = [0.0] * target_length
        return trimmed
    if mode == "pad":
        if not values:
            return [0.0] * target_length
        padded = values[:]
        while len(padded) < target_length:
            padded.append(padded[-1])
        return padded[:target_length]
    return values


def _equity_timestamps(trades: Sequence[Trade]) -> list[datetime]:
    if not trades:
        return []
    timestamps = [trades[0].entry.timestamp]
    timestamps.extend(trade.exit.timestamp for trade in trades)
    return timestamps


def _to_epoch(ts: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp()
