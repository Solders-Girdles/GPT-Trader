"""Quick benchmark for TradingBot mark updates and symbol processing dispatch.

The script simulates quote fetching latency and compares the historical
sequential update path with the new concurrent batching implemented in
``TradingBot.update_marks``. Results are printed in milliseconds along with the
expected speed-up factor so we have an easy before/after datapoint.

Usage example::

    uv run python scripts/analysis/perps_bot_hot_path_benchmark.py --symbols 6 --latency-ms 25

The defaults keep the run very short (<1s) while still demonstrating the
performance delta.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from collections.abc import Iterable


@dataclass
class Quote:
    symbol: str
    last: float
    ts: datetime


class FakeBroker:
    """Simulates blocking quote retrieval with configurable latency."""

    def __init__(self, latency_seconds: float) -> None:
        self._latency_seconds = latency_seconds

    def get_quote(self, symbol: str) -> Quote:
        time.sleep(self._latency_seconds)
        return Quote(symbol=symbol, last=100.0, ts=datetime.now(UTC))


def _consume_quote(symbol: str, quote: Quote, mark_windows: dict[str, list[Decimal]]) -> None:
    if quote is None:
        raise RuntimeError(f"No quote for {symbol}")
    last_price = getattr(quote, "last", getattr(quote, "last_price", None))
    if last_price is None:
        raise RuntimeError(f"Quote missing price for {symbol}")
    mark = Decimal(str(last_price))
    if mark <= 0:
        raise RuntimeError(f"Invalid mark price: {mark} for {symbol}")
    mark_windows[symbol].append(mark)
    if len(mark_windows[symbol]) > 20:
        mark_windows[symbol].pop(0)


async def _sequential_update(symbols: Iterable[str], broker: FakeBroker) -> float:
    symbols = tuple(symbols)
    mark_windows = {symbol: [] for symbol in symbols}
    start = time.perf_counter()
    for symbol in symbols:
        quote = await asyncio.to_thread(broker.get_quote, symbol)
        _consume_quote(symbol, quote, mark_windows)
    return (time.perf_counter() - start) * 1000


async def _batched_update(symbols: Iterable[str], broker: FakeBroker) -> float:
    symbols = tuple(symbols)
    mark_windows = {symbol: [] for symbol in symbols}
    start = time.perf_counter()
    quotes = await asyncio.gather(
        *(asyncio.to_thread(broker.get_quote, symbol) for symbol in symbols),
        return_exceptions=True,
    )
    for symbol, result in zip(symbols, quotes):
        if isinstance(result, Exception):
            raise RuntimeError(f"Quote fetch failed for {symbol}") from result
        _consume_quote(symbol, result, mark_windows)
    return (time.perf_counter() - start) * 1000


async def run_benchmark(symbol_count: int, iterations: int, latency_seconds: float) -> None:
    symbols = [f"SYM-{i}" for i in range(symbol_count)]
    broker = FakeBroker(latency_seconds=latency_seconds)

    sequential_runs: list[float] = []
    batched_runs: list[float] = []

    for _ in range(iterations):
        sequential_runs.append(await _sequential_update(symbols, broker))
        batched_runs.append(await _batched_update(symbols, broker))

    seq_mean = statistics.fmean(sequential_runs)
    batched_mean = statistics.fmean(batched_runs)
    seq_std = statistics.pstdev(sequential_runs) if len(sequential_runs) > 1 else 0.0
    batched_std = statistics.pstdev(batched_runs) if len(batched_runs) > 1 else 0.0
    speed_up = seq_mean / batched_mean if batched_mean else float("inf")

    print(f"Sequential (mean±σ): {seq_mean:.2f}±{seq_std:.2f} ms over {iterations} runs")
    print(f"Batched    (mean±σ): {batched_mean:.2f}±{batched_std:.2f} ms over {iterations} runs")
    print(f"Speed-up factor: {speed_up:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TradingBot mark update strategies")
    parser.add_argument("--symbols", type=int, default=6, help="Number of symbols to simulate")
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=20.0,
        help="Simulated per-quote latency in milliseconds",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of timing iterations per strategy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_benchmark(
            symbol_count=max(1, args.symbols),
            iterations=max(1, args.iterations),
            latency_seconds=max(args.latency_ms, 0.0) / 1000.0,
        )
    )


if __name__ == "__main__":
    main()
