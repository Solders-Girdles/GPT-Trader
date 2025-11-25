"""Bar-by-bar runner for backtesting simulations."""

from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta
from decimal import Decimal

from gpt_trader.backtesting.engine.clock import SimulationClock
from gpt_trader.backtesting.types import ClockSpeed
from gpt_trader.features.brokerages.core.interfaces import Candle, Quote


class ClockedBarRunner:
    """
    Time-based replay engine that feeds historical bars to strategy coordinator.

    The runner advances through historical data bar-by-bar, updating the
    simulation clock and providing market data to the strategy.

    Features:
    - Configurable granularity (1m, 5m, 1h, 1d)
    - Clock control (real-time, fast-forward, instant)
    - Symbol alignment (synchronize bars across multiple products)
    - Event hooks (on_bar_start, on_bar_end)
    """

    def __init__(
        self,
        data_provider: "IHistoricalDataProvider",
        symbols: list[str],
        granularity: str,
        start_date: datetime,
        end_date: datetime,
        clock_speed: ClockSpeed = ClockSpeed.INSTANT,
    ):
        """
        Initialize bar runner.

        Args:
            data_provider: Provider for historical candle data
            symbols: List of symbols to run
            granularity: Bar granularity (e.g., "ONE_MINUTE", "FIVE_MINUTE")
            start_date: Start of backtest period
            end_date: End of backtest period
            clock_speed: Replay speed
        """
        self.data_provider = data_provider
        self.symbols = symbols
        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date

        # Clock
        self.clock = SimulationClock(speed=clock_speed, start_time=start_date)

        # Granularity to timedelta mapping
        self._granularity_delta = self._parse_granularity(granularity)

        # Event hooks
        self._on_bar_start_hooks: list[Callable[[datetime, dict[str, Candle]], None]] = []
        self._on_bar_end_hooks: list[Callable[[datetime, dict[str, Candle]], None]] = []

        # State
        self._current_time = start_date
        self._bars_processed = 0

    def _parse_granularity(self, granularity: str) -> timedelta:
        """Parse granularity string to timedelta."""
        mapping = {
            "ONE_MINUTE": timedelta(minutes=1),
            "FIVE_MINUTE": timedelta(minutes=5),
            "FIFTEEN_MINUTE": timedelta(minutes=15),
            "THIRTY_MINUTE": timedelta(minutes=30),
            "ONE_HOUR": timedelta(hours=1),
            "TWO_HOUR": timedelta(hours=2),
            "SIX_HOUR": timedelta(hours=6),
            "ONE_DAY": timedelta(days=1),
        }

        if granularity not in mapping:
            raise ValueError(
                f"Unsupported granularity: {granularity}. Supported: {list(mapping.keys())}"
            )

        return mapping[granularity]

    async def run(self) -> AsyncIterator[tuple[datetime, dict[str, Candle], dict[str, Quote]]]:
        """
        Run backtest simulation bar-by-bar.

        Yields:
            Tuple of (bar_time, bars, quotes) for each time step

        Example:
            async for bar_time, bars, quotes in runner.run():
                broker.update_market_data(bar_time, bars, quotes)
                await strategy_coordinator.run_cycle()
        """
        current_time = self.start_date

        while current_time < self.end_date:
            # Fetch bars for all symbols at current time
            bars = await self._fetch_bars_for_time(current_time)

            # Skip if we don't have data for any symbols
            if not bars:
                current_time += self._granularity_delta
                continue

            # Generate quotes from bars
            quotes = self._bars_to_quotes(bars, current_time)

            # Trigger on_bar_start hooks
            for hook in self._on_bar_start_hooks:
                hook(current_time, bars)

            # Advance clock
            await self.clock.advance_async(self._granularity_delta)
            self._current_time = current_time
            self._bars_processed += 1

            # Yield data to caller
            yield current_time, bars, quotes

            # Trigger on_bar_end hooks
            for hook in self._on_bar_end_hooks:
                hook(current_time, bars)

            # Move to next bar
            current_time += self._granularity_delta

    async def _fetch_bars_for_time(self, bar_time: datetime) -> dict[str, Candle]:
        """
        Fetch bars for all symbols at a specific time.

        Args:
            bar_time: Target bar timestamp

        Returns:
            Dictionary of {symbol: Candle}
        """
        bars = {}

        for symbol in self.symbols:
            # Fetch single bar at this timestamp
            candles = await self.data_provider.get_candles(
                symbol=symbol,
                granularity=self.granularity,
                start=bar_time,
                end=bar_time + self._granularity_delta,
            )

            # Take the first candle if available
            if candles:
                bars[symbol] = candles[0]

        return bars

    def _bars_to_quotes(
        self,
        bars: dict[str, Candle],
        bar_time: datetime,
    ) -> dict[str, Quote]:
        """
        Generate quotes from candles.

        For simulation, we estimate bid/ask from the close price using a
        small spread (0.05% on each side).

        Args:
            bars: Candles for each symbol
            bar_time: Current bar timestamp

        Returns:
            Dictionary of {symbol: Quote}
        """
        quotes = {}

        for symbol, bar in bars.items():
            # Estimate bid/ask with 0.05% spread on each side
            mid = bar.close
            spread_bps = Decimal("5")  # 5 basis points = 0.05%
            spread_half = mid * spread_bps / Decimal("20000")  # Divide by 2 for each side

            bid = mid - spread_half
            ask = mid + spread_half

            quotes[symbol] = Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=bar.close,
                ts=bar_time,
            )

        return quotes

    def on_bar_start(self, callback: Callable[[datetime, dict[str, Candle]], None]) -> None:
        """Register callback to be called at the start of each bar."""
        self._on_bar_start_hooks.append(callback)

    def on_bar_end(self, callback: Callable[[datetime, dict[str, Candle]], None]) -> None:
        """Register callback to be called at the end of each bar."""
        self._on_bar_end_hooks.append(callback)

    @property
    def progress_pct(self) -> float:
        """Calculate backtest progress percentage."""
        total_duration = (self.end_date - self.start_date).total_seconds()
        elapsed_duration = (self._current_time - self.start_date).total_seconds()

        if total_duration == 0:
            return 100.0

        return (elapsed_duration / total_duration) * 100.0

    @property
    def bars_remaining(self) -> int:
        """Estimate number of bars remaining."""
        remaining_duration = self.end_date - self._current_time
        bar_duration = self._granularity_delta.total_seconds()

        if bar_duration == 0:
            return 0

        return int(remaining_duration.total_seconds() / bar_duration)


# Historical data provider interface (to be implemented)
class IHistoricalDataProvider:
    """
    Interface for historical data provider.

    This is implemented by HistoricalDataManager which fetches from
    cache or API.
    """

    async def get_candles(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Fetch candles for symbol in time range."""
        raise NotImplementedError
