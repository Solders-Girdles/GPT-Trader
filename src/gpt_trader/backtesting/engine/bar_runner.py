"""Bar-by-bar runner for backtesting simulations."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

from gpt_trader.backtesting.engine.clock import SimulationClock
from gpt_trader.backtesting.types import ClockSpeed
from gpt_trader.core import Candle, Quote

if TYPE_CHECKING:
    from gpt_trader.backtesting.chaos.engine import ChaosEngine
    from gpt_trader.backtesting.simulation.broker import SimulatedBroker


class FundingRateProvider(Protocol):
    """Protocol for providing funding rates."""

    def get_rate(self, symbol: str, at_time: datetime) -> Decimal | None:
        """Get the funding rate for a symbol at a given time."""
        ...


@dataclass
class ConstantFundingRates:
    """Simple provider that returns constant rates per symbol."""

    rates_8h: dict[str, Decimal] = field(default_factory=dict)

    def get_rate(self, symbol: str, at_time: datetime) -> Decimal | None:
        """Return constant rate regardless of time."""
        return self.rates_8h.get(symbol)


@dataclass
class FundingProcessor:
    """
    Manages funding rate processing during backtesting.

    Tracks the last time funding was processed for each symbol and
    determines when funding should be applied based on the accrual interval.

    Usage:
        processor = FundingProcessor(
            rate_provider=ConstantFundingRates({"BTC-PERP-USDC": Decimal("0.0001")}),
            accrual_interval_hours=1,
        )

        async for bar_time, bars, quotes in runner.run():
            for symbol, bar in bars.items():
                broker.update_bar(symbol, bar)
            processor.process_funding(broker, bar_time, list(bars.keys()))
            # ... strategy logic ...
    """

    rate_provider: FundingRateProvider
    accrual_interval_hours: int = 1
    enabled: bool = True

    # Internal state
    _last_funding_time: dict[str, datetime] = field(default_factory=dict)
    _total_funding_processed: Decimal = field(default=Decimal("0"))

    def should_process(self, symbol: str, current_time: datetime) -> bool:
        """Check if funding should be processed for a symbol."""
        if not self.enabled:
            return False

        if symbol not in self._last_funding_time:
            return True

        elapsed = current_time - self._last_funding_time[symbol]
        hours = elapsed.total_seconds() / 3600
        return hours >= self.accrual_interval_hours

    def process_funding(
        self,
        broker: "SimulatedBroker",
        current_time: datetime,
        symbols: list[str],
    ) -> Decimal:
        """
        Process funding for all eligible symbols.

        Args:
            broker: SimulatedBroker instance
            current_time: Current simulation time
            symbols: List of symbols to potentially process

        Returns:
            Total funding processed this call (positive = paid, negative = received)
        """
        if not self.enabled:
            return Decimal("0")

        total_funding = Decimal("0")

        for symbol in symbols:
            if not self.should_process(symbol, current_time):
                continue

            rate = self.rate_provider.get_rate(symbol, current_time)
            if rate is None:
                continue

            # Process funding through broker
            funding = broker.process_funding(symbol, rate)
            total_funding += funding

            # Update tracking
            self._last_funding_time[symbol] = current_time

        self._total_funding_processed += total_funding
        return total_funding

    def get_total_funding(self) -> Decimal:
        """Get total funding processed across all calls."""
        return self._total_funding_processed

    def reset(self) -> None:
        """Reset internal state for a new backtest run."""
        self._last_funding_time.clear()
        self._total_funding_processed = Decimal("0")


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
        chaos_engine: "ChaosEngine | None" = None,
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
            chaos_engine: Optional ChaosEngine for data perturbations
        """
        self.data_provider = data_provider
        self.symbols = symbols
        self.granularity = granularity
        self.start_date = start_date
        self.end_date = end_date

        # Clock
        self.clock = SimulationClock(speed=clock_speed, start_time=start_date)

        # Optional chaos injection
        self._chaos_engine = chaos_engine

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
                for symbol, bar in bars.items():
                    broker.update_bar(symbol, bar)
                await strategy_coordinator.run_cycle()
        """
        current_time = self.start_date

        while current_time < self.end_date:
            # Fetch bars for all symbols at current time
            bars = await self._fetch_bars_for_time(current_time)

            # Apply chaos to candles (drops/adjustments)
            bars = self._apply_chaos_to_bars(current_time, bars)

            # Skip if we don't have data for any symbols
            if not bars:
                current_time += self._granularity_delta
                continue

            # Apply optional latency to the delivered timestamp
            effective_time = current_time
            if self._chaos_engine and self._chaos_engine.is_enabled():
                effective_time = self._chaos_engine.apply_latency(current_time)

            # Generate quotes from bars
            quotes = self._bars_to_quotes(bars, effective_time)

            # Trigger on_bar_start hooks
            for hook in self._on_bar_start_hooks:
                hook(effective_time, bars)

            # Advance clock
            await self.clock.advance_async(self._granularity_delta)
            self._current_time = current_time
            self._bars_processed += 1

            # Yield data to caller
            yield effective_time, bars, quotes

            # Trigger on_bar_end hooks
            for hook in self._on_bar_end_hooks:
                hook(effective_time, bars)

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
                ts=bar.ts or bar_time,
            )

        return quotes

    def on_bar_start(self, callback: Callable[[datetime, dict[str, Candle]], None]) -> None:
        """Register callback to be called at the start of each bar."""
        self._on_bar_start_hooks.append(callback)

    def on_bar_end(self, callback: Callable[[datetime, dict[str, Candle]], None]) -> None:
        """Register callback to be called at the end of each bar."""
        self._on_bar_end_hooks.append(callback)

    def set_chaos_engine(self, chaos_engine: "ChaosEngine | None") -> None:
        """Attach or clear the ChaosEngine for this runner."""
        self._chaos_engine = chaos_engine

    def _apply_chaos_to_bars(
        self,
        bar_time: datetime,
        bars: dict[str, Candle],
    ) -> dict[str, Candle]:
        if not self._chaos_engine or not self._chaos_engine.is_enabled():
            return bars

        adjusted: dict[str, Candle] = {}
        for symbol, candle in bars.items():
            processed = self._chaos_engine.process_candle(symbol, candle, bar_time)
            if processed is None:
                continue
            adjusted[symbol] = processed

        return adjusted

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


class IHistoricalDataProvider(ABC):
    """
    Abstract interface for historical data providers.

    This is implemented by HistoricalDataManager which fetches from
    cache or Coinbase API.
    """

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """
        Fetch candles for symbol in time range.

        Args:
            symbol: Trading pair (e.g., "BTC-USD", "ETH-PERP-USDC")
            granularity: Candle granularity (e.g., "ONE_MINUTE", "FIVE_MINUTE")
            start: Start time (inclusive)
            end: End time (exclusive)

        Returns:
            List of candles sorted by timestamp
        """
        ...
