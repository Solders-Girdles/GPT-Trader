"""Paper trade stress test for long-duration simulation validation.

This script runs a 30-day paper trade simulation with:
- ~43,200 bars of synthetic 1-minute data
- Realistic funding rates (0.01% per 8h)
- Full funding rate processing through the FundingProcessor
- EventStore persistence validation (optional)

Usage:
    poetry run python scripts/analysis/paper_trade_stress_test.py
    poetry run python scripts/analysis/paper_trade_stress_test.py --days 7 --persist
"""

from __future__ import annotations

import argparse
import asyncio
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from gpt_trader.backtesting.engine.bar_runner import (
    ConstantFundingRates,
    ClockedBarRunner,
    FundingProcessor,
    IHistoricalDataProvider,
)
from gpt_trader.backtesting.metrics.risk import RiskMetrics, calculate_risk_metrics
from gpt_trader.backtesting.metrics.statistics import TradeStatistics, calculate_trade_statistics
from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.types import FeeTier, SimulationConfig
from gpt_trader.core import Candle, OrderSide, OrderType, Product, Quote
from gpt_trader.persistence.event_store import EventStore


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic price data generation."""

    initial_price: Decimal = Decimal("50000")  # BTC starting price
    volatility_pct: Decimal = Decimal("0.0005")  # Per-bar volatility (0.05%)
    trend_pct: Decimal = Decimal("0.00001")  # Slight upward bias
    seed: int | None = 42


class SyntheticDataProvider(IHistoricalDataProvider):
    """Generates synthetic candle data for stress testing."""

    def __init__(self, config: SyntheticDataConfig | None = None):
        self.config = config or SyntheticDataConfig()
        self._price_cache: dict[str, dict[datetime, Candle]] = {}
        if self.config.seed is not None:
            random.seed(self.config.seed)
        self._current_prices: dict[str, Decimal] = {}

    def _generate_candle(self, symbol: str, timestamp: datetime) -> Candle:
        """Generate a single candle with realistic OHLCV data."""
        # Get or initialize current price
        if symbol not in self._current_prices:
            self._current_prices[symbol] = self.config.initial_price

        price = self._current_prices[symbol]
        vol = float(self.config.volatility_pct)
        trend = float(self.config.trend_pct)

        # Generate random walk with slight trend
        change = Decimal(str(random.gauss(trend, vol)))
        new_price = price * (Decimal("1") + change)
        new_price = max(new_price, Decimal("1"))  # Prevent negative prices

        # Generate OHLC from open and close
        open_price = price
        close_price = new_price

        # High/low within the range
        range_pct = abs(float(change)) + vol
        high_offset = Decimal(str(random.uniform(0, range_pct)))
        low_offset = Decimal(str(random.uniform(0, range_pct)))

        high_price = max(open_price, close_price) * (Decimal("1") + high_offset)
        low_price = min(open_price, close_price) * (Decimal("1") - low_offset)

        # Update current price for next candle
        self._current_prices[symbol] = close_price

        # Realistic volume (scales with price)
        base_volume = Decimal("10")
        volume = base_volume * Decimal(str(random.uniform(0.5, 2.0)))

        return Candle(
            ts=timestamp,
            open=open_price.quantize(Decimal("0.01")),
            high=high_price.quantize(Decimal("0.01")),
            low=low_price.quantize(Decimal("0.01")),
            close=close_price.quantize(Decimal("0.01")),
            volume=volume.quantize(Decimal("0.001")),
        )

    async def get_candles(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Generate candles for the requested time range."""
        # Initialize cache for symbol
        if symbol not in self._price_cache:
            self._price_cache[symbol] = {}

        # Check cache first
        if start in self._price_cache[symbol]:
            return [self._price_cache[symbol][start]]

        # Generate and cache the candle
        candle = self._generate_candle(symbol, start)
        self._price_cache[symbol][start] = candle
        return [candle]


@dataclass
class StressTestConfig:
    """Configuration for the stress test run."""

    # Time range
    days: int = 30
    granularity: str = "ONE_MINUTE"

    # Capital
    initial_equity_usd: Decimal = Decimal("100000")
    position_fraction: Decimal = Decimal("0.1")  # 10% of equity per trade

    # Funding rates (8-hour rates, typical Coinbase INTX)
    funding_rate_btc: Decimal = Decimal("0.0001")  # 0.01% per 8h (~0.91% APR)

    # Strategy
    trade_interval_bars: int = 30  # Trade every 30 minutes
    leverage: int = 3  # Moderate leverage

    # Persistence
    persist_events: bool = False

    # Validation thresholds (scale with days)
    min_trades_per_day: int = 5  # At least 5 trades per day
    max_funding_drag_pct: Decimal = Decimal("5.0")  # Max 5% funding drag


@dataclass
class StressTestResult:
    """Results from a stress test run."""

    # Duration
    wall_time_seconds: float
    bars_processed: int
    simulation_days: int

    # Performance
    initial_equity: Decimal
    final_equity: Decimal
    total_return_pct: Decimal
    max_drawdown_pct: Decimal

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Decimal

    # Funding impact
    funding_pnl: Decimal
    funding_drag_pct: Decimal

    # Fees
    total_fees: Decimal

    # Risk metrics
    sharpe_ratio: Decimal | None
    sortino_ratio: Decimal | None

    # Persistence
    events_persisted: int
    persistence_validated: bool

    # Validation
    all_validations_passed: bool
    validation_errors: list[str]


class SimpleStrategy:
    """Simple momentum strategy for stress testing.

    Trades based on recent price direction:
    - Go long if price up over last N bars
    - Go short if price down over last N bars
    - Close if reversal detected
    """

    def __init__(
        self,
        lookback: int = 20,
        threshold_pct: Decimal = Decimal("0.002"),
        leverage: int = 5,
        position_fraction: Decimal = Decimal("0.2"),
    ):
        self.lookback = lookback
        self.threshold = threshold_pct
        self.leverage = leverage
        self.position_fraction = position_fraction
        self._price_history: dict[str, list[Decimal]] = {}

    def update(self, symbol: str, price: Decimal) -> None:
        """Update price history."""
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append(price)
        # Keep only lookback * 2 prices
        if len(self._price_history[symbol]) > self.lookback * 2:
            self._price_history[symbol].pop(0)

    def decide(self, symbol: str) -> str:
        """Return 'BUY', 'SELL', 'CLOSE', or 'HOLD'."""
        history = self._price_history.get(symbol, [])
        if len(history) < self.lookback:
            return "HOLD"

        recent = history[-self.lookback :]
        old_price = recent[0]
        new_price = recent[-1]

        if old_price == 0:
            return "HOLD"

        change = (new_price - old_price) / old_price

        if change > self.threshold:
            return "BUY"
        elif change < -self.threshold:
            return "SELL"
        return "HOLD"


async def run_stress_test(config: StressTestConfig) -> StressTestResult:
    """Run the full stress test simulation."""
    start_wall_time = time.time()
    validation_errors: list[str] = []

    # Time range
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(days=config.days)

    # Create simulation config
    sim_config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        granularity=config.granularity,
        initial_equity_usd=config.initial_equity_usd,
        fee_tier=FeeTier.TIER_2,
        enable_funding_pnl=True,
        funding_accrual_hours=1,
        funding_settlement_hours=8,
        funding_rates_8h={"BTC-PERP-USDC": config.funding_rate_btc},
    )

    # Create components
    data_provider = SyntheticDataProvider()
    broker = SimulatedBroker(
        initial_equity_usd=config.initial_equity_usd,
        fee_tier=FeeTier.TIER_2,
        config=sim_config,
    )
    broker.connect()

    # Register product
    broker.products["BTC-PERP-USDC"] = Product(
        symbol="BTC-PERP-USDC",
        base_asset="BTC",
        quote_asset="USDC",
        market_type="PERPETUAL",
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=20,
        expiry=None,
        funding_rate=config.funding_rate_btc,
        next_funding_time=None,
    )

    # Create funding processor
    rate_provider = ConstantFundingRates(rates_8h=sim_config.funding_rates_8h or {})
    funding_processor = FundingProcessor(
        rate_provider=rate_provider,
        accrual_interval_hours=sim_config.funding_accrual_hours,
        enabled=sim_config.enable_funding_pnl,
    )

    # Create bar runner
    symbols = ["BTC-PERP-USDC"]
    runner = ClockedBarRunner(
        data_provider=data_provider,
        symbols=symbols,
        granularity=config.granularity,
        start_date=start_date,
        end_date=end_date,
    )

    # Create strategy
    strategy = SimpleStrategy(
        lookback=20,
        threshold_pct=Decimal("0.002"),
        leverage=config.leverage,
        position_fraction=config.position_fraction,
    )

    # Optional event store
    event_store: EventStore | None = None
    temp_dir: str | None = None
    if config.persist_events:
        temp_dir = tempfile.mkdtemp(prefix="stress_test_")
        event_store = EventStore(root=Path(temp_dir))

    # Run simulation
    bars_processed = 0
    trade_counter = 0

    print(f"Starting {config.days}-day stress test (~{config.days * 24 * 60} bars expected)...")
    progress_interval = 10000

    async for current_time, bars, quotes in runner.run():
        bars_processed += 1

        if bars_processed % progress_interval == 0:
            pct = (bars_processed / (config.days * 24 * 60)) * 100
            print(f"  Progress: {bars_processed:,} bars ({pct:.1f}%)")

        # Update broker with market data
        for symbol in symbols:
            if symbol in bars:
                broker._current_bar[symbol] = bars[symbol]
                broker._simulation_time = current_time

                # Build candle history
                if symbol not in broker._candle_history:
                    broker._candle_history[symbol] = []
                broker._candle_history[symbol].append(bars[symbol])
                if len(broker._candle_history[symbol]) > 100:
                    broker._candle_history[symbol].pop(0)

            if symbol in quotes:
                broker._current_quote[symbol] = quotes[symbol]

        # Update equity curve
        broker.update_equity_curve()

        # Process funding
        funding_processor.process_funding(
            broker=broker,
            current_time=current_time,
            symbols=symbols,
        )

        # Update strategy and potentially trade
        for symbol in symbols:
            if symbol in bars:
                strategy.update(symbol, bars[symbol].close)

        # Trade at intervals
        trade_counter += 1
        if trade_counter >= config.trade_interval_bars:
            trade_counter = 0

            for symbol in symbols:
                decision = strategy.decide(symbol)
                position = broker.get_position(symbol)
                mark = broker.get_mark_price(symbol)

                if mark is None:
                    continue

                if decision == "BUY":
                    if position is None or position.quantity <= 0:
                        # Close short if exists
                        if position and position.quantity < 0:
                            broker.place_order(
                                symbol=symbol,
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                quantity=abs(position.quantity),
                                reduce_only=True,
                            )
                        # Open long
                        target_notional = (
                            broker.equity * config.position_fraction * Decimal(str(config.leverage))
                        )
                        quantity = target_notional / mark
                        if quantity >= Decimal("0.001"):
                            broker.place_order(
                                symbol=symbol,
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                quantity=quantity,
                                leverage=config.leverage,
                            )

                elif decision == "SELL":
                    if position is None or position.quantity >= 0:
                        # Close long if exists
                        if position and position.quantity > 0:
                            broker.place_order(
                                symbol=symbol,
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                quantity=abs(position.quantity),
                                reduce_only=True,
                            )
                        # Open short
                        target_notional = (
                            broker.equity * config.position_fraction * Decimal(str(config.leverage))
                        )
                        quantity = target_notional / mark
                        if quantity >= Decimal("0.001"):
                            broker.place_order(
                                symbol=symbol,
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                quantity=quantity,
                                leverage=config.leverage,
                            )

        # Persist events periodically
        if event_store and bars_processed % 1000 == 0:
            event_store.append(
                "equity_snapshot",
                {
                    "timestamp": current_time.isoformat(),
                    "equity": str(broker.get_equity()),
                    "bars_processed": bars_processed,
                },
            )

    # Calculate final metrics
    stats = broker.get_statistics()
    risk_metrics = calculate_risk_metrics(broker)
    trade_stats = calculate_trade_statistics(broker)

    # Validate persistence
    events_persisted = 0
    persistence_validated = True
    if event_store:
        events_persisted = len(event_store.events)
        # Verify we can read events back
        if events_persisted > 0:
            recent = event_store.get_recent(10)
            if len(recent) == 0:
                persistence_validated = False
                validation_errors.append("Failed to read events from store")

        # Final snapshot
        event_store.append(
            "simulation_complete",
            {
                "timestamp": datetime.now().isoformat(),
                "final_equity": str(broker.get_equity()),
                "total_bars": bars_processed,
                "total_trades": trade_stats.total_trades,
            },
        )
        events_persisted = len(event_store.events)
        event_store.close()

    # Validation checks
    final_equity = broker.get_equity()
    funding_pnl = stats["funding_pnl"]

    # Calculate funding drag as percentage of initial
    funding_drag_pct = abs(funding_pnl) / config.initial_equity_usd * Decimal("100")

    # Check minimum trades (scales with duration)
    min_trades_expected = config.min_trades_per_day * config.days
    if trade_stats.total_trades < min_trades_expected:
        validation_errors.append(
            f"Too few trades: {trade_stats.total_trades} < {min_trades_expected} ({config.min_trades_per_day}/day)"
        )

    # Check funding drag
    if funding_drag_pct > config.max_funding_drag_pct:
        validation_errors.append(
            f"Funding drag too high: {funding_drag_pct:.2f}% > {config.max_funding_drag_pct}%"
        )

    # Check equity didn't go to zero
    if final_equity <= Decimal("0"):
        validation_errors.append(f"Equity went to zero or negative: {final_equity}")

    wall_time = time.time() - start_wall_time

    return StressTestResult(
        wall_time_seconds=wall_time,
        bars_processed=bars_processed,
        simulation_days=config.days,
        initial_equity=config.initial_equity_usd,
        final_equity=final_equity,
        total_return_pct=stats["total_return_pct"],
        max_drawdown_pct=stats["max_drawdown_pct"],
        total_trades=trade_stats.total_trades,
        winning_trades=trade_stats.winning_trades,
        losing_trades=trade_stats.losing_trades,
        win_rate=trade_stats.win_rate,
        funding_pnl=funding_pnl,
        funding_drag_pct=funding_drag_pct,
        total_fees=trade_stats.total_fees_paid,
        sharpe_ratio=risk_metrics.sharpe_ratio,
        sortino_ratio=risk_metrics.sortino_ratio,
        events_persisted=events_persisted,
        persistence_validated=persistence_validated,
        all_validations_passed=len(validation_errors) == 0,
        validation_errors=validation_errors,
    )


def print_results(result: StressTestResult) -> None:
    """Print formatted stress test results."""
    print("\n" + "=" * 60)
    print("PAPER TRADE STRESS TEST RESULTS")
    print("=" * 60)

    print(f"\n📊 Simulation Summary")
    print(f"   Duration: {result.simulation_days} days ({result.bars_processed:,} bars)")
    print(f"   Wall time: {result.wall_time_seconds:.1f} seconds")
    print(f"   Speed: {result.bars_processed / result.wall_time_seconds:,.0f} bars/sec")

    print(f"\n💰 Performance")
    print(f"   Initial Equity: ${result.initial_equity:,.2f}")
    print(f"   Final Equity:   ${result.final_equity:,.2f}")
    print(f"   Total Return:   {result.total_return_pct:+.2f}%")
    print(f"   Max Drawdown:   {result.max_drawdown_pct:.2f}%")

    print(f"\n📈 Trade Statistics")
    print(f"   Total Trades: {result.total_trades}")
    print(f"   Winners: {result.winning_trades}")
    print(f"   Losers:  {result.losing_trades}")
    print(f"   Win Rate: {result.win_rate:.1f}%")

    print(f"\n💸 Costs")
    print(f"   Total Fees Paid: ${result.total_fees:,.2f}")
    print(f"   Funding PnL:     ${result.funding_pnl:+,.2f}")
    print(f"   Funding Drag:    {result.funding_drag_pct:.2f}%")

    print(f"\n📉 Risk Metrics")
    if result.sharpe_ratio is not None:
        print(f"   Sharpe Ratio:  {result.sharpe_ratio:.2f}")
    else:
        print(f"   Sharpe Ratio:  N/A (insufficient data)")
    if result.sortino_ratio is not None:
        print(f"   Sortino Ratio: {result.sortino_ratio:.2f}")
    else:
        print(f"   Sortino Ratio: N/A (insufficient data)")

    if result.events_persisted > 0:
        print(f"\n💾 Persistence")
        print(f"   Events Persisted: {result.events_persisted:,}")
        print(f"   Validation: {'✅ PASSED' if result.persistence_validated else '❌ FAILED'}")

    print(f"\n✅ Validation")
    if result.all_validations_passed:
        print("   All validations PASSED")
    else:
        print("   Validation FAILURES:")
        for error in result.validation_errors:
            print(f"   ❌ {error}")

    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run long-duration paper trade stress test")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to simulate (default: 30)",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Enable event persistence to SQLite",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=100000,
        help="Initial equity in USD (default: 100000)",
    )
    parser.add_argument(
        "--leverage",
        type=int,
        default=5,
        help="Trading leverage (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    config = StressTestConfig(
        days=args.days,
        initial_equity_usd=Decimal(str(args.equity)),
        leverage=args.leverage,
        persist_events=args.persist,
    )

    # Set random seed
    random.seed(args.seed)

    result = asyncio.run(run_stress_test(config))
    print_results(result)

    return 0 if result.all_validations_passed else 1


if __name__ == "__main__":
    exit(main())
