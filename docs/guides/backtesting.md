# Backtesting Simulation Harness

## Overview

The backtesting simulation harness provides a production-grade framework for validating trading strategies against historical data. The key principle is **simulation fidelity**: the backtest reuses the same strategy decision logic as live trading, with only the broker/data adapters swapped (and optional guard-parity execution via `BacktestGuardedExecutor`).

## Architecture Principles

### 1. Zero Logic Drift
- Same strategy `decide()` implementation as live trading (via `create_strategy`)
- Optional guard-parity execution via `BacktestGuardedExecutor`
- Same circuit breakers, position sizing, and PnL tracking
- Only difference: swappable broker implementation (live `CoinbaseRestService` vs `SimulatedBroker`)

### 2. Production Parity
- Realistic order fills with slippage and spread modeling
- Accurate fee calculation matching Coinbase Advanced Trade tiers
- Funding PnL accrual for perpetual positions
- Risk controls fire identically when using `BacktestGuardedExecutor`

### 3. Fast Iteration
- Candle cache stored as JSON with coverage index (`HistoricalDataManager`)
- Bar-by-bar replay with configurable clock speed
- Validation framework for comparing sim vs. live decisions

## Component Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    Backtesting Framework                       │
└───────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   ┌────────────┐      ┌────────────┐     ┌────────────┐
   │ Historical │      │ Simulation │     │ Validation │
   │   Data     │      │   Broker   │     │  Framework │
   │  Manager   │      │            │     │            │
   └────────────┘      └────────────┘     └────────────┘
          │                   │                   │
          │                   │                   │
          ▼                   ▼                   ▼
   ┌────────────┐      ┌────────────┐     ┌────────────┐
   │   JSON     │      │   Order    │     │  Golden    │
   │   Cache    │      │    Fill    │     │   Path     │
   │            │      │   Model    │     │  Replays   │
   └────────────┘      └────────────┘     └────────────┘
          │                   │                   │
          │                   │                   │
          ▼                   ▼                   ▼
   ┌────────────┐      ┌────────────┐     ┌────────────┐
   │ Coverage   │      │    Fee     │     │   Chaos    │
   │   Index    │      │Calculator  │     │   Tests    │
   │            │      │            │     │            │
   └────────────┘      └────────────┘     └────────────┘
                              │
                              ▼
                       ┌────────────┐
                       │  Funding   │
                       │    PnL     │
                       │  Tracker   │
                       └────────────┘
```

## Key Components

### Broker Implementations
The backtesting framework uses concrete broker implementations that share a common API surface:

**Live Trading:**
- `CoinbaseRestService`: Production REST service for Coinbase Advanced Trade API

**Backtesting:**
- `SimulatedBroker`: In-memory simulation with order fill model

Both implementations provide the same core methods used by the live trading engine:
- `list_balances()` - Get account balances
- `list_positions()` - Get current positions
- `place_order(...)` - Place a new order
- `get_quote(symbol)` - Get current bid/ask quote
- `get_candles(...)` - Fetch historical candles

This API surface keeps strategy logic consistent between live and backtest runs; when you
need guard parity, route orders through `BacktestGuardedExecutor` against `SimulatedBroker`.

## Simulation Engine

### ClockedBarRunner
Time-based replay engine that feeds historical bars to your backtest loop.

**Features:**
- Configurable granularity (`ONE_MINUTE`, `FIVE_MINUTE`, `ONE_HOUR`, `ONE_DAY`, etc.)
- Clock control (real-time, 10x, 100x, instant)
- Symbol alignment (synchronize bars across multiple products)
- Event hooks (on_bar_start, on_bar_end)

**Usage:**
```python
runner = ClockedBarRunner(
    data_provider=data_provider,
    symbols=["BTC-PERP-USDC", "ETH-PERP-USDC"],
    granularity="FIVE_MINUTE",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    clock_speed=ClockSpeed.INSTANT,
)

async for bar_time, bars, quotes in runner.run():
    # bars: dict[str, Candle] - one per symbol
    for symbol, bar in bars.items():
        broker.update_bar(symbol, bar)
    # Drive your strategy or guarded executor for this timestep.
```

### Guarded Execution (Optional)
Use `BacktestGuardedExecutor` to run order submissions through the live validation
stack (OrderValidator + OrderSubmitter) against `SimulatedBroker`. This provides
the closest parity with live guard behavior when you need risk checks and
rejection tracking in backtests.

### Order Fill Model

**Market Orders:**
- Filled at next bar's open price
- Slippage: 0.02% for liquid pairs (BTC, ETH), 0.05% for others
- Spread impact: half of current spread added/subtracted based on side

**Limit Orders:**
- **Touched**: Limit price within bar's high/low range
- **Traded**: Assume filled if volume > 2x order size (conservative)
- **Queue priority** (optional): model partial fills based on volume profile

**Stop Orders:**
- Triggered when stop price breached
- Filled as market order on next bar

**Implementation:**
```python
class OrderFillModel:
    def __init__(
        self,
        slippage_bps: dict[str, Decimal],  # Per-symbol slippage
        spread_impact_pct: Decimal = Decimal("0.5"),  # 50% of spread
        volume_threshold: Decimal = Decimal("2.0"),  # 2x for limit fills
    ):
        ...

    def fill_market_order(
        self,
        order: Order,
        current_bar: Candle,
        best_bid: Decimal,
        best_ask: Decimal,
    ) -> Fill:
        """Simulate market order fill."""
        ...

    def try_fill_limit_order(
        self,
        order: Order,
        current_bar: Candle,
    ) -> Fill | None:
        """Attempt to fill limit order if touched."""
        ...
```

### Fee Calculation

Coinbase Advanced Trade uses a tiered maker/taker model based on 30-day volume.

**Fee Tiers (as of 2024):**
| 30-Day Volume (USD) | Maker | Taker |
|---------------------|-------|-------|
| < 10K               | 0.60% | 0.80% |
| 10K - 50K           | 0.40% | 0.60% |
| 50K - 100K          | 0.25% | 0.40% |
| 100K - 1M           | 0.15% | 0.25% |
| 1M - 15M            | 0.10% | 0.20% |
| 15M - 75M           | 0.05% | 0.15% |
| 75M - 250M          | 0.03% | 0.10% |
| > 250M              | 0.00% | 0.05% |

**Implementation:**
```python
class FeeCalculator:
    def __init__(self, tier: FeeTier = FeeTier.TIER_1):
        self.maker_bps = TIER_RATES[tier].maker
        self.taker_bps = TIER_RATES[tier].taker

    def calculate(
        self,
        notional: Decimal,
        is_maker: bool,
    ) -> Decimal:
        """Calculate fee for a fill."""
        rate = self.maker_bps if is_maker else self.taker_bps
        return notional * rate / Decimal("10000")
```

### Funding PnL for Perpetuals

Funding rates are exchanged every 8 hours for perps (00:00, 08:00, 16:00 UTC).

**Accrual Model:**
- Track hourly funding rate from product metadata
- Accrue every hour: `funding_pnl += position_size * mark_price * funding_rate`
- Settle twice daily at 00:00 and 12:00 UTC (Coinbase convention)
- Include in equity curve calculation

**Implementation:**
```python
class FundingPnLTracker:
    def __init__(self):
        self.accrued_funding: dict[str, Decimal] = {}  # Per symbol
        self.last_funding_ts: dict[str, datetime] = {}

    def accrue(
        self,
        symbol: str,
        position_size: Decimal,
        mark_price: Decimal,
        funding_rate: Decimal,
        current_time: datetime,
    ) -> Decimal:
        """Accrue funding for the current hour."""
        hours_elapsed = (current_time - self.last_funding_ts[symbol]).total_seconds() / 3600
        funding = position_size * mark_price * funding_rate * Decimal(hours_elapsed)
        self.accrued_funding[symbol] += funding
        self.last_funding_ts[symbol] = current_time
        return funding

    def settle(self, symbol: str) -> Decimal:
        """Settle accrued funding and reset."""
        settled = self.accrued_funding.get(symbol, Decimal("0"))
        self.accrued_funding[symbol] = Decimal("0")
        return settled
```

## Historical Data Management

Historical candles are fetched via `CoinbaseHistoricalFetcher` and cached by
`HistoricalDataManager`. The cache is JSON-based and tracks coverage ranges so
backtests only fetch missing windows. Default cache location is
`~/.gpt_trader/cache/candles` (override with `cache_dir`).

**Cache layout:**
- `<cache_dir>/<symbol>_<granularity>.json`
- `<cache_dir>/_coverage_index.json`

**Example:**
```python
from datetime import datetime
from pathlib import Path

from gpt_trader.backtesting.data import create_coinbase_data_provider
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient

client = CoinbaseClient(api_mode="advanced")  # public endpoints (no auth required)
data_provider = create_coinbase_data_provider(
    client=client,
    cache_dir=Path("runtime_data/dev/cache/candles"),
    validate_quality=True,
    spike_threshold_pct=15.0,
    volume_anomaly_std=6.0,
)

candles = await data_provider.get_candles(
    symbol="BTC-USD",
    granularity="FIVE_MINUTE",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 2, 1),
)
quality_report = data_provider.get_quality_report("BTC-USD", "FIVE_MINUTE")
```

## CLI quickstart

Use the backtest runner to execute single runs or walk-forward sweeps from the CLI.

```bash
uv run python scripts/backtest_runner.py --profile canary --symbol BTC-USD --granularity TWO_HOUR \
  --strategy-type mean_reversion --mean-reversion-trend-filter --enable-shorts \
  --start 2025-10-01 --end 2026-01-17
```

Walk-forward flags:

- `--walk-forward`: enable walk-forward mode (default is a single backtest).
- `--wf-windows`: number of walk-forward windows.
- `--wf-window-days`: window length in days.
- `--wf-step-days`: step size in days.
- `--wf-require-all-pass`: require every window to pass the strategy gates.
- `--end`: anchor_end for the walk-forward schedule; defaults to UTC midnight today.

Outputs (single run):

- `runtime_data/<profile>/reports/backtest_<run_id>.json`
- `runtime_data/<profile>/reports/backtest_<run_id>.txt`

Outputs (walk-forward):

- `runtime_data/<profile>/reports/walk_forward_<timestamp>/summary.md`
- `runtime_data/<profile>/reports/walk_forward_<timestamp>/summary.json`
- Per-window subdirectories for each walk-forward slice

## Validation Framework

### Golden-Path Comparison

Compare logged decisions from live and simulated runs using `DecisionLogger` and
`GoldenPathValidator`.

**Workflow:**
1. Record decisions during live/backtest runs with `DecisionLogger`.
2. Pair live vs simulated `StrategyDecision` entries.
3. Validate pairs and generate a report.

**Example:**
```python
from decimal import Decimal
from pathlib import Path

from gpt_trader.backtesting.validation import (
    DecisionLogger,
    GoldenPathValidator,
    StrategyDecision,
)

logger = DecisionLogger(storage_path=Path("runtime_data/dev/decision_logs"))
cycle_id = logger.start_cycle()

live = StrategyDecision.create(
    cycle_id=cycle_id,
    symbol="BTC-USD",
    equity=Decimal("10000"),
    position_quantity=Decimal("0"),
    position_side=None,
    mark_price=Decimal("42000"),
    recent_marks=[Decimal("41900"), Decimal("42000")],
).with_action("BUY", quantity=Decimal("0.01"))

sim = StrategyDecision.create(
    cycle_id=cycle_id,
    symbol="BTC-USD",
    equity=Decimal("10000"),
    position_quantity=Decimal("0"),
    position_side=None,
    mark_price=Decimal("42000"),
    recent_marks=[Decimal("41900"), Decimal("42000")],
).with_action("BUY", quantity=Decimal("0.01"))

logger.log_decision(live)
logger.log_decision(sim)

validator = GoldenPathValidator()
validator.validate_decision(live, sim)
report = validator.generate_report(cycle_id=cycle_id)
print(report.match_rate)
```

### Chaos Testing

Inject edge cases to validate system resilience.

**Scenarios:**
1. **Missing candles**: Random gaps in historical data
2. **Stale marks**: Delay mark price updates by N seconds
3. **Spiky spreads**: Widen bid/ask spread to 5x normal
4. **Exchange errors**: Simulate order rejection (insufficient funds, rate limit)
5. **Partial fills**: Limit orders only partially filled
6. **Network latency**: Add artificial delays to API calls

**Expected Outcomes:**
- Circuit breakers fire when volatility exceeds threshold
- Risk guards prevent orders when marks are stale
- Reconciliation loop detects and handles divergences
- Graceful degradation (reduce-only mode, no crashes)

**Implementation:**
```python
from decimal import Decimal

from gpt_trader.backtesting.chaos import ChaosEngine
from gpt_trader.backtesting.types import ChaosScenario

chaos_engine = ChaosEngine(broker)
chaos_engine.add_scenario(
    ChaosScenario(
        name="volatile_market",
        missing_candles_probability=Decimal("0.05"),
        stale_marks_delay_seconds=30,
        spread_multiplier=Decimal("3.0"),
        partial_fill_probability=Decimal("0.2"),
    )
)
chaos_engine.enable()

runner = ClockedBarRunner(
    data_provider=data_provider,
    symbols=symbols,
    granularity=granularity,
    start_date=start_date,
    end_date=end_date,
    clock_speed=ClockSpeed.INSTANT,
    chaos_engine=chaos_engine,
)
```

## Integration Example

For a runnable script, see `examples/backtesting_example.py`.

```python
from datetime import datetime
from decimal import Decimal

from gpt_trader.backtesting import ClockedBarRunner, SimulatedBroker
from gpt_trader.backtesting.data import create_coinbase_data_provider
from gpt_trader.backtesting.types import ClockSpeed, FeeTier, SimulationConfig
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient


async def run_backtest() -> None:
    client = CoinbaseClient(api_mode="advanced")  # public endpoints (no auth required)
    data_provider = create_coinbase_data_provider(client)

    symbols = ["BTC-PERP-USDC", "ETH-PERP-USDC"]
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    granularity = "FIVE_MINUTE"

    config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        granularity=granularity,
        initial_equity_usd=Decimal("100000"),
        fee_tier=FeeTier.TIER_2,
    )
    broker = SimulatedBroker(
        initial_equity_usd=config.initial_equity_usd,
        fee_tier=config.fee_tier,
        config=config,
    )

    runner = ClockedBarRunner(
        data_provider=data_provider,
        symbols=symbols,
        granularity=granularity,
        start_date=start_date,
        end_date=end_date,
        clock_speed=ClockSpeed.INSTANT,
    )

    async for bar_time, bars, _quotes in runner.run():
        for symbol, bar in bars.items():
            broker.update_bar(symbol, bar)
        # TODO: run your strategy/executor for this timestep.

    report = broker.generate_report()
    print(report)
```

## Reports and Metrics

Backtest reporting utilities live in `gpt_trader.backtesting.metrics`:

- `BacktestReporter`: aggregated statistics and summaries
- `calculate_trade_statistics` / `calculate_risk_metrics`
- `generate_backtest_report` for a quick `BacktestResult`

Example:
```python
from gpt_trader.backtesting.metrics import BacktestReporter

reporter = BacktestReporter(broker)
result = reporter.generate_result(start_date, end_date)
print(result.total_return)
```

## References

- [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs/rest-api-overview)
- [Candles Endpoint](https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getcandles)
- [Fee Structure](https://help.coinbase.com/en/advanced-trade/trading-and-funding/advanced-trade-fees)
- [Funding Rates](https://help.coinbase.com/en/coinbase/trading-and-funding/perpetual-futures/funding-rates)
