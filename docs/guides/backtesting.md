# Backtesting Simulation Harness

## Overview

The backtesting simulation harness provides a production-grade framework for validating trading strategies against historical data. The key principle is **simulation fidelity**: the backtest uses the exact same `StrategyCoordinator.run_cycle()` logic as live trading, with only the broker/data adapters swapped.

## Architecture Principles

### 1. Zero Logic Drift
- Same strategy coordinator, execution coordinator, and risk manager
- Same circuit breakers, position sizing, and PnL tracking
- Only difference: swappable broker implementation (live `CoinbaseRestService` vs `SimulatedBroker`)

### 2. Production Parity
- Realistic order fills with slippage and spread modeling
- Accurate fee calculation matching Coinbase Advanced Trade tiers
- Funding PnL accrual for perpetual positions
- Risk controls fire identically (daily loss limits, circuit breakers, reduce-only)

### 3. Fast Iteration
- Historical data cached in Parquet format with DuckDB queries
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
   │  Parquet   │      │   Order    │     │  Golden    │
   │   Cache    │      │    Fill    │     │   Path     │
   │            │      │   Model    │     │  Replays   │
   └────────────┘      └────────────┘     └────────────┘
          │                   │                   │
          │                   │                   │
          ▼                   ▼                   ▼
   ┌────────────┐      ┌────────────┐     ┌────────────┐
   │  DuckDB    │      │    Fee     │     │   Chaos    │
   │  Queries   │      │Calculator  │     │   Tests    │
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

Both implementations provide the same core methods used by the orchestration layer:
- `list_balances()` - Get account balances
- `list_positions()` - Get current positions
- `place_order(...)` - Place a new order
- `get_quote(symbol)` - Get current bid/ask quote
- `get_candles(...)` - Fetch historical candles

This API surface ensures **zero logic drift** between live and backtest environments - the strategy coordinator and risk manager interact identically with either implementation.

## Simulation Engine

### ClockedBarRunner
Time-based replay engine that feeds historical bars to the strategy coordinator.

**Features:**
- Configurable granularity (1m, 5m, 1h, 1d)
- Clock control (real-time, 10x, 100x, instant)
- Symbol alignment (synchronize bars across multiple products)
- Event hooks (on_bar_start, on_bar_end)

**Usage:**
```python
runner = ClockedBarRunner(
    data_manager=historical_data,
    granularity="5m",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    clock_speed=ClockSpeed.INSTANT,
)

async for bar_time, bars in runner.run():
    # bars: dict[str, Candle] - one per symbol
    await strategy_coordinator.run_cycle()
```

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

### Data Fetching
Pull candles from Coinbase REST API and cache locally.

**Endpoint:**
```
GET /api/v3/brokerage/products/{product_id}/candles
```

**Parameters:**
- `start`: Unix timestamp (seconds)
- `end`: Unix timestamp (seconds)
- `granularity`: ONE_MINUTE, FIVE_MINUTE, ONE_HOUR, ONE_DAY

**Rate Limits:**
- 10 requests/second (public endpoints)
- Pagination: max 300 candles per request

**Implementation Strategy:**
1. Chunk date ranges into 300-candle batches
2. Parallel fetching with rate limit throttling (10 RPS)
3. Cache raw responses in Parquet partitioned by symbol/granularity
4. Build manifest of available data coverage

### Parquet Cache Schema

```python
# Candles table
┌──────────────┬──────────┬─────────┐
│ Column       │ Type     │ Notes   │
├──────────────┼──────────┼─────────┤
│ symbol       │ string   │ Partition│
│ granularity  │ string   │ Partition│
│ timestamp    │ int64    │ Unix (s) │
│ open         │ decimal  │ Price    │
│ high         │ decimal  │ Price    │
│ low          │ decimal  │ Price    │
│ close        │ decimal  │ Price    │
│ volume       │ decimal  │ Base     │
│ fetched_at   │ int64    │ Metadata │
└──────────────┴──────────┴─────────┘

# Product metadata table
┌──────────────────┬──────────┬──────────┐
│ Column           │ Type     │ Notes    │
├──────────────────┼──────────┼──────────┤
│ symbol           │ string   │ PK       │
│ min_order_size   │ decimal  │          │
│ price_increment  │ decimal  │          │
│ quote_increment  │ decimal  │          │
│ leverage_max     │ decimal  │ Perps    │
│ funding_rate     │ decimal  │ Perps    │
│ market_type      │ string   │ SPOT/PERP│
│ is_active        │ bool     │          │
└──────────────────┴──────────┴──────────┘
```

### DuckDB Queries

```python
class HistoricalDataManager:
    def __init__(self, cache_dir: Path):
        self.conn = duckdb.connect(str(cache_dir / "candles.duckdb"))
        self._register_parquet_tables()

    def fetch_candles(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Fetch candles from cache or API."""
        # Check cache coverage
        cached = self._query_cache(symbol, granularity, start, end)

        if self._has_gaps(cached, start, end):
            # Fetch missing data from API
            missing = self._fetch_from_api(symbol, granularity, start, end)
            self._write_to_cache(missing)
            cached.extend(missing)

        return sorted(cached, key=lambda c: c.timestamp)

    def _query_cache(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Query Parquet cache via DuckDB."""
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE symbol = ?
          AND granularity = ?
          AND timestamp >= ?
          AND timestamp < ?
        ORDER BY timestamp
        """
        return self.conn.execute(query, [
            symbol,
            granularity,
            int(start.timestamp()),
            int(end.timestamp()),
        ]).fetchall()
```

## Validation Framework

### Golden-Path Replays

Compare simulation decisions against recent live trading cycles.

**Workflow:**
1. Export last N live cycles from event store (decisions, orders, fills)
2. Replay same time period in simulation
3. Compare decision outputs (action, side, quantity, price)
4. Flag divergences with detailed diff

**Implementation:**
```python
class GoldenPathValidator:
    def __init__(
        self,
        event_store: EventStore,
        sim_broker: SimulatedBroker,
    ):
        ...

    async def validate_cycle(
        self,
        cycle_id: str,
    ) -> ValidationReport:
        """Replay a live cycle and compare decisions."""
        # 1. Load live cycle from event store
        live_events = await self.event_store.get_cycle_events(cycle_id)
        live_decisions = self._extract_decisions(live_events)

        # 2. Replay in simulation
        sim_decisions = await self._replay_cycle(cycle_id, self.sim_broker)

        # 3. Compare
        diffs = self._compare_decisions(live_decisions, sim_decisions)

        return ValidationReport(
            cycle_id=cycle_id,
            total_decisions=len(live_decisions),
            divergences=len(diffs),
            diffs=diffs,
        )
```

**Example Diff Output:**
```
Cycle: 2024-03-15T14:30:00
Symbol: BTC-PERP-USDC

Live Decision:
  Action: BUY
  Quantity: 0.05
  Price: 67500.00 (limit)

Sim Decision:
  Action: BUY
  Quantity: 0.05
  Price: 67500.00 (limit)

✓ MATCH

---

Symbol: ETH-PERP-USDC

Live Decision:
  Action: SELL
  Quantity: 0.5
  Price: MARKET

Sim Decision:
  Action: HOLD
  Quantity: 0.0
  Price: N/A

✗ DIVERGENCE
  Reason: Sim mark price (3850.25) vs Live mark price (3851.00)
  Impact: Missed threshold by 0.02%
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
class ChaosEngine:
    def __init__(self, broker: SimulatedBroker):
        self.broker = broker
        self.chaos_modes: dict[str, bool] = {
            "missing_candles": False,
            "stale_marks": False,
            "wide_spreads": False,
            "order_errors": False,
        }

    def enable(self, mode: str):
        """Enable a chaos mode."""
        self.chaos_modes[mode] = True

    def inject_missing_candles(
        self,
        symbol: str,
        gap_probability: float = 0.05,
    ):
        """Randomly remove candles."""
        ...

    def inject_stale_marks(
        self,
        delay_seconds: int = 30,
    ):
        """Delay mark price updates."""
        ...
```

## Integration Example

```python
from gpt_trader.backtesting import (
    SimulatedBroker,
    HistoricalDataManager,
    ClockedBarRunner,
    GoldenPathValidator,
)
from gpt_trader.features.live_trade.engines.runtime.coordinator import RuntimeCoordinator

async def run_backtest():
    # 1. Initialize historical data
    data_manager = HistoricalDataManager(
        cache_dir=Path("data/cache"),
        api_client=coinbase_client,
    )

    # 2. Create simulated broker
    sim_broker = SimulatedBroker(
        market_data=data_manager,
        initial_equity=Decimal("100000"),
        fee_tier=FeeTier.TIER_2,
    )

    # 3. Setup bar runner
    runner = ClockedBarRunner(
        data_manager=data_manager,
        symbols=["BTC-PERP-USDC", "ETH-PERP-USDC"],
        granularity="5m",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
    )

    # 4. Create coordinator context (same as live!)
    context = CoordinatorContext(
        config=bot_config,
        broker=sim_broker,  # Only difference!
        risk_manager=risk_manager,
        event_store=event_store,
        orders_store=orders_store,
        runtime_state=runtime_state,
        symbols=("BTC-PERP-USDC", "ETH-PERP-USDC"),
    )

    # 5. Run backtest
    coordinator = RuntimeCoordinator()
    coordinator.initialize(context)

    async for bar_time, bars in runner.run():
        # Update sim broker with current bars
        sim_broker.update_market_data(bar_time, bars)

        # Run strategy cycle (identical to live!)
        await coordinator.run_cycle()

    # 6. Generate report
    report = sim_broker.generate_report()
    print(report)

# Run validation
validator = GoldenPathValidator(event_store, sim_broker)
await validator.validate_last_n_cycles(n=10)
```

## Performance Metrics

### Backtest Speed
- **Instant replay**: ~1000 bars/second (3 months in ~10 seconds)
- **Real-time**: 1:1 with clock time
- **10x speed**: 10 minutes = 100 minutes of data

### Data Storage
- 1 year of 1-minute candles (BTC-PERP): ~500MB (Parquet compressed)
- 10 symbols × 3 granularities × 1 year: ~15GB

### Memory Usage
- DuckDB in-memory cache: ~2GB for 1-year dataset
- Simulation state: ~100MB per symbol

## Next Steps

1. **Phase 1**: Implement core interfaces and SimulatedBroker (Week 1)
2. **Phase 2**: Build historical data fetcher and Parquet cache (Week 1-2)
3. **Phase 3**: Add order fill model and fee calculator (Week 2)
4. **Phase 4**: Implement funding PnL tracking (Week 2-3)
5. **Phase 5**: Build validation framework (Week 3)
6. **Phase 6**: Add chaos testing capabilities (Week 4)
7. **Phase 7**: Full integration test and documentation (Week 4)

## Production-Parity Backtesting

> **WARNING: PARTIALLY IMPLEMENTED**
>
> This section documents features that are currently being built in `gpt_trader.features.optimize`.
> While the `optimize` package exists, the specific `backtest_engine.py` interface described below may differ from the actual implementation in `batch_runner.py`.
> For existing backtesting functionality, see `gpt_trader.backtesting` module.

### Key Features

- **Zero Code Duplication**: Reuses `BaselinePerpsStrategy.decide()` directly
- **Decision Logging**: Records every decision with full context
- **Parity Validation**: Compare backtest vs live decision logs
- **Human-Readable Logs**: JSON format for easy inspection

### Quick Start

```python
from decimal import Decimal
import pandas as pd
# NOTE: These imports reference unimplemented modules
from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import (
    BaselinePerpsStrategy,
    StrategyConfig,  # StrategyConfig is in strategy.py, not config.py
)
from gpt_trader.features.optimize.backtest_engine import run_backtest_production  # NOT IMPLEMENTED
from gpt_trader.features.optimize.types_v2 import BacktestConfig  # NOT IMPLEMENTED

# Load historical data
data = pd.read_csv("historical_btc.csv")  # Must have 'close' column

# Configure strategy (same config you use in production)
strategy_config = StrategyConfig(
    short_ma_period=5,
    long_ma_period=20,
    position_fraction=0.1,
    enable_shorts=False,
)

# Create strategy instance
strategy = BaselinePerpsStrategy(config=strategy_config)

# Configure backtest
backtest_config = BacktestConfig(
    initial_capital=Decimal("10000"),
    commission_rate=Decimal("0.001"),  # 0.1% = 10 bps
    slippage_rate=Decimal("0.0005"),   # 0.05% = 5 bps
    enable_decision_logging=True,
)

# Run backtest
result = run_backtest_production(
    strategy=strategy,
    data=data,
    symbol="BTC-USD",
    config=backtest_config,
)

print(result.summary())
```

### Decision Logging

Every decision is logged with complete context in JSON format:

```json
{
  "run_id": "bt_20250122_143052_BTC-USD",
  "decisions": [
    {
      "context": {
        "timestamp": "2025-01-15T10:30:00",
        "current_mark": "42350.50",
        "position_state": null,
        "equity": "10500.00"
      },
      "decision": {
        "action": "buy",
        "quantity": "0.1",
        "reason": "Bullish MA crossover"
      },
      "execution": {
        "filled": true,
        "fill_price": "42352.00"
      }
    }
  ]
}
```

Logs stored in: `backtesting/decision_logs/YYYY-MM-DD/bt_{timestamp}_{symbol}.json`

### Parity Validation

Compare backtest decisions against live trading:

```python
from gpt_trader.features.optimize.decision_logger import compare_decision_logs

comparison = compare_decision_logs(
    backtest_log=backtest_path,
    live_log=live_path,
)

print(f"Parity Rate: {comparison['parity_rate']:.2%}")
assert comparison['parity_rate'] > 0.99, "Parity validation failed!"
```

**Go/No-Go Criteria**: Parity rate > 99% on 24-hour shadow run

### API Reference

```python
def run_backtest_production(
    *,
    strategy: BaselinePerpsStrategy,
    data: pd.DataFrame,
    symbol: str,
    product: Product | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run production-parity backtest."""

@dataclass
class BacktestConfig:
    initial_capital: Decimal = Decimal("10000")
    commission_rate: Decimal = Decimal("0.001")   # 10 bps
    slippage_rate: Decimal = Decimal("0.0005")    # 5 bps
    enable_decision_logging: bool = True
    log_directory: str = "backtesting/decision_logs"
```

### Troubleshooting

**No trades in backtest**: Ensure `len(data) > long_ma_period` and data contains volatility.

**Parity mismatch**: Check MA periods match, position state format, equity calculations.

## References

- [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs/rest-api-overview)
- [Candles Endpoint](https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getcandles)
- [Fee Structure](https://help.coinbase.com/en/advanced-trade/trading-and-funding/advanced-trade-fees)
- [Funding Rates](https://help.coinbase.com/en/coinbase/trading-and-funding/perpetual-futures/funding-rates)
