# Backtesting Simulation Harness

## Overview

The backtesting simulation harness provides a production-grade framework for validating trading strategies against historical data. The key principle is **simulation fidelity**: the backtest uses the exact same `StrategyCoordinator.run_cycle()` logic as live trading, with only the broker/data adapters swapped.

## Architecture Principles

### 1. Zero Logic Drift
- Same strategy coordinator, execution coordinator, and risk manager
- Same circuit breakers, position sizing, and PnL tracking
- Only difference: swappable broker implementation (`IBrokerage` interface)

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

## Key Interfaces

### IMarketData
Provides historical and real-time market data abstraction.

```python
class IMarketData(Protocol):
    """Market data interface for both live and simulated environments."""

    def candles(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Fetch candles for the given symbol and time range."""
        ...

    def best_bid_ask(self, symbol: str) -> tuple[Decimal, Decimal]:
        """Get current best bid/ask quotes."""
        ...

    def mark_price(self, symbol: str) -> Decimal:
        """Get current mark price (mid of best bid/ask)."""
        ...
```

**Implementations:**
- `LiveMarketData`: Wraps `CoinbaseBrokerage.get_quote()` and `get_candles()`
- `SimulatedMarketData`: Replays historical candles from cache

### IExecution
Abstracts order placement and position tracking.

```python
class IExecution(Protocol):
    """Execution interface for order management."""

    async def place(self, order_spec: OrderSpec) -> Order:
        """Place a new order."""
        ...

    async def cancel(self, order_id: str) -> bool:
        """Cancel an existing order."""
        ...

    async def fills(self, symbol: str | None = None) -> list[Fill]:
        """Fetch fill history."""
        ...

    async def positions(self) -> list[Position]:
        """Get current positions."""
        ...
```

**Implementations:**
- `LiveExecution`: Wraps `CoinbaseBrokerage` REST API
- `SimulatedExecution`: In-memory order matching with fill model

### IPortfolio
Tracks balances and enables transfers between spot/futures portfolios.

```python
class IPortfolio(Protocol):
    """Portfolio management interface."""

    async def balances(self) -> list[Balance]:
        """Get current asset balances."""
        ...

    async def equity(self) -> Decimal:
        """Calculate total portfolio equity in USD."""
        ...

    async def transfer(
        self,
        amount: Decimal,
        currency: str,
        from_portfolio: PortfolioType,
        to_portfolio: PortfolioType,
    ) -> bool:
        """Transfer funds between portfolios."""
        ...
```

**Implementations:**
- `LivePortfolio`: Wraps `CoinbaseAccountManager`
- `SimulatedPortfolio`: In-memory balance tracking

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
from bot_v2.backtesting import (
    SimulatedBroker,
    HistoricalDataManager,
    ClockedBarRunner,
    GoldenPathValidator,
)
from bot_v2.orchestration.coordinators import StrategyCoordinator

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
    strategy_coordinator = StrategyCoordinator()
    strategy_coordinator.initialize(context)

    async for bar_time, bars in runner.run():
        # Update sim broker with current bars
        sim_broker.update_market_data(bar_time, bars)

        # Run strategy cycle (identical to live!)
        await strategy_coordinator.run_cycle()

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

## References

- [Coinbase Advanced Trade API](https://docs.cloud.coinbase.com/advanced-trade-api/docs/rest-api-overview)
- [Candles Endpoint](https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getcandles)
- [Fee Structure](https://help.coinbase.com/en/advanced-trade/trading-and-funding/advanced-trade-fees)
- [Funding Rates](https://help.coinbase.com/en/coinbase/trading-and-funding/perpetual-futures/funding-rates)
