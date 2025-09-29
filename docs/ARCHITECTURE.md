# System Architecture

---
status: current
last-updated: 2025-09-29
---

## Current State

GPT-Trader V2 is a production-ready Coinbase **spot** trading system that retains future-ready perpetuals logic. Perps execution remains disabled in live environments until Coinbase grants INTX access, but the architecture keeps those paths compiled and testable.

## Trading Capabilities Matrix

| Environment | Products | Authentication | API Version | WebSocket | Use Case |
|------------|----------|----------------|-------------|-----------|----------|
| **Production (default)** | Spot (BTC-USD, ETH-USD, …) | HMAC | Advanced v3 | Real-time | Live trading |
| **Production (perps)** | Perpetuals (INTX-gated) | CDP (JWT) + `COINBASE_ENABLE_DERIVATIVES=1` | Advanced v3 | Real-time | Future activation |
| **Sandbox** | Not used (API diverges) | — | — | — | Paper/mock via `PERPS_PAPER=1` |

## Component Architecture

### Vertical Slice Design

The system is organized into vertical feature slices under `src/bot_v2/features/`. Production-critical slices (e.g., `live_trade`, `brokerages`, `position_sizing`) ship with full test coverage, while research/demo slices (`backtest`, `ml_strategy`, `market_regime`, `monitoring_dashboard`) are tagged `__experimental__` and excluded from the core trading loop. The former workflow engine was removed; recover it from repository history if you need a reference.

```
src/bot_v2/features/
├── live_trade/          # Production trading engine
├── paper_trade/         # Simulated trading harness
├── backtest/            # Historical simulation utilities
├── adaptive_portfolio/  # Tier-based portfolio management
├── analyze/             # Market analytics helpers
├── market_regime/       # Market condition detection
├── position_sizing/     # Kelly & intelligent sizing utilities
├── strategies/          # Baseline and experimental strategies
├── strategy_tools/      # Shared helpers for strategy slices
├── brokerages/          # Exchange integrations
├── data/                # Data acquisition helpers
└── optimize/            # Parameter optimisation experiments
```

Additional cross-cutting packages now live at the top level:

```
src/bot_v2/
├── monitoring/          # Runtime guards, alerting, system telemetry
└── validation/          # Declarative validators and decorators
```

### High-Level Flow

```
CLI (perps-bot) → Config (BotConfig) → Service Registry → LiveExecutionEngine →
Risk Guards → Coinbase Brokerage Adapter → Metrics + Telemetry
```

### Entry Point & Service Wiring

- `poetry run perps-bot` invokes `bot_v2.cli:main`, producing a `BotConfig` from
  CLI arguments and environment overrides.
- `bot_v2/orchestration/bootstrap.py` hydrates the `ServiceRegistry`, wiring the
  broker adapter, risk manager, execution engine, and telemetry surfaces before
  handing the bundle to `PerpsBot`.

### Core Subsystems

| Module | Purpose |
|--------|---------|
| `bot_v2/features/live_trade` | Main control loop, position tracking, and order routing |
| `bot_v2/features/brokerages/coinbase` | REST/WS integration for Coinbase Advanced Trade spot markets |
| `bot_v2/features/brokerages/coinbase/client/` | Modular client package with mixins (accounts, orders, portfolio, market data) |
| `bot_v2/features/position_sizing` | Kelly-style sizing with guardrails |
| `bot_v2/monitoring` | Runtime guard orchestration, alert dispatch, system metrics |
| `bot_v2/validation` | Predicate-based validators and input decorators |
| `bot_v2/features/quantization.py` | Price and quantity rounding helpers |

#### Coinbase Client Package

The previous monolithic `client.py` was replaced with a composable package (`client/__init__.py`
plus mixins). Each mixin owns a REST surface (accounts, orders, market data, portfolio), while the
base class centralises retry, throttling, and auth wiring. Scripts and slices now import through
`bot_v2.features.brokerages.coinbase.client import CoinbaseClient` to ensure consistent
initialisation.

#### Monitoring & Validation Framework

- **Validators** (`bot_v2/validation`): the base `Validator` now accepts inline predicates and
  optional value coercion, enabling concise one-off validations while keeping legacy subclasses.
- **Runtime guards** (`bot_v2/monitoring/runtime_guards.py`): guard evaluation supports rich
  comparison modes (`gt`, `lt`, `abs_gt`, etc.), warning bands, and contextual messaging to power
  both orchestration checks and monitoring dashboards.
- **Alerts** (`bot_v2/monitoring/alerts.py`): the base alert channel now degrades gracefully,
  logging when no transport is configured instead of raising. Concrete channels (Slack, PagerDuty,
  email, webhook) continue to extend `_send_impl`.
- **Risk metrics aggregation** (`bot_v2/features/live_trade/risk_metrics.py`): periodic EventStore
  snapshots feed into the monitoring stack for dashboards and analytics.

### Derivatives Gate

Perpetual futures remain behind `COINBASE_ENABLE_DERIVATIVES` and Coinbase INTX
credentials. The code paths stay compiled, but runtime flags default to spot-only
behaviour until the derivatives gate opens.

### Feature Slice Reference

#### Backtest (`features/backtest/`)
- **Purpose:** Run historical strategy simulations with token-efficient helpers such as `run_backtest`.
- **Usage:**
    from features.backtest import run_backtest

    result = run_backtest(
        strategy="MomentumStrategy",
        symbol="AAPL",
        start=datetime(2024, 1, 1),
        end=datetime(2024, 12, 31),
        initial_capital=10_000,
    )
    print(result.summary())
- **Outputs:** Returns `BacktestResult` objects with trade logs, equity curve, Sharpe/max drawdown, and win-rate metrics.

#### Paper Trade (`features/paper_trade/`)
- **Purpose:** Self-contained realtime simulation with local data feed, execution, risk, and metrics.
- **Highlights:** Quick-start helpers `start_paper_trading`, `get_status`, and `stop_paper_trading` with configurable commission, slippage, and update intervals.
- **Further reading:** See the [Paper Trading Guide](guides/paper_trading.md) for workflow, configuration, and performance tracking tips.

#### Position Sizing (`features/position_sizing/`)
- **Purpose:** Intelligent position sizing that combines Kelly Criterion math, confidence modifiers, and market-regime scaling while remaining slice-local.
- **Usage:**
    from features.position_sizing import PositionSizeRequest, calculate_position_size

    recommendation = calculate_position_size(
        PositionSizeRequest(
            symbol="AAPL",
            current_price=150,
            portfolio_value=10_000,
            strategy_name="momentum",
            win_rate=0.65,
            avg_win=0.08,
            avg_loss=-0.04,
            confidence=0.75,
            market_regime="bull_quiet",
        )
    )
- **Integration hooks:** Built to ingest ML strategy signals and market-regime detectors without cross-slice imports.

### Key Design Principles

1. **Slice Isolation**: Production slices limit cross-dependencies; experimental ones stay sandboxed.
2. **Token Awareness**: Documentation highlights slice entry points so agents can load only what they need.
3. **Type Safety**: Shared interfaces defined in `features/brokerages/core/interfaces.py`.
4. **Environment Separation**: `perps_bot` normalizes symbols to spot unless INTX derivatives access is detected.

### Orchestration Infrastructure

- `orchestration/session_guard.py` and `orchestration/market_monitor.py` encapsulate trading window enforcement and market-data freshness so `perps_bot` stays focused on orchestration glue.
- `features/live_trade/indicators.py`, `features/live_trade/risk_calculations.py`, and
  `features/live_trade/risk_runtime.py` centralize indicator math plus leverage/MMR/risk guard
  helpers, giving strategies and the risk manager shared, tested primitives. Runtime integrations
  now call into `bot_v2/monitoring/runtime_guards.py` for consistent evaluation and alert routing.
- `orchestration/configuration.py` plus the new `orchestration/config_controller.py` provide
  profile-aware defaults (`BotConfig`, `ConfigManager`) with unit coverage
  (`tests/unit/bot_v2/orchestration/test_configuration.py`). Adaptive portfolio config
  serialisation helpers now live under `features/adaptive_portfolio/config_manager.py` with tests
  to guarantee round-tripping.
- `orchestration/service_registry.py` provides an explicit container for runtime dependencies so
  the main bot can accept a prepared bundle instead of instantiating stores/brokers inline. Future
  phases will wire this into the CLI bootstrapper.
- Legacy status reports that used to live under `src/bot_v2/*.md` were removed; pull them from
  repository history if you need a reference.
- Historical V1/V2 integration and system tests depending on the legacy `bot.*` package lived under
  `archived/legacy_tests/` before the cleanup. Recover them from git history if you need a
  reference. The active pytest suite now focuses exclusively on the `bot_v2` stack and passes via
  `poetry run pytest`.

## What's Actually Working

### ✅ Fully Operational
- Coinbase spot trading via Advanced Trade (REST/WebSocket); dev profile defaults to the deterministic broker stub and can be pointed at live APIs with `SPOT_FORCE_LIVE=1`
- Order placement/management through `LiveExecutionEngine`
- Account telemetry snapshots and cycle metrics persisted for monitoring
- Runtime safety rails: daily loss guard, liquidation buffer enforcement, mark staleness detection, volatility circuit breaker, correlation checks
- 446 active tests selected at collection time (`poetry run pytest --collect-only`)

### ⚠️ Partially Working / Future Activation
- Perpetual futures execution: code paths compile and tests run, but live trading remains disabled without INTX
- Advanced WebSocket user-event handling: baseline support exists; enrichment/backfill still in progress
- Durable restart state (OrdersStore/EventStore) needs production hardening

### ❌ Not Yet Implemented
- Funding rate accrual in deterministic broker stub
- Order modification/amend flows beyond cancel
- Partial fill handling in mock (market fills remain immediate)

## Trading Profiles

### Development Profile
```yaml
profile: dev
broker: mock
positions: tiny (0.001 BTC)
logging: verbose
fills: deterministic
universe: [BTC, ETH, SOL, XRP, LTC, ADA, DOGE, BCH, AVAX, LINK]
```

### Canary Profile
```yaml
profile: canary
broker: coinbase
positions: 0.01 BTC max
daily_loss_limit: $10
trading_window: 14:00-15:00 UTC
circuit_breakers: multiple
```

### Production Profile
```yaml
profile: prod
broker: coinbase
positions: full sizing
risk_limits: production
monitoring: real-time
```

## Risk Management

### Pre-Trade Validation
- Position size limits
- Margin requirement checks
- Impact cost validation (<50bps)
- Reduce-only mode enforcement

> Spot profiles load `config/risk/spot_top10.yaml` by default, enforcing
> per-symbol notional caps and leverage=1 across the top-ten USD markets.

### Runtime Guards
- Daily loss limits
- Error rate monitoring (>50% triggers shutdown)
- Stale data detection (30s timeout)
- Drawdown protection (10% max)
- Generic guard evaluation with configurable comparison operators (`gt`, `lt`, `abs_gt`, etc.),
  warning thresholds, and contextual messaging. Guard instrumentation raises structured
  `RiskGuard*Error` exceptions: recoverable failures emit `risk.guards.<name>.recoverable_failures`
  counters and log warnings, while critical failures escalate to reduce-only mode and emit
  `risk.guards.<name>.critical_failures`.

### Circuit Breakers
- Consecutive loss protection
- Volatility spike detection
- Liquidity monitoring
- API error thresholds

## Performance & Observability

- **Cycle Metrics**: persisted to `var/data/perps_bot/<profile>/metrics.json` and exposed via the
  Prometheus exporter (`scripts/monitoring/export_metrics.py`). The live risk manager now emits
  snapshot events consumed by dashboards and the monitoring stack.
- **Account Snapshots**: periodic telemetry via `CoinbaseAccountManager` with fee/limit tracking.
- **System Monitoring**: `bot_v2/monitoring/system/` provides resource telemetry collectors used by
  the runtime guard manager and dashboards.
- **System Footprint**: bot process typically <50 MB RSS with sub-100 ms WebSocket latency in spot
  mode.
- **Test Discovery** (`pytest --collect-only`): 476 discovered / 472 selected / 4 deselected /
  4 skipped (a duplicate-named test file is tracked for follow-up).

## Verification Path

1. **Regression Suite**: `poetry run pytest -q`
2. **Smoke Test**: `poetry run perps-bot --profile dev --dev-fast`
3. **Validation**: `python scripts/validation/verify_core.py --check all`

## Dependencies

### Core Requirements
- Python 3.12+
- Poetry 1.0+
- Coinbase Advanced Trade API (for perpetuals)
- CDP API keys (for production)

### Key Libraries
- `coinbase-advanced-py` - Official Coinbase SDK
- `pandas` - Data manipulation
- `scikit-learn` - ML models
- `websocket-client` - Real-time data
- `pydantic` - Data validation

## Future Roadmap

### Near Term (Q1 2025)
- [ ] Full WebSocket user event handling
- [ ] Durable state recovery
- [ ] Order modification flows
- [ ] Partial fill handling

### Medium Term (Q2 2025)
- [ ] Multi-exchange support
- [ ] Advanced order types
- [ ] Portfolio-level risk management
- [ ] Real-time ML adaptation

### Long Term (2025+)
- [ ] Distributed execution
- [ ] Cross-exchange arbitrage
- [ ] Options integration
- [ ] Institutional features

---

*For implementation details, see [Trading Logic - Perpetuals](reference/trading_logic_perps.md)*
