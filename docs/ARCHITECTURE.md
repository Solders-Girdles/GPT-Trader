# System Architecture

---
status: current
last-updated: 2025-10-05
---

## Current State

GPT-Trader V2 is a production-ready Coinbase **spot** trading system that retains future-ready perpetuals logic. Perps execution remains disabled in live environments until Coinbase grants INTX access, but the architecture keeps those paths compiled and testable.

**Recent Refactoring (Oct 2025 - Phase 0-3):**
- ✅ **Phase 0:** MarketDataService & StreamingService extraction
  - Extracted `features/market_data/` package from PerpsBot monolith
  - Separated WebSocket streaming logic into dedicated service
  - Feature flags `USE_NEW_MARKET_DATA_SERVICE` and `USE_NEW_STREAMING_SERVICE` retired (services always active)
- ✅ **Phase 1:** CLI modularization
  - Modular command handlers in `cli/commands/`
  - Feature flag `USE_NEW_CLI_HANDLERS` default=true
- ✅ **Phase 2:** Live trade service extraction
  - Advanced execution, liquidity service, order policy composition
  - Spot profile service, risk gate validator
- ✅ **Phase 3:** PerpsBotBuilder pattern
  - Builder-based orchestration with `orchestration/builders/perps_bot_builder.py`
  - Feature flag `USE_PERPS_BOT_BUILDER` default=true
- Result: 5,007 passing tests (up from 2,145), 87.52% coverage
- Rollback capability: Feature flags enable same-day rollback without redeploy

**Previous Refactoring (2025-09-29 to 2025-10-06):**
- Archived experimental features (backtest, ml_strategy, market_regime, monitoring_dashboard) → `archived/experimental_features_2025_09_29/`
- Retired over-engineered systems (adaptive_portfolio, state management platform) → `archived/features_adaptive_portfolio/`, `archived/state_platform/`
- Extracted models from large files into subpackages:
  - `features/live_trade/advanced_execution_models/` - Execution models (72 lines)
- Removed 16 orphaned config files from deleted/archived features
- Result: Cleaner separation of concerns, improved maintainability, reduced configuration drift

## Trading Capabilities Matrix

| Environment | Products | Authentication | API Version | WebSocket | Use Case |
|------------|----------|----------------|-------------|-----------|----------|
| **Production (default)** | Spot (BTC-USD, ETH-USD, …) | HMAC | Advanced v3 | Real-time | Live trading |
| **Production (perps)** | Perpetuals (INTX-gated) | CDP (JWT) + `COINBASE_ENABLE_DERIVATIVES=1` | Advanced v3 | Real-time | Future activation |
| **Sandbox** | Not used (API diverges) | — | — | — | Paper/mock via `PERPS_PAPER=1` |

## Component Architecture

### Vertical Slice Design

The system is organized into vertical feature slices under `src/bot_v2/features/`. Production-critical slices (e.g., `live_trade`, `brokerages`, `position_sizing`) ship with full test coverage and form the core trading engine.

```
src/bot_v2/features/
├── live_trade/          # Production trading engine
├── paper_trade/         # Simulated trading harness
├── analyze/             # Market analytics helpers
├── position_sizing/     # Kelly & intelligent sizing utilities
├── strategies/          # Baseline and experimental strategies
├── strategy_tools/      # Shared helpers for strategy slices
├── brokerages/          # Exchange integrations
├── data/                # Data acquisition helpers
└── optimize/            # Parameter optimisation experiments
```

**Note**: `adaptive_portfolio/` archived to `archived/features_adaptive_portfolio/` (over-engineered for current needs)

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
| `bot_v2/features/market_data/` | **Phase 0:** Market data service with WebSocket/REST fallback, mark price management |
| `bot_v2/features/live_trade` | Risk management, execution engines, strategies, and trading utilities |
| `bot_v2/features/live_trade/risk/` | Risk management subpackage: position sizing, pre-trade validation, runtime monitoring, state management |
| `bot_v2/orchestration/` | Core orchestration layer: config management, execution/runtime/strategy coordination, telemetry, reconciliation |
| `bot_v2/orchestration/builders/` | **Phase 3:** PerpsBotBuilder pattern for service composition and initialization |
| `bot_v2/orchestration/execution/` | Execution subpackage: guards, validation, order submission, state collection |
| `bot_v2/features/brokerages/coinbase` | REST/WS integration for Coinbase Advanced Trade spot markets |
| `bot_v2/features/brokerages/coinbase/client/` | Modular client package with mixins (accounts, orders, portfolio, market data) |
| `bot_v2/features/brokerages/coinbase/rest/` | REST service layer: orders, portfolio, products, P&L calculation |
| `bot_v2/features/position_sizing` | Kelly-style sizing with guardrails |
| `bot_v2/cli/commands/` | **Phase 1:** Modular CLI command handlers (accounts, orders, convert, etc.) |
| `bot_v2/monitoring` | Runtime guard orchestration, alert dispatch, system metrics |
| `bot_v2/validation` | Predicate-based validators and input decorators |

#### Coinbase Client Package

The previous monolithic `client.py` was replaced with a composable package (`client/__init__.py`
plus mixins). Each mixin owns a REST surface (accounts, orders, market data, portfolio), while the
base class centralises retry, throttling, and auth wiring. Scripts and slices now import through
`bot_v2.features.brokerages.coinbase.client import CoinbaseClient` to ensure consistent
initialisation.

A parallel REST service layer (`rest/`) provides higher-level operations:
- `rest/base.py` - Base REST service with shared utilities
- `rest/orders.py` - Order management operations
- `rest/portfolio.py` - Portfolio operations and conversions
- `rest/products.py` - Product catalog and specifications
- `rest/pnl.py` - P&L calculation and reconciliation

This two-layer design separates low-level HTTP concerns (client/) from business logic (rest/).

#### Risk Management Framework

The `features/live_trade/risk/` subpackage provides comprehensive risk controls (refactored from 1,044-line monolith):

- `manager.py` (351 lines) - `LiveRiskManager` facade delegating to specialized components
- `position_sizing.py` (192 lines) - `PositionSizer` with Kelly Criterion and confidence modifiers
- `pre_trade_checks.py` (522 lines) - `PreTradeValidator` for synchronous order validation
- `runtime_monitoring.py` (314 lines) - `RuntimeMonitor` for async/periodic guard checks
- `state_management.py` (119 lines) - `StateManager` for reduce-only mode and state tracking

This modular design achieves 66% file size reduction with clear separation of concerns: sizing, validation, monitoring, and state management are independently testable.

#### Monitoring & Validation Framework

- **Validators** (`bot_v2/validation`): the base `Validator` now accepts inline predicates and
  optional value coercion, enabling concise one-off validations while keeping legacy subclasses.
- **Runtime guards** (`bot_v2/monitoring/runtime_guards/`): modular guard system with base classes,
  built-in guards, and manager for evaluation. Supports rich comparison modes (`gt`, `lt`, `abs_gt`),
  warning bands, and contextual messaging to power orchestration checks and monitoring dashboards.
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

**Note:** Experimental features (backtest, ml_strategy, market_regime) were archived on 2025-09-29 to streamline the codebase. Restore from `archived/experimental_features_2025_09_29/` if needed.

Legacy adaptive portfolio and state-management stacks were retired in Q4 2025 and are preserved for reference under `archived/features_adaptive_portfolio/` and `archived/state_platform/`.

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

The orchestration layer provides coordinated control across trading operations through specialized modules:

**Configuration & Symbol Management:**
- `configuration.py` - Profile-aware defaults (`BotConfig`, `ConfigManager`) with validation
- `config_controller.py` - Dynamic configuration management and hot-reloading
- `symbols.py` - Symbol normalization, derivatives gating, and profile-specific defaults

**Execution Coordination:**
- `live_execution.py` - Main execution engine facade (395 lines, down from 1,063)
- `execution/` subpackage:
  - `guards.py` - Runtime guard management (368 lines)
  - `validation.py` - Pre-trade validation (272 lines)
  - `order_submission.py` - Order submission and recording (231 lines)
  - `state_collection.py` - Account state collection (185 lines)

**Strategy & Runtime Coordination:**
- `strategy_orchestrator.py` - Strategy lifecycle management and symbol processing
- `runtime_coordinator.py` - Runtime orchestration and derivatives validation
- `execution_coordinator.py` - Execution flow coordination across strategies

**Services & Telemetry:**
- `spot_profile_service.py` - Spot trading profile loading and rule management
- `account_telemetry.py` - Account metrics tracking and periodic snapshots
- `order_reconciler.py` - Order state reconciliation on startup
- `system_monitor.py` - System health monitoring and metrics publication

**Infrastructure:**
- `service_registry.py` - Explicit dependency container for runtime components
- `storage.py` - Persistent storage abstraction for checkpoints and state
- `broker_factory.py` - Broker instantiation with environment-based configuration
- `session_guard.py` - Trading window enforcement
- `market_monitor.py` - Market data freshness monitoring
- `perps_bot.py` - Main orchestrator coordinating all components
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
- Comprehensive test suite with 100% pass rate on active code (`poetry run pytest --collect-only` for current count)

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
- **Test Discovery**: Run `poetry run pytest --collect-only` to see current test count and selection status.

## Verification Path

1. **Regression Suite**: `poetry run pytest -q`
2. **Smoke Test**: `poetry run perps-bot --profile dev --dev-fast`
3. **Validation**: `python scripts/validation/verify_core.py --check all`

## Configuration System

### Profile-Based Configuration

The bot uses **hardcoded profile-based configuration** via `ConfigManager` (`orchestration/configuration.py`), NOT YAML file loading for most settings.

**Active Profiles**:
- `dev` - Mock broker, tiny positions, extensive logging (hardcoded in ConfigManager)
- `demo` - $100 max position, 1x leverage (hardcoded)
- `spot` - $50k max position, spot-only (hardcoded)
- `canary` - Ultra-safe prod testing, $10 daily loss limit (loads `config/profiles/canary.yaml`)
- `prod` - Full sizing, perps-capable (hardcoded)

**YAML Configs** (only 3 loaded):
- `config/profiles/canary.yaml` - Canary profile overrides
- `config/profiles/spot.yaml` - Used by alerts manager and profile service
- `config/profiles/dev_entry.yaml` - Dev/demo profile reference

### Risk Configuration

Risk limits use **environment variables** as primary method, with optional JSON file override:

1. **RISK_CONFIG_PATH** (optional override):
   ```bash
   export RISK_CONFIG_PATH=config/risk/dev_dynamic.json
   ```
   - Must be valid JSON (YAML not supported)
   - Currently available: `dev_dynamic.json`

2. **Environment Variables** (default):
   ```bash
   export RISK_MAX_LEVERAGE=3
   export RISK_DAILY_LOSS_LIMIT=0.02
   export RISK_MAX_EXPOSURE_PCT=0.80
   ```
   See `config/risk/README.md` for complete `RISK_*` env var list.

3. **Fallback**: Hardcoded defaults in `RiskConfig` class

### Configuration Migration (Oct 2025)

**Removed** (Phase 1 cleanup):
- 14 orphaned config files from deleted features (adaptive_portfolio, backtest, ml_strategy, etc.)
- 2 broken risk configs (YAML format incompatible with JSON-only loader)

**Current State**:
- Profile configs: Hardcoded + 3 YAMLs
- Risk configs: Env vars + optional `dev_dynamic.json`
- No dynamic config loading, fully static

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
