# System Architecture

## ⚠️ Architecture Transition (Complete)

**Important Note for Developers and Agents:**

This project completed a major architectural migration from a legacy "Orchestration" pattern (monolithic builders) to a modern **Dependency Injection** pattern using `ApplicationContainer`.

- **Legacy Core:** removed during the DI migration (no longer present)
- **Modern Core:** `src/gpt_trader/app/` (Composition Root) and `src/gpt_trader/features/` (Vertical Slices)
- **Migration Guide:** See [MIGRATION_STATUS.md](MIGRATION_STATUS.md) for current migration progress and patterns.

While this document describes the **current architecture**, you may still encounter legacy references in documentation. Always prefer the patterns defined in `src/gpt_trader/app` and `src/gpt_trader/features`.

---
status: current
last-updated: 2026-01-08
---

## Current State

GPT-Trader is a production-ready Coinbase Advanced Trade trading system supporting **spot** and **CFM futures** trading. INTX perpetuals code paths remain compiled and testable but require international account access.

> 📘 **Trust reminder:** Confirm this document's details against `docs/agents/Document_Verification_Matrix.md` before acting on them.

## Trading Capabilities Matrix

| Environment | Products | Authentication | API Version | WebSocket | Use Case |
|------------|----------|----------------|-------------|-----------|----------|
| **Production (spot)** | Spot (BTC-USD, ETH-USD, …) | JWT (CDP key) | Advanced v3 | Real-time | Live spot trading |
| **Production (CFM)** | US Futures (BTC, ETH, SOL, etc.) | CDP (JWT) | Advanced v3 | Real-time | Regulated US futures |
| **Production (INTX)** | Perpetuals (international) | CDP (JWT) + `COINBASE_ENABLE_INTX_PERPS=1` (legacy: `COINBASE_ENABLE_DERIVATIVES=1`) | Advanced v3 | Real-time | Requires INTX account |
| **Paper** | All products | — | — | — | Simulated via `PERPS_PAPER=1` |

### Derivatives Access Summary
- **CFM (Coinbase Financial Markets)**: US-regulated futures with expiration dates. Endpoints: `cfm_balance_summary`, `cfm_positions`, `cfm_sweeps`, margin settings.
- **INTX (International Exchange)**: Perpetual futures for non-US users. Endpoints implemented but gated by account access.

## Component Architecture

### Vertical Slice Design

The system is organized into vertical feature slices under `src/gpt_trader/features/`. Production-critical slices (e.g., `live_trade`, `brokerages`, `intelligence`) ship with full test coverage.

```
src/gpt_trader/features/
├── brokerages/          # Exchange integrations
├── data/                # Data acquisition helpers
├── intelligence/        # Strategy intelligence (sizing, regime, ensemble)
│   └── sizing/          # Kelly criterion position sizing
├── live_trade/          # Production trading engine
├── optimize/            # Parameter optimisation experiments
├── research/            # Research and evaluation (wrapping backtesting tools)
├── strategy_dev/        # Strategy development lab
└── strategy_tools/      # Shared helpers for strategy slices
```

Additional cross-cutting packages now live at the top level. These intentionally span feature slices:

```
src/gpt_trader/
├── backtesting/          # Canonical backtesting framework
├── errors/              # Centralized error hierarchy providing consistent exception types
├── monitoring/          # Runtime guards, configuration guardian, system logger
├── persistence/          # Event/order stores and persistence utilities
├── preflight/           # Production preflight verification and startup checks
├── security/            # Security primitives: input sanitization, secrets management
├── tui/                 # Terminal User Interface (Textual-based)
└── validation/          # Declarative validators and decorators
```

> **Note:** `src/gpt_trader/backtesting/` is the canonical backtesting framework. `features/research/` hosts
> research workflows and adapters that build on top of the backtesting package.

### High-Level Flow

```
CLI (gpt-trader) → Config (BotConfig) → ApplicationContainer → TradingBot →
Risk Guards → Coinbase Brokerage Adapter → Metrics + Telemetry
```

### Capability Map

For a detailed breakdown of system capabilities, runtime flow diagrams, and "where to change things" guidance, see **[CAPABILITIES.md](CAPABILITIES.md)**.

Key capabilities documented:
- Configuration + Feature Flags (with [FEATURE_FLAGS.md](FEATURE_FLAGS.md) reference)
- Trading Decisioning, Pre-trade Validation, Order Execution
- Runtime Guards vs Pre-trade Guards
- Risk & Degradation, Streaming, Observability

### Entry Point & Service Wiring

- `uv run gpt-trader` invokes `gpt_trader.cli:main`, producing a `BotConfig` from
  CLI arguments and environment overrides.
- `ApplicationContainer` (`gpt_trader/app/container.py`) is the **canonical composition root**.
  It lazily initializes all services (broker, risk manager, event store, etc.) and wires
  them into `TradingBot` via `container.create_bot()`.
- `TradingBot` receives services directly from the container—no intermediate registry.

> **Note:** The legacy `gpt_trader/orchestration/` package was removed during the DI migration.
> Use `gpt_trader/app/bootstrap.py` and `ApplicationContainer` for all new code.

#### Container Sub-Containers

`ApplicationContainer` delegates to specialized sub-containers:

| Sub-Container | Services |
|---------------|----------|
| `ConfigContainer` | Config controller, profile loader |
| `ObservabilityContainer` | Notifications, health state, secrets manager |
| `PersistenceContainer` | Event store, orders store, runtime paths |
| `BrokerageContainer` | Broker, market data service, product catalog |
| `RiskValidationContainer` | Risk manager, validation failure tracker |

#### Container Requirement (Testing)

Service locators such as `get_failure_tracker()` always require an application
container. Tests should set up `ApplicationContainer` via fixtures or explicit
registration before calling container-resolved helpers.

#### Removed Modules (DI migration)

The `gpt_trader.orchestration` package was removed during the DI migration. Use the canonical paths:

| Removed Path | Canonical Path |
|--------------|----------------|
| `gpt_trader.orchestration.execution.degradation` | `gpt_trader.features.live_trade.degradation` |
| `gpt_trader.orchestration.configuration.risk.model` | `gpt_trader.features.live_trade.risk.config` |
| `gpt_trader.orchestration.configuration.bot_config` | `gpt_trader.app.config` |

`LiveExecutionEngine` was removed in v4.0. Use the TradingEngine guard stack (`_validate_and_place_order` in the live loop; `submit_order` for external callers).

**Order Execution:** Use the TradingEngine guard stack (`_validate_and_place_order` in the live loop; `submit_order` for external callers).

> **Historical deprecation tracker:** See [DEPRECATIONS.md](DEPRECATIONS.md) for migration history.

### Core Subsystems

| Module | Purpose |
|--------|---------|
| `gpt_trader/features/live_trade` | Main control loop, position tracking, and order routing |
| `gpt_trader/features/live_trade/risk/` | Risk management subpackage: position sizing, pre-trade validation, runtime monitoring, state management |
| `gpt_trader/features/live_trade/execution/` | Execution subpackage: guards, validation, order submission, state collection |
| `gpt_trader/app/` | Application bootstrap, DI container, and configuration management |
| `gpt_trader/features/brokerages/coinbase` | REST/WS integration for Coinbase Advanced Trade spot markets |
| `gpt_trader/features/brokerages/coinbase/client/` | Modular client package with mixins (accounts, orders, portfolio, market data) |
| `gpt_trader/features/brokerages/coinbase/rest/` | REST service layer: orders, portfolio, products, P&L calculation |
| `gpt_trader/features/intelligence/sizing` | Kelly-style sizing with regime awareness |
| `gpt_trader/monitoring` | Runtime guard orchestration, alert dispatch, system metrics |
| `gpt_trader/tui` | Terminal User Interface for monitoring and control |
| `gpt_trader/validation` | Predicate-based validators and input decorators |

#### Engine Context Pattern (Current)

Live trading uses a simplified engine pattern centered on `CoordinatorContext`.

**Key components**

- `CoordinatorContext` (`src/gpt_trader/features/live_trade/engines/base.py`) - snapshot of broker, risk, stores, and notifications
- `TradingEngine` (`src/gpt_trader/features/live_trade/engines/strategy.py`) - main trading loop and guard stack
- Telemetry helpers (`src/gpt_trader/features/live_trade/engines/telemetry_health.py`, `telemetry_streaming.py`) - WS health and streaming
- Runtime stub (`src/gpt_trader/features/live_trade/engines/runtime/coordinator.py`) - placeholder for future lifecycle splits
- Strategy orchestration helpers (`src/gpt_trader/features/live_trade/orchestrator/`)

**Lifecycle flow**

1. `TradingBot` builds a `CoordinatorContext` and instantiates `TradingEngine`.
2. `TradingEngine.start_background_tasks()` launches background tasks (health checks, streaming, status, maintenance).
3. `TradingEngine.shutdown()` stops background tasks and health checks.

Legacy orchestration facades were removed during the DI migration. Use `features/live_trade/` and `app/` paths.

#### Coinbase Client Package

The previous monolithic `client.py` was replaced with a composable package (`client/__init__.py`
plus mixins). Each mixin owns a REST surface (accounts, orders, market data, portfolio), while the
base class centralises retry, throttling, and auth wiring. Scripts and slices now import through
`gpt_trader.features.brokerages.coinbase.client import CoinbaseClient` to ensure consistent
initialisation.

A parallel REST service layer (`rest/`) provides higher-level operations:
- `rest/base.py` - Base REST service with shared utilities
- `rest/orders.py` - Order management operations
- `rest/portfolio.py` - Portfolio operations and conversions
- `rest/products.py` - Product catalog and specifications
- `rest/pnl.py` - P&L calculation and reconciliation

This two-layer design separates low-level HTTP concerns (client/) from business logic (rest/).

#### Risk Management Framework

The `features/live_trade/risk/` package defines the live risk manager and config:

- `risk/config.py` - `RiskConfig` dataclass for risk limits and thresholds (env-driven).
- `risk/manager/__init__.py` - `LiveRiskManager` handling leverage, daily loss, exposure, reduce-only.
- `risk/protocols.py` - protocol interfaces for DI and testing.

Runtime guards are orchestrated by `features/live_trade/execution/guard_manager.py` with
guard implementations under `features/live_trade/execution/guards/`. Pre-trade
validation lives in `features/live_trade/execution/validation.py` and
`security/security_validator.py`.

#### Monitoring & Validation Framework

- **Validators** (`gpt_trader/validation`): the base `Validator` now accepts inline predicates and
  optional value coercion, enabling concise one-off validations while keeping legacy subclasses.
- **Runtime guards** (`gpt_trader/features/live_trade/execution/guard_manager.py`): coordinates
  per-cycle guard checks using implementations under `features/live_trade/execution/guards/`.
- **Guard alert dispatcher** (`gpt_trader/features/live_trade/guard_errors.py`): wraps the lightweight
  alert manager to emit guard failures without depending on the retired alert stack. Rebuild
  multi-channel routing as needed for email, Slack, or PagerDuty integrations.
- **Risk metrics aggregation** (`gpt_trader/features/live_trade/execution/guards/risk_metrics.py`):
  runtime guard snapshots feed into the monitoring stack for dashboards and analytics.

#### Filter Pipeline Pattern

The strategy orchestration uses a **Filter Pipeline** pattern to evaluate trade signals.

- **Interface**: `Filter` (abstract base class)
- **Implementation**: `src/gpt_trader/features/live_trade/orchestrator/spot_filters.py`
- **Current Status**: The pipeline is currently a pass-through. Specific filter implementations (Volume, Momentum, Trend) have been removed and will be reintroduced as needed.
- **Usage**: `SpotFiltersMixin` provides the hook for these checks.

#### Signal Ensemble Architecture

The system now uses a **Signal Ensemble** approach to combine multiple trading signals.

- **Components**:
    - `Signal`: Individual trading signal (e.g., RSI, MACD).
    - `Ensemble`: Collection of signals with weights.
    - `Voter`: Logic to combine signal outputs into a final decision.
- **Location**: `src/gpt_trader/features/live_trade/strategies/ensemble.py`
- **Status**: Implemented and wired via the strategy factory + `TradingEngine`.

#### TUI Architecture (`gpt_trader/tui/`)

The Terminal User Interface is built on [Textual](https://textual.textualize.io/) and uses a modular CSS system.

**CSS System**

The TUI uses a concatenated CSS approach due to Textual not supporting `@import`:

- **Source modules**: `styles/theme/`, `styles/layout/`, `styles/components/`, `styles/widgets/`, `styles/screens/`
- **Build script**: `scripts/build_tui_css.py` concatenates modules in dependency order
- **Generated file**: `styles/main.tcss` (~2,200 lines) - DO NOT EDIT DIRECTLY

**Critical Design Constraints**

1. **Single Grid Definition**: The main layout grid (`#bento-grid`) must be defined in exactly one place (`layout/workspace.tcss`). Multiple grid definitions cause unpredictable tile spanning behavior.

2. **DEFAULT_CSS Cannot Use Variables**: Widget-level `DEFAULT_CSS` strings in Python files do NOT have access to TCSS variables (`$bg-primary`, `$accent`, etc.). Use hardcoded hex values with comments referencing the variable name:
   ```python
   DEFAULT_CSS = """
   MyWidget {
       background: #3B4252;  /* $bg-secondary */
       color: #ECEFF4;       /* $text-primary */
   }
   """
   ```

3. **Namespaced Class Names**: Use widget-specific class names to prevent style bleed:
   - `.widget-header` for widget headers (defined in `headers.tcss`)
   - `.screen-header` for screen-level headers
   - Avoid generic names like `.header`, `.value`, `.row`

4. **SCOPED_CSS Pattern**: Widgets that need global style access set `SCOPED_CSS = False`. Document the reason:
   ```python
   SCOPED_CSS = False  # Uses global styles from dashboard.tcss
   ```

**State Management**

- `StateRegistry` broadcasts `TuiState` to registered widgets via `on_state_updated()`
- Widgets implement `StateObserver` protocol for automatic state updates
- Delta tracking (`_changed_fields`) identifies which state components changed

**Services Layer**

The TUI services layer (`tui/services/`) provides:

- `AlertManager` - Threshold-based alert rules with cooldown, history, and notification integration
- `ExecutionTelemetryCollector` - Thread-safe order submission metrics (success rate, latency percentiles, retry tracking)
- `OnboardingService` - First-run wizard state management
- `FocusManager` - Keyboard navigation and focus ring management

**App Organization (Mixin Pattern)**

The main `TraderApp` class uses mixins to organize functionality into focused modules:

- `app_mode_flow.py` - Mode selection, credential validation, mode switching
- `app_lifecycle.py` - Mount/unmount lifecycle, initialization, cleanup
- `tui/app_bootstrap.py` - Bootstrap snapshot, read-only data feed startup
- `app_status.py` - Status updates, observer connections, state sync
- `app_actions.py` - Action methods and event handlers

Import `TraderApp` from `gpt_trader.tui.app`; mixins are internal implementation details.

**Data Freshness & Resilience UX**

The `staleness_helpers.py` module provides unified data trust signals:

- **Thresholds**: Fresh (<10s), Stale (10-30s), Critical (>30s or connection unhealthy)
- **Execution Health**: Circuit breaker status, success rate warnings (<95%), critical alerts (<80%)
- **Banner Priority**: Reconnecting → Degraded mode → Execution health → Data staleness
- **Empty States**: Standardized configurations for stopped, connecting, and error states

**Alert System**

Built-in alert rules with configurable cooldowns:

| Category | Rules |
|----------|-------|
| SYSTEM | `connection_lost`, `rate_limit_high`, `bot_stopped`, `circuit_breaker_open`, `execution_critical`, `execution_degraded`, `execution_p95_spike`, `execution_retry_high` |
| RISK | `reduce_only_active`, `daily_loss_warning` |
| POSITION | `large_unrealized_loss` |
| TRADE | `stale_open_orders`, `failed_orders`, `expired_orders` |

Recovery hints are provided via `notification_helpers.py` for actionable guidance.

**Rebuilding CSS**

After editing any `.tcss` module file:
```bash
python scripts/build_tui_css.py
```

### Derivatives Gate

**CFM Futures** (US-regulated): Available for approved US futures accounts. Enable via `TRADING_MODES=cfm` (or `spot,cfm`) and `CFM_ENABLED=1`; uses CDP authentication and CFM endpoints (`cfm_balance_summary`, `cfm_positions`, etc.).

**INTX Perpetuals**: Require international INTX account access. Enable via `COINBASE_ENABLE_INTX_PERPS=1` (legacy: `COINBASE_ENABLE_DERIVATIVES=1`). Code paths stay compiled for future use.

### Feature Slice Reference


#### Position Sizing (`features/intelligence/sizing/`)
- **Purpose:** Regime-aware position sizing combining Kelly Criterion, ATR-based volatility scaling, and risk budgeting.
- **Usage:**
    ```python
    from gpt_trader.features.intelligence.sizing import (
        PositionSizer,
        PositionSizingConfig,
        SizingResult,
    )

    config = PositionSizingConfig(...)
    sizer = PositionSizer(config)
    result: SizingResult = sizer.calculate_size(...)
    ```
- **Integration hooks:** Built to ingest market-regime detectors from `intelligence/regime/` without cross-slice imports.

### Key Design Principles

1. **Slice Isolation**: Production slices limit cross-dependencies; experimental ones stay sandboxed.
2. **Token Awareness**: Documentation highlights slice entry points so agents can load only what they need.
3. **Type Safety**: Shared interfaces defined in `features/brokerages/core/protocols.py`.
4. **Environment Separation**: GPT-Trader normalizes symbols to spot unless INTX derivatives access is detected.

### Order Execution Pipeline

The order execution pipeline ensures reliable order submission with proper ID tracking, telemetry, and error classification.

**Flow Diagram**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TradingEngine._cycle()                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TradingEngine._validate_and_place_order()                │
│                                                                             │
│  1. Degradation gate (pause + reduce-only allowance)                        │
│  2. Position sizing                                                         │
│  3. Reduce-only request gate (reject if no position)                        │
│  4. Security validation (hard limits)                                       │
│  5. Reduce-only mode clamp + check                                          │
│  6. Mark staleness gate (allow reduce-only if configured)                   │
│  7. OrderValidator: exchange rules → pre-trade validation                   │
│     slippage guard → order preview                                          │
│  8. Delegates to OrderSubmitter                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OrderSubmitter.submit_order()                         │
│                                                                             │
│  • Generates stable client_order_id (or uses provided)                      │
│  • Sets correlation context for tracing                                     │
│  • Delegates to BrokerExecutor                                              │
│  • Tracks order in open_orders list                                         │
│  • Records telemetry (success/rejection with classified reason)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BrokerExecutor._execute_broker_order()                  │
│                                                                             │
│  • Calls broker.place_order(client_id=submit_id, ...)                       │
│  • Passes client_order_id to broker for idempotency                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Broker (Coinbase/Mock)                             │
│                                                                             │
│  • Executes order via REST API                                              │
│  • Returns Order object with broker-assigned order_id                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Implementation Details**

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `TradingEngine` | `features/live_trade/engines/strategy.py` | Guard stack, validation, position tracking |
| `OrderSubmitter` | `features/live_trade/execution/order_submission.py` | ID generation, telemetry, open order tracking |
| `BrokerExecutor` | `features/live_trade/execution/broker_executor.py` | Broker API abstraction |
| `OrderEventRecorder` | `features/live_trade/execution/order_event_recorder.py` | Event store persistence |

**client_order_id Flow**

1. **Generation**: `OrderSubmitter._generate_submit_id()` creates `{bot_id}_{uuid}` format
2. **Propagation**: Passed to `broker.place_order(client_id=...)` for broker-side idempotency
3. **Tracking**: Used in logs, metrics, and event store for end-to-end tracing
4. **Retry Safety**: Callers can pass explicit `client_order_id` to ensure retries don't create duplicates

**Error Classification**

Rejection reasons are normalized via `normalize_rejection_reason()` to stable codes
(see `docs/OBSERVABILITY.md` for the canonical list). These codes feed into metrics
labels and event payloads for consistent telemetry.

### Live Trade Infrastructure

The live trade layer provides coordinated control across trading operations through modules in `features/live_trade/` and `app/`.

**Configuration & Symbol Management:**
- `app/config/` - Profile-aware configuration (`BotConfig`, profiles, validation)
- `app/config/controller.py` - Dynamic configuration management and hot-reloading
- `features/live_trade/symbols.py` - Symbol normalization, derivatives gating, and defaults

**Execution Coordination:**
- `features/live_trade/engines/strategy.py` - Main trading loop and guard stack
- `features/live_trade/execution/` subpackage:
  - `guard_manager.py` - Runtime guard management
  - `validation.py` - Pre-trade validation
  - `order_submission.py` - Order submission and recording
  - `state_collection.py` - Account state collection
  - `broker_executor.py` - Broker execution abstraction

**Strategy & Runtime Coordination:**
- `features/live_trade/orchestrator/` - Strategy lifecycle management and symbol processing
  - `orchestrator.py` - Main strategy orchestrator
  - `decision.py` - Trading decision logic
  - `spot_filters.py` - Spot-specific filtering rules

**Services & Telemetry:**
- `features/live_trade/orchestrator/spot_profile_service.py` - Spot trading profile loading and rule management
- `features/live_trade/telemetry/account.py` - Account metrics tracking and periodic snapshots
- `src/gpt_trader/monitoring/system/metrics.py` - System metrics collection
- `src/gpt_trader/monitoring/system/positions.py` - Position monitoring

**Infrastructure:**
- `features/live_trade/bot.py` - TradingBot facade coordinating all components
- `features/brokerages/mock/deterministic.py` - Deterministic broker for testing
- `features/brokerages/paper/hybrid.py` - Paper trading broker implementation
- `app/runtime/paths.py` - Runtime path resolution
- `app/bootstrap.py` - Bot creation helpers (`build_bot`, `bot_from_profile`)

## What's Actually Working

### ✅ Fully Operational
- Coinbase spot trading via Advanced Trade (REST/WebSocket); dev profile defaults to the deterministic broker stub and can be pointed at live APIs with `SPOT_FORCE_LIVE=1`
- Order placement/management through the `TradingEngine` guard stack (`_validate_and_place_order` -> `OrderSubmitter` -> `BrokerExecutor`)
- Account telemetry snapshots and cycle metrics persisted for monitoring
- Runtime safety rails: daily loss guard, liquidation buffer enforcement, mark staleness detection, volatility circuit breaker, correlation checks
- Active test suite (`uv run pytest --collect-only` to verify)

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
daily_loss_limit_pct: 1%
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

> Spot risk templates are archived; see `docs/archive/risk_templates/spot_top10.yaml`
> for example per-symbol caps and leverage=1 limits across the top-ten USD markets.

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

### Resilience & Retry Policy

The execution pipeline uses a configurable `RetryPolicy` for transient failures:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_attempts` | 3 | Maximum broker call attempts |
| `base_delay` | 0.5s | Initial backoff delay |
| `max_delay` | 30.0s | Maximum backoff delay |
| `exponential_base` | 2.0 | Backoff multiplier |

**Configuration:**
- Code: `BrokerExecutor(retry_policy=RetryPolicy(...))`
- Location: `features/live_trade/execution/broker_executor.py`

**Retry Classification:**
- **Retryable**: `TimeoutError`, `ConnectionError`, rate limits (429)
- **Non-retryable**: `ValueError` (rejections), `InvalidOrder`, insufficient funds

The `client_order_id` is preserved across retries for broker-side idempotency.

### Degradation Lifecycle

The `DegradationState` manages trading restrictions with monotonic progression:

```
NORMAL → REDUCE_ONLY → PAUSED → HALTED
```

**Pause Types:**

| Method | Scope | Use Case |
|--------|-------|----------|
| `pause_all(seconds)` | Global | System-wide issues |
| `pause_symbol(symbol, seconds)` | Per-symbol | Symbol-specific problems |

**Key Behaviors:**
- Pauses are time-bounded and auto-expire
- `allow_reduce_only=True` permits position-closing orders during pause
- Symbol pauses expire independently
- Guard failures can trigger automatic pause via `record_broker_failure()`

**Location:** `features/live_trade/degradation.py`

## Performance & Observability

- **Cycle Metrics**: persisted to `runtime_data/<profile>/metrics.json` and `runtime_data/<profile>/events.db`,
  exposed via the Prometheus exporter (`scripts/monitoring/export_metrics.py`) which reads events.db first,
  falling back to metrics.json/events.jsonl when needed. The live risk manager emits snapshot events
  consumed by dashboards and the monitoring stack.
- **Account Snapshots**: periodic telemetry via `CoinbaseAccountManager` with fee/limit tracking.
- **System Monitoring**: `src/gpt_trader/monitoring/system/` provides resource telemetry collectors used by
  the runtime guard manager and dashboards.
- **System Footprint**: bot process typically <50 MB RSS with sub-100 ms WebSocket latency in spot
  mode.
- **Test Discovery**: `uv run pytest --collect-only`

## Verification Path

1. **Regression Suite**: `uv run pytest -q`
2. **Smoke Test**: `uv run gpt-trader run --profile dev --dev-fast`
3. **Validation**: `uv run python scripts/production_preflight.py --profile dev --warn-only`

## Dependencies

### Core Requirements
- Python 3.12+
- uv (package manager)
- Coinbase Advanced Trade API (for perpetuals)
- CDP API keys (for production)

### Key Libraries
- `coinbase-advanced-py` - Official Coinbase SDK
- `pandas` - Data manipulation
- `scikit-learn` - Statistical utilities (ML integration future-ready)
- `websocket-client` - Real-time data
- `pydantic` - Data validation

## Future Roadmap

### Near Term (Q1 2025)
- [ ] Full WebSocket user event handling
- [ ] Durable state recovery
- [ ] Order modification flows
- [ ] Partial fill handling

### Medium Term (Q2 2025)
- [ ] Advanced order types (OCO, bracket orders)
- [ ] Portfolio-level risk management
- [ ] Real-time ML adaptation
- [ ] Enhanced Coinbase INTX integration

### Long Term (2025+)
- [ ] Distributed execution
- [ ] Options integration (when Coinbase supports)
- [ ] Institutional features
- [ ] Advanced derivatives strategies

---

*For implementation details, see [Trading Logic - Perpetuals](reference/trading_logic_perps.md)*
