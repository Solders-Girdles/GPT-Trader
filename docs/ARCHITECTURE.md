# System Architecture

## ⚠️ Architecture Transition (Active)

**Important Note for Developers and Agents:**

This project is currently undergoing a major architectural migration from a legacy "Orchestration" pattern (monolithic builders) to a modern **Dependency Injection** pattern using `ApplicationContainer`.

- **Legacy Core:** `src/gpt_trader/orchestration/` (being phased out/refactored)
- **Modern Core:** `src/gpt_trader/app/` (Composition Root) and `src/gpt_trader/features/` (Vertical Slices)
- **Migration Guide:** See [MIGRATION_STATUS.md](MIGRATION_STATUS.md) for current migration progress and patterns.

While this document describes the **target architecture**, you will encounter "zombie" files, legacy shims, and competing patterns (e.g., `PerpsBotBuilder` vs `ApplicationContainer`) in the codebase. Always prefer the patterns defined in `src/gpt_trader/app` and `src/gpt_trader/features` over those in `src/gpt_trader/orchestration` where they conflict.

---
status: transition
last-updated: 2026-01-05
---

## Current State

GPT-Trader V2 is a production-ready Coinbase trading system supporting **spot** and **CFM futures** trading. INTX perpetuals code paths remain compiled and testable but require international account access.

> 📘 **Trust reminder:** Confirm this document's details against `docs/agents/Document_Verification_Matrix.md` before acting on them.

## Trading Capabilities Matrix

| Environment | Products | Authentication | API Version | WebSocket | Use Case |
|------------|----------|----------------|-------------|-----------|----------|
| **Production (spot)** | Spot (BTC-USD, ETH-USD, …) | HMAC or CDP | Advanced v3 | Real-time | Live spot trading |
| **Production (CFM)** | US Futures (BTC, ETH, SOL, etc.) | CDP (JWT) | Advanced v3 | Real-time | Regulated US futures |
| **Production (INTX)** | Perpetuals (international) | CDP (JWT) + `COINBASE_ENABLE_DERIVATIVES=1` | Advanced v3 | Real-time | Requires INTX account |
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
├── research/            # Research and backtesting evaluation
├── strategy_dev/        # Strategy development lab
└── strategy_tools/      # Shared helpers for strategy slices
```

Additional cross-cutting packages now live at the top level. These intentionally span feature slices:

```
src/gpt_trader/
├── errors/              # Centralized error hierarchy providing consistent exception types
├── monitoring/          # Runtime guards, configuration guardian, system logger
├── preflight/           # Production preflight verification and startup checks
├── security/            # Security primitives: input sanitization, secrets management
├── tui/                 # Terminal User Interface (Textual-based)
└── validation/          # Declarative validators and decorators
```

### High-Level Flow

```
CLI (coinbase-trader) → Config (BotConfig) → ApplicationContainer → TradingBot →
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

- `uv run coinbase-trader` invokes `gpt_trader.cli:main`, producing a `BotConfig` from
  CLI arguments and environment overrides.
- `ApplicationContainer` (`gpt_trader/app/container.py`) is the **canonical composition root**.
  It lazily initializes all services (broker, risk manager, event store, etc.) and wires
  them into `TradingBot` via `container.create_bot()`.
- `TradingBot` receives services directly from the container—no intermediate registry.

> **Note:** Legacy `gpt_trader/orchestration/bootstrap.py` still exists for backwards
> compatibility but internally delegates to `ApplicationContainer`. New code should use
> `ApplicationContainer` directly.

#### Container Sub-Containers

`ApplicationContainer` delegates to specialized sub-containers:

| Sub-Container | Services |
|---------------|----------|
| `ConfigContainer` | Config controller, profile loader |
| `ObservabilityContainer` | Notifications, health state, secrets manager |
| `PersistenceContainer` | Event store, orders store, runtime paths |
| `BrokerageContainer` | Broker, market data service, product catalog |
| `RiskValidationContainer` | Risk manager, validation failure tracker |

#### Strict Container Mode (Testing)

For tests, enable strict container mode to ensure proper DI usage:

```bash
GPT_TRADER_STRICT_CONTAINER=1 uv run pytest tests/unit
```

When enabled, `get_failure_tracker()` and similar service locators raise `RuntimeError`
instead of falling back to module-level singletons. This catches tests that bypass the
container.

The `tests/conftest.py` enables strict mode automatically via an autouse fixture. Tests
that intentionally verify fallback behavior can disable it:

```python
def test_fallback_behavior(monkeypatch):
    monkeypatch.delenv("GPT_TRADER_STRICT_CONTAINER", raising=False)
    # ... test fallback path
```

#### Deprecated Import Paths

The following modules are **deprecated** (removal target: **v3.0**) and emit a single-shot `DeprecationWarning` on import. Update to the canonical paths:

| Deprecated Path | Canonical Path |
|-----------------|----------------|
| `gpt_trader.orchestration.execution.degradation` | `gpt_trader.features.live_trade.degradation` |
| `gpt_trader.orchestration.configuration.risk.model` | `gpt_trader.features.live_trade.risk.config` |
| `gpt_trader.orchestration.configuration.bot_config` | `gpt_trader.app.config` |
| `gpt_trader.orchestration.live_execution.LiveExecutionEngine` | `TradingEngine.submit_order()` |

The deprecated modules remain as thin re-exports for backwards compatibility.

**Order Execution:** Use `TradingEngine.submit_order()` for the canonical guard stack. `LiveExecutionEngine.place_order()` and `OrderRouter.execute()` are deprecated.

> **Full deprecation tracker:** See [DEPRECATIONS.md](DEPRECATIONS.md) for removal timelines and migration checklists.

### Core Subsystems

| Module | Purpose |
|--------|---------|
| `gpt_trader/features/live_trade` | Main control loop, position tracking, and order routing |
| `gpt_trader/features/live_trade/risk/` | Risk management subpackage: position sizing, pre-trade validation, runtime monitoring, state management |
| `gpt_trader/orchestration/` | Core orchestration layer: config management, execution/runtime/strategy coordination, telemetry, reconciliation |
| `gpt_trader/orchestration/execution/` | Execution subpackage: guards, validation, order submission, state collection |
| `gpt_trader/features/brokerages/coinbase` | REST/WS integration for Coinbase Advanced Trade spot markets |
| `gpt_trader/features/brokerages/coinbase/client/` | Modular client package with mixins (accounts, orders, portfolio, market data) |
| `gpt_trader/features/brokerages/coinbase/rest/` | REST service layer: orders, portfolio, products, P&L calculation |
| `gpt_trader/features/intelligence/sizing` | Kelly-style sizing with regime awareness |
| `gpt_trader/monitoring` | Runtime guard orchestration, alert dispatch, system metrics |
| `gpt_trader/tui` | Terminal User Interface for monitoring and control |
| `gpt_trader/validation` | Predicate-based validators and input decorators |

#### Coordinator Pattern (new 2025-10)

The orchestration layer now follows a coordinator pattern that centralises lifecycle management and
dependency injection. See [ADR 002](architecture/decisions/002-coordinator-pattern.md) for the full
rationale.

**Key components**

- `CoordinatorContext` – immutable snapshot of broker, risk, stores, and orchestration services
- `BaseCoordinator` / `Coordinator` protocol – shared lifecycle contract (`initialize`, background
  tasks, `shutdown`, `health_check`)
- `CoordinatorRegistry` – registers coordinators, drives initialise/start/shutdown, and pushes
  updated context snapshots to every coordinator

**Registered coordinators**

- Runtime (`gpt_trader/orchestration/coordinators/runtime.py`) – broker & risk bootstrap, reconciliation
- Execution (`gpt_trader/orchestration/coordinators/execution.py`) – order placement, guards, locks
- Strategy (`gpt_trader/orchestration/coordinators/strategy.py`) – trading cycle, mark management,
  strategy orchestration
- Telemetry (`gpt_trader/orchestration/coordinators/telemetry.py`) – account telemetry, market monitor,
  websocket streaming

_TradingBot facade_

```
TradingBot
└── CoordinatorRegistry
    ├── RuntimeCoordinator
    ├── ExecutionCoordinator
    ├── StrategyCoordinator
    └── TelemetryCoordinator
```

**Lifecycle flow**

1. `TradingBot` constructs a `CoordinatorRegistry` using the initial context from `ApplicationContainer`.
2. `LifecycleManager.bootstrap()` calls `initialize_all()` so each coordinator can create
   dependencies and emit context updates.
3. `LifecycleManager.run()` delegates to `start_all_background_tasks()` to launch async work.
4. `LifecycleManager.shutdown()` invokes `shutdown_all()` for clean cancellation and teardown.

Legacy facades (e.g., `gpt_trader/orchestration/telemetry_coordinator.py`) remain as thin wrappers to
preserve existing imports while downstream consumers migrate to the new modules.

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

The `features/live_trade/risk/` subpackage provides comprehensive risk controls (refactored from 1,044-line monolith):

- `manager.py` (351 lines) - `LiveRiskManager` facade delegating to specialized components
- `position_sizing.py` (192 lines) - `PositionSizer` with Kelly Criterion and confidence modifiers
- `pre_trade_checks.py` (522 lines) - `PreTradeValidator` for synchronous order validation
- `runtime_monitoring.py` (314 lines) - `RuntimeMonitor` for async/periodic guard checks
- `state_management.py` (119 lines) - `StateManager` for reduce-only mode and state tracking

This modular design achieves 66% file size reduction with clear separation of concerns: sizing, validation, monitoring, and state management are independently testable.

#### Monitoring & Validation Framework

- **Validators** (`gpt_trader/validation`): the base `Validator` now accepts inline predicates and
  optional value coercion, enabling concise one-off validations while keeping legacy subclasses.
- **Runtime guards** (`gpt_trader/monitoring/runtime_guards.py`): guard evaluation supports rich
  comparison modes (`gt`, `lt`, `abs_gt`, etc.), warning bands, and contextual messaging to power
  both orchestration checks and monitoring dashboards.
- **Guard alert dispatcher** (`gpt_trader/features/live_trade/guard_errors.py`): wraps the lightweight
  alert manager to emit guard failures without depending on the retired alert stack. Restore the
  multi-channel router from the legacy bundle (`docs/archive/legacy_recovery.md`) if you still need
  email, Slack, or PagerDuty integrations.
- **Risk metrics aggregation** (`gpt_trader/features/live_trade/risk_metrics.py`): periodic EventStore
  snapshots feed into the monitoring stack for dashboards and analytics.

#### Filter Pipeline Pattern

The strategy orchestration uses a **Filter Pipeline** pattern to evaluate trade signals.

- **Interface**: `Filter` (abstract base class)
- **Implementation**: `src/gpt_trader/orchestration/strategy_orchestrator/spot_filters.py`
- **Current Status**: The pipeline is currently a pass-through. Specific filter implementations (Volume, Momentum, Trend) have been removed and will be reintroduced as needed.
- **Usage**: `SpotFiltersMixin` provides the hook for these checks.

#### Signal Ensemble Architecture

The system now uses a **Signal Ensemble** approach to combine multiple trading signals.

- **Components**:
    - `Signal`: Individual trading signal (e.g., RSI, MACD).
    - `Ensemble`: Collection of signals with weights.
    - `Voter`: Logic to combine signal outputs into a final decision.
- **Location**: `src/gpt_trader/features/strategy/ensemble/`
- **Status**: Implemented and integrated into `StrategyEngine`.

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
- `app_bootstrap.py` - Bootstrap snapshot, read-only data feed startup
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

**CFM Futures** (US-regulated): Available now. No special gate required - use CDP authentication and CFM endpoints (`cfm_balance_summary`, `cfm_positions`, etc.).

**INTX Perpetuals**: Remain behind `COINBASE_ENABLE_DERIVATIVES` flag and require international INTX account access. Code paths stay compiled for future use.

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
3. **Type Safety**: Shared interfaces defined in `features/brokerages/core/interfaces.py`.
4. **Environment Separation**: GPT-Trader normalizes symbols to spot unless INTX derivatives access is detected.

### Orchestration Infrastructure

The orchestration layer provides coordinated control across trading operations through specialized modules:

**Configuration & Symbol Management:**
- `configuration/` - Profile-aware configuration (`BotConfig`, profiles, validation)
- `config_controller.py` - Dynamic configuration management and hot-reloading
- `symbols.py` - Symbol normalization, derivatives gating, and profile-specific defaults

**Execution Coordination:**
- `live_execution.py` - Main execution engine facade
- `execution/` subpackage:
  - `guard_manager.py` - Runtime guard management
  - `validation.py` - Pre-trade validation
  - `order_submission.py` - Order submission and recording
  - `state_collection.py` - Account state collection
  - `broker_executor.py` - Broker execution abstraction

**Strategy & Runtime Coordination:**
- `strategy_orchestrator/` - Strategy lifecycle management and symbol processing
  - `orchestrator.py` - Main strategy orchestrator
  - `decision.py` - Trading decision logic
  - `spot_filters.py` - Spot-specific filtering rules

**Services & Telemetry:**
- `spot_profile_service.py` - Spot trading profile loading and rule management
- `account_telemetry.py` - Account metrics tracking and periodic snapshots
- `system_monitor_metrics.py` - System metrics collection
- `system_monitor_positions.py` - Position monitoring

**Infrastructure:**
- `trading_bot/bot.py` - Main orchestrator coordinating all components
- `deterministic_broker.py` - Deterministic broker for testing
- `hybrid_paper_broker.py` - Paper trading broker implementation
- `runtime_paths.py` - Runtime path resolution
- `bootstrap.py` - Bot creation helpers (`build_bot`, `bot_from_profile`)

## What's Actually Working

### ✅ Fully Operational
- Coinbase spot trading via Advanced Trade (REST/WebSocket); dev profile defaults to the deterministic broker stub and can be pointed at live APIs with `SPOT_FORCE_LIVE=1`
- Order placement/management through `LiveExecutionEngine`
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

- **Cycle Metrics**: persisted to `var/data/coinbase_trader/<profile>/metrics.json` and exposed via the
  Prometheus exporter (`scripts/monitoring/export_metrics.py`). The live risk manager now emits
  snapshot events consumed by dashboards and the monitoring stack.
- **Account Snapshots**: periodic telemetry via `CoinbaseAccountManager` with fee/limit tracking.
- **System Monitoring**: `gpt_trader/monitoring/system/` provides resource telemetry collectors used by
  the runtime guard manager and dashboards.
- **System Footprint**: bot process typically <50 MB RSS with sub-100 ms WebSocket latency in spot
  mode.
- **Test Discovery**: `uv run pytest --collect-only`

## Verification Path

1. **Regression Suite**: `uv run pytest -q`
2. **Smoke Test**: `uv run coinbase-trader run --profile dev --dev-fast`
3. **Validation**: `python scripts/validation/verify_core.py --check all`

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
