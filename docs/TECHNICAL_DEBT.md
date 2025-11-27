# Technical Debt Inventory

**Last Updated:** 2025-11-27
**Status:** Active tracking document

This document tracks identified technical debt for systematic cleanup. Items are prioritized by impact and complexity.

---

## Completed

| Date | Item | Action Taken |
|------|------|--------------|
| 2025-11-27 | Type safety: Core objects | Created `BrokerProtocol`, `RiskManagerProtocol`, `ServiceRegistryProtocol` |
| 2025-11-27 | Any-typed TradingBot/CoordinatorContext | Replaced with Protocol-based types |
| 2025-11-27 | Duplicate get_product() | Unified via `BrokerProtocol` interface |
| 2025-11-27 | `HMACAuth` class | Deprecated with warning - use `SimpleAuth` or `CDPJWTAuth` |
| 2025-11-27 | `CDPJWTAuth` incomplete implementation | Completed with `generate_jwt()` and `get_headers()` methods |
| 2025-11-27 | Legacy Bundling Script | Deleted `scripts/maintenance/create_legacy_bundle.py` |
| 2025-11-27 | AlertLevel Backward Compat Alias | Removed from `monitoring/interfaces.py` |
| 2025-11-27 | Empty Console Functions | Removed 6 stub functions from `features/data/data.py` |
| 2025-11-27 | `check_correlation_risk()` no-op | Removed method and its call in guards.py |
| 2025-11-27 | Health Check Stubs | Removed `HealthChecker`, `HealthCheckEndpoint`, `setup_basic_health_checks` |
| 2025-11-27 | `optimize_imports()` placeholder | Removed from profiling.py |
| 2025-11-27 | Duplicate `SimpleAuth` in security/ | Deleted `security/simple_auth.py` (unused duplicate) |
| 2025-11-26 | `data_providers` module (390 LOC) | Deleted - zero imports |
| 2025-11-26 | `backtest_production_example.py` | Deleted - referenced non-existent optimize module |
| 2025-11-26 | Empty directories (optimize, position_sizing/kelly, test dirs) | Deleted |
| 2025-11-26 | `reproduce_strategy.py` import | Fixed import path |
| 2025-11-27 | Guards module decomposition | Refactored 467 LOC → guards/ subpackage with Protocol-based guards |
| 2025-11-27 | GuardStateCache extraction | Extracted temporal caching to `guards/cache.py` |
| 2025-11-27 | Individual guard classes | Created 6 Protocol-compliant guards: DailyLoss, LiquidationBuffer, MarkStaleness, RiskMetrics, Volatility, PnLTelemetry |
| 2025-11-27 | Integration mode deprecation | Added deprecation warning for `INTEGRATION_TEST_MODE` env var |
| 2025-11-27 | StateCollector explicit mode | Added `integration_mode` parameter to avoid env var dependency |
| 2025-11-27 | Integration mode pattern removal | Completed full removal of `INTEGRATION_TEST_MODE`, `isawaitable()` checks, and legacy signature fallback |

---

## Priority 1: Legacy Patterns

### 1.1 Multiple Authentication Implementations
**Location:** `src/gpt_trader/features/brokerages/coinbase/auth.py`
**Issue:** Three auth methods coexist: `HMACAuth`, `CDPJWTAuth`, `SimpleAuth`
**Status:** `HMACAuth` deprecated (warning emitted), `CDPJWTAuth` now complete
**Next Step:** Remove `HMACAuth` entirely in future cleanup

### 1.2 Removed Provider Error Handlers
**Location:** `src/gpt_trader/data_providers/__init__.py` (lines 347-353)
**Issue:** Error messages for Alpaca/Coinbase providers that were removed
**Status:** Module deleted 2025-11-26

---

## Priority 2: Placeholder/Stub Code

### 2.1 No-op check_changes() (Deferred)
**Location:** `src/gpt_trader/monitoring/configuration_guardian/state_validator.py` (line 28)
**Issue:** Returns empty list, validates nothing
**Status:** Cannot remove - required by abstract base class `ConfigurationMonitor`
**Next Step:** Either implement properly or refactor base class interface

---

## Priority 3: Architectural Debt

### 3.1 CoinbaseRestService Mixin Architecture ✓
**Location:** `src/gpt_trader/features/brokerages/coinbase/rest_service.py`
**Current State:** Already uses composition via mixins - each mixin is a separate file:
- `rest/base.py` (326 LOC) - CoinbaseRestServiceBase
- `rest/product.py` (106 LOC) - ProductRestMixin
- `rest/orders.py` (238 LOC) - OrderRestMixin
- `rest/portfolio.py` (364 LOC) - PortfolioRestMixin
- `rest/pnl.py` (127 LOC) - PnLRestMixin

**Status:** Adequate - mixins are well-separated, each focused on single concern

### 3.2 ApplicationContainer
**Location:** `src/gpt_trader/app/container.py` (180 lines)
**Current State:** Simple lazy-loading DI container with clear responsibilities
- `create_brokerage()` factory already extracted
- Each component is a lazy property
- Well-tested (15 test cases)

**Status:** Adequate - not a god class, reasonable size and complexity

### 3.3 Guards Module ✓ COMPLETED
**Location:** `src/gpt_trader/orchestration/execution/guards/` (subpackage)
**Completed:** 2025-11-27

Refactored from single 467 LOC file to modular subpackage:
- `protocol.py` - Guard Protocol + RuntimeGuardState dataclass
- `cache.py` - GuardStateCache for temporal caching
- `daily_loss.py` - DailyLossGuard
- `liquidation_buffer.py` - LiquidationBufferGuard
- `mark_staleness.py` - MarkStalenessGuard
- `risk_metrics.py` - RiskMetricsGuard
- `volatility.py` - VolatilityGuard
- `pnl_telemetry.py` - PnLTelemetryGuard
- `guard_manager.py` - GuardManager orchestrator (moved from guards.py)

### 3.4 Duplicate Broker Implementations
**Locations:**
- `backtesting/simulation/broker.py` - `get_product()`
- `orchestration/deterministic_broker.py` - `get_product()`
- `brokerages/coinbase/rest/product.py` - `get_product()`
- `orchestration/trading_bot/bot.py` - `get_product()`

**Issue:** Four separate implementations with similar logic
**Status:** Partially addressed - `BrokerProtocol` now defines unified interface
**Next Step:** Consider shared implementation via composition

### 3.5 Integration Mode Pattern ✓ COMPLETED
**Completed:** 2025-11-27

Removed anti-pattern of environment-variable-triggered runtime behavior:
- Removed `INTEGRATION_TEST_MODE` from `tests/constants.py`
- Removed `inspect.isawaitable()` runtime checks from `state_collection.py`
- Removed legacy signature fallback from `broker_executor.py`
- Simplified to explicit `integration_mode` parameter only
- Deleted ~800 lines of legacy test code

### 3.6 Inconsistent Broker Abstraction
**Issue:** Two broker abstractions not interchangeable:
- Core Interfaces (classes): Used by backtesting
- REST Service (mixins): Used by live trading

**Recommendation:** Unify or explicitly document the separation

---

## Priority 4: Type Safety (Improved)

### 4.1 Protocol Definitions Added
**New Files:**
- `features/brokerages/core/protocols.py` - `BrokerProtocol`, `ExtendedBrokerProtocol`, `MarketDataProtocol`
- `features/live_trade/risk/protocols.py` - `RiskManagerProtocol`
- `orchestration/protocols.py` - `ServiceRegistryProtocol`, `RuntimeStateProtocol`, `EventStoreProtocol`

**Updated Files:**
- `orchestration/trading_bot/bot.py` - Uses Protocol types for registry, broker, risk_manager
- `features/live_trade/engines/base.py` - `CoordinatorContext` uses Protocol types
- `orchestration/service_registry.py` - `ServiceRegistry` typed with Protocols
- `orchestration/execution/broker_executor.py` - Uses `BrokerProtocol`

**Remaining:** Some `Any` types remain for `account_manager`, `account_telemetry`, `orders_store`

---

## Priority 5: Stale Documentation

### 5.1 Backtesting Guide
**Location:** `docs/guides/backtesting.md`
**Issue:** References non-existent `features.optimize` module
**Status:** Needs update to reflect current implementation

---

## Excluded (Intentional Design)

- `validation/calculation_validator.py` - `manual_backtest_example()` retained for compatibility
- `backtesting/types.py` - Feature flags `enable_golden_path_validation`, `enable_chaos_testing` are disabled by default (experimental)
- Commented debug logging in `brokerages/coinbase/utilities.py` - Development convenience

---

## How to Use This Document

1. **Before starting new work:** Check if it can address items here
2. **When refactoring:** Update completed section
3. **When discovering debt:** Add to appropriate priority section
4. **Monthly review:** Re-prioritize based on current focus
