# Technical Debt Inventory

**Last Updated:** 2025-11-27
**Status:** Active tracking document

This document tracks identified technical debt for systematic cleanup. Items are prioritized by impact and complexity.

---

## Completed

| Date | Item | Action Taken |
|------|------|--------------|
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

### 3.1 Complex Mixin Inheritance
**Location:** `src/gpt_trader/features/brokerages/coinbase/rest_service.py`
**Issue:** Single class inherits from 5 mixins (~1161 LOC total)
```python
class CoinbaseRestService(
    CoinbaseRestServiceBase,  # 326 lines
    ProductRestMixin,         # 106 lines
    OrderRestMixin,           # 238 lines
    PortfolioRestMixin,       # 364 lines
    PnLRestMixin,             # 127 lines
):
```
**Impact:** Hard to understand, debug, and test
**Recommendation:** Consider composition over inheritance

### 3.2 God Class: ApplicationContainer
**Location:** `src/gpt_trader/app/container.py` (180 lines)
**Issue:** Handles broker instantiation, config, event stores, credentials, file I/O
**Impact:** Violates single responsibility, hard to test
**Recommendation:** Extract credential loading, separate factory concerns

### 3.3 God Module: Guards
**Location:** `src/gpt_trader/orchestration/execution/guards.py` (467 lines)
**Issue:** Handles runtime guard state, equity calculation, mark staleness, PnL tracking, liquidation, volatility, error handling
**Impact:** Too many responsibilities in one module
**Recommendation:** Split into focused modules

### 3.4 Duplicate Broker Implementations
**Locations:**
- `backtesting/simulation/broker.py` - `get_product()`
- `orchestration/deterministic_broker.py` - `get_product()`
- `brokerages/coinbase/rest/product.py` - `get_product()`
- `orchestration/trading_bot/bot.py` - `get_product()`

**Issue:** Four separate implementations with similar logic
**Recommendation:** Create unified product abstraction

### 3.5 Test Fallback in Production Code
**Location:** `src/gpt_trader/orchestration/execution/broker_executor.py` (lines 95-114)
**Issue:** Production code handles `TypeError` for legacy signatures, checks `isawaitable()` at runtime
**Impact:** Test-specific logic mixed into production paths
**Recommendation:** Clean interface, remove runtime type checking

### 3.6 Inconsistent Broker Abstraction
**Issue:** Two broker abstractions not interchangeable:
- Core Interfaces (classes): Used by backtesting
- REST Service (mixins): Used by live trading

**Recommendation:** Unify or explicitly document the separation

---

## Priority 4: Type Safety Gaps

### 4.1 Any-typed Core Objects
**Locations:**
- `orchestration/trading_bot/bot.py` (lines 24-42): `registry: Any`, `event_store: Any`, `orders_store: Any`
- `features/live_trade/engines/base.py` (lines 13-20): `registry: Any`, `broker: Any`, `runtime_state: Any`

**Issue:** Key domain objects use `Any` type hints
**Impact:** Defeats static type checking, runtime errors
**Recommendation:** Define proper protocols/interfaces

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
