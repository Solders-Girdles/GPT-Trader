# Tooling Quick Reference Guide

## Tool Categories Map

```
┌─────────────────────────────────────────────────────────────────┐
│                      GPT-TRADER TOOLING                          │
└─────────────────────────────────────────────────────────────────┘

FOUNDATION TIER (Core Utilities)
├── Async Tools (7 modules)
│   ├─ AsyncRetry ........................... Exponential backoff
│   ├─ AsyncRateLimiter ..................... Token bucket
│   ├─ AsyncCache ........................... TTL caching
│   ├─ AsyncBatchProcessor .................. Batch concurrency
│   ├─ gather_with_concurrency .............. Semaphore control
│   ├─ wait_for_first ....................... First completion
│   └─ Async/Sync bridges ................... Boundary crossing
│
├── Error Handling (Solid)
│   ├─ TradingError hierarchy ............... 13 error types
│   ├─ CircuitBreaker ....................... State machine
│   ├─ RecoveryStrategy enum ................ 5 modes (RETRY, FALLBACK, etc)
│   └─ log_error() .......................... Structured logging
│
└── Configuration (Solid)
    ├─ LiveTradeConfig ...................... Pydantic validators
    ├─ ConfigBaselinePayload ................ Snapshot + diff
    └─ EnvUtilParser ........................ Env parsing

ORCHESTRATION TIER (Service Coordination)
├── Coordinator Pattern
│   ├─ CoordinatorContext ................... Immutable snapshots
│   ├─ BaseCoordinator ...................... Lifecycle contract
│   ├─ CoordinatorRegistry .................. Orchestrator
│   ├─ RuntimeCoordinator ................... Runtime init
│   ├─ StrategyCoordinator .................. Strategy lifecycle
│   ├─ ExecutionCoordinator ................. Order execution
│   └─ TelemetryCoordinator ................. Metrics collection

MONITORING TIER (Observability)
├── Guards & Validation
│   ├─ ConfigurationGuardian ................ Drift detection
│   ├─ RuntimeGuardManager .................. Guard lifecycle
│   └─ MarketConditionFilters ............... Market validation
│
├── Health & Metrics
│   ├─ HealthChecker ....................... Component health
│   ├─ HealthCheckEndpoint .................. HTTP endpoints
│   ├─ PerformanceMonitor ................... Execution timing
│   ├─ PerformanceProfiler .................. CPU/memory
│   ├─ PerformanceCollector ................. Metrics aggregation
│   └─ ResourceMonitor ...................... System resources
│
└── Domain Monitors
    ├─ PerpsLiquidationMonitor .............. Liquidation risk
    └─ PerpsMarginMonitor ................... Margin tracking

LOGGING & TELEMETRY TIER
├── Structured Logging
│   ├─ StructuredLogger .................... Standardized fields
│   ├─ ConsoleLogger ........................ Rich output
│   ├─ log_operation ....................... Context manager
│   └─ LOG_FIELDS registry .................. Standardized names
│
└── Telemetry
    └─ emit_metric() ....................... Custom metrics

TRADING OPERATIONS TIER
├── TradingOperations ..................... Order placement + error handling
├── PositionManager ....................... Position lifecycle
└─ create_trading_operations() ........... Factory

STRATEGY TOOLS TIER (Developing)
├─ MarketConditionFilters ............... Market condition checks
├─ RiskGuards ........................... Risk validation
├─ StrategyEnhancements ................. Strategy features
└─ Factory functions ..................... Presets (conservative, aggressive)
```

## Usage Patterns Quick Reference

### Pattern 1: Async Resilience Stack
```python
@async_retry(max_attempts=3, base_delay=1.0)
@async_rate_limit(rate_limit=10.0, burst_limit=5)
@async_cache(ttl=300.0)
async def fetch_data(symbol: str):
    # Automatic retry, rate limiting, caching
    pass
```
**Usage**: Market data APIs, broker communication

---

### Pattern 2: Structured Operation Logging
```python
with log_operation("process_order", logger, symbol="BTC-USD", qty=1.5):
    # Automatic timing, logging, context
    result = await place_order(symbol, qty)
    # Logs: timing, parameters, results
```
**Usage**: Every major operation

---

### Pattern 3: Error Handling with Recovery
```python
try:
    result = await risky_operation()
except NetworkError as e:
    log_error(e)
    handler = get_error_handler()
    # Circuit breaker checks, recovery strategies
    result = await handler.recover(e)
```
**Usage**: API calls, external integrations

---

### Pattern 4: Coordinator Lifecycle
```python
context = coord.initialize(context)  # Build dependencies
tasks = await coord.start_background_tasks()  # Start async work
health = coord.health_check()  # Check health
await coord.shutdown()  # Cleanup
```
**Usage**: Orchestration layer

---

### Pattern 5: Configuration Management
```python
config = LiveTradeConfig(...)  # Pydantic validation
baseline = ConfigBaselinePayload.from_config(config)
# Later...
new_baseline = ConfigBaselinePayload.from_config(new_config)
diff = baseline.diff(new_baseline)  # Detect changes
```
**Usage**: Configuration tracking and drift detection

---

## Gap Analysis & Priorities

### Critical Gaps (Do First)

| Gap | Reason | Impact | Effort |
|-----|--------|--------|--------|
| Distributed Tracing | No correlation IDs | HIGH | MEDIUM |
| Metrics Aggregation | Scattered collection | HIGH | MEDIUM |
| Validation Rules | Duplicated logic | MEDIUM | LOW |
| Unified Logging | Multiple systems | MEDIUM | LOW |
| DI Framework | Manual wiring | MEDIUM | HIGH |

### Enhancement Gaps (Do Later)

| Gap | Reason | Impact | Effort |
|-----|--------|--------|--------|
| Bulkhead Pattern | Resource isolation | LOW | MEDIUM |
| Timeout Decorator | Operation bounds | MEDIUM | LOW |
| Circuit Breaker Decorator | Decorator pattern | MEDIUM | LOW |
| Adaptive Backoff | Smart retries | LOW | MEDIUM |
| Saga Framework | Multi-step operations | MEDIUM | HIGH |

---

## File Location Reference

### Async Tools
```
src/bot_v2/utilities/async_tools/
├── retry.py (AsyncRetry class)
├── rate_limit.py (AsyncRateLimiter class)
├── cache.py (AsyncCache class)
├── wrappers.py (Async/Sync bridges)
└── helpers.py (gather_with_concurrency, etc.)
```

### Error Handling
```
src/bot_v2/errors/
├── __init__.py (Error hierarchy)
└── handler.py (CircuitBreaker, RecoveryStrategy)
```

### Orchestration
```
src/bot_v2/orchestration/
├── coordinators/
│   ├── base.py (CoordinatorContext, BaseCoordinator)
│   ├── registry.py (CoordinatorRegistry)
│   ├── runtime.py (RuntimeCoordinator)
│   ├── strategy.py (StrategyCoordinator)
│   └── execution.py (ExecutionCoordinator)
└── coordinator_facades.py (Legacy facades)
```

### Monitoring
```
src/bot_v2/monitoring/
├── configuration_guardian.py
├── runtime_guards.py
├── health_checks.py
├── guards/ (base, manager, builtins)
└── health/ (registry, checks, endpoint)
```

### Logging
```
src/bot_v2/utilities/
├── logging_patterns.py (StructuredLogger, log_operation)
├── console_logging.py (ConsoleLogger)
└── telemetry.py (emit_metric)
```

### Configuration
```
src/bot_v2/config/
├── live_trade_config.py (LiveTradeConfig)
└── utilities/config.py (ConfigBaselinePayload)
```

### Strategy Tools
```
src/bot_v2/features/strategy_tools/
├── filters.py (MarketConditionFilters)
├── guards.py (RiskGuards)
└── enhancements.py (StrategyEnhancements)
```

---

## Implementation Priority Matrix

```
┌────────────────────────────────────────────────────────────┐
│ EFFORT (Days)                                              │
│     ^                                                      │
│ 5   │        DI Container        Saga Framework           │
│     │        (High Impact)       (Medium Impact)           │
│ 4   │                                                      │
│     │   Bulkhead Pattern      Config Hot-Reload           │
│ 3   │   Advanced Resilience                               │
│     │        Validation Rules Lib  Distributed Tracing    │
│ 2   │   Unified Logging              Metrics Store        │
│     │   (Quick Wins)                                       │
│ 1   │                                                      │
│     │                                                      │
│ 0   └─────────────────────────────────────────────────────│
│     LOW    MEDIUM    HIGH         VERY HIGH (Impact)       │
│
Legend:
  Bottom-Left: Quick wins (high impact, low effort) - DO FIRST
  Top-Left: Time-intensive (lower impact) - DO LATER
  Bottom-Right: High impact, reasonable effort - DO PHASE 1
  Top-Right: High impact but complex - DO PHASE 2+
```

---

## Maturity Assessment

```
Tool Category        │ Current │ Target │ Status
─────────────────────┼─────────┼────────┼──────────────────
Async Tools          │   8/10  │   9/10 │ ✓ Solid, minor gaps
Error Handling       │   8/10  │   9/10 │ ✓ Solid, underused
Configuration        │   8/10  │   9/10 │ ✓ Solid, no reload
Orchestration        │   7/10  │   9/10 │ ✗ Needs composition
Monitoring           │   5/10  │   8/10 │ ✗ Missing metrics hub
Logging              │   5/10  │   8/10 │ ✗ Needs unification
Strategy Tools       │   4/10  │   8/10 │ ✗ Not composable
Performance          │   4/10  │   8/10 │ ✗ Underutilized

OVERALL: 6.1/10 → Target 8.5/10
```

---

## Success Indicators

- [ ] All correlation IDs appear in logs
- [ ] Centralized metrics queryable
- [ ] No duplicate validation logic
- [ ] Decorators used for resilience patterns
- [ ] Unified logging across system
- [ ] DI container bootstraps 90%+ of services
- [ ] Bulkheads isolate components
- [ ] Health checks < 50ms
- [ ] Build time unchanged (< 5% growth)
- [ ] Test coverage 85%+

---

## Contact & Questions

For detailed information:
- **Executive Overview**: See TOOLING_EXECUTIVE_SUMMARY.md
- **Technical Deep Dive**: See TOOLING_LANDSCAPE_ANALYSIS.md
- **Navigation Guide**: See TOOLING_ANALYSIS_INDEX.md
