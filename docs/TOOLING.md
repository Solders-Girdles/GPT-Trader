# Internal Tooling Reference

Quick reference for the internal utilities and patterns in the GPT-Trader codebase.

## Tool Categories

```
FOUNDATION TIER
├── Async Tools (retry, rate limiting, caching, batch processing)
├── Error Handling (TradingError hierarchy, CircuitBreaker, RecoveryStrategy)
└── Configuration (LiveTradeConfig, ConfigBaselinePayload)

ORCHESTRATION TIER
├── CoordinatorContext (immutable snapshots)
├── BaseCoordinator (lifecycle contract)
├── RuntimeCoordinator, StrategyCoordinator, ExecutionCoordinator
└── TelemetryCoordinator

MONITORING TIER
├── Guards (ConfigurationGuardian, RuntimeGuardManager)
├── Health (HealthChecker, PerformanceMonitor)
└── Domain Monitors (LiquidationMonitor, MarginMonitor)

LOGGING TIER
├── StructuredLogger, ConsoleLogger
├── log_operation context manager
└── emit_metric() for custom metrics
```

## Common Patterns

### Async Resilience Stack
```python
@async_retry(max_attempts=3, base_delay=1.0)
@async_rate_limit(rate_limit=10.0, burst_limit=5)
@async_cache(ttl=300.0)
async def fetch_data(symbol: str):
    pass
```

### Structured Operation Logging
```python
with log_operation("process_order", logger, symbol="BTC-USD", qty=1.5):
    result = await place_order(symbol, qty)
```

### Error Handling with Recovery
```python
try:
    result = await risky_operation()
except NetworkError as e:
    log_error(e)
    handler = get_error_handler()
    result = await handler.recover(e)
```

### Coordinator Lifecycle
```python
context = coord.initialize(context)
tasks = await coord.start_background_tasks()
health = coord.health_check()
await coord.shutdown()
```

## File Locations

### Async Tools
```
src/gpt_trader/utilities/async_tools/
├── retry.py (AsyncRetry)
├── rate_limit.py (AsyncRateLimiter)
├── cache.py (AsyncCache)
└── helpers.py (gather_with_concurrency)
```

### Error Handling
```
src/gpt_trader/errors/
├── __init__.py (Error hierarchy)
└── handler.py (CircuitBreaker, RecoveryStrategy)
```

### Orchestration
```
src/gpt_trader/orchestration/
├── coordinators/ (base, registry, runtime, strategy, execution, telemetry)
└── trading_bot/bot.py
```

### Monitoring
```
src/gpt_trader/monitoring/
├── configuration_guardian.py
├── runtime_guards.py
├── guards/ (base, manager, builtins)
└── health/ (registry, checks, endpoint)
```

### Logging
```
src/gpt_trader/utilities/
├── logging_patterns.py (StructuredLogger, log_operation)
├── console_logging.py (ConsoleLogger)
└── telemetry.py (emit_metric)
```

## Maturity Assessment

| Category | Current | Target | Status |
|----------|---------|--------|--------|
| Async Tools | 8/10 | 9/10 | Solid |
| Error Handling | 8/10 | 9/10 | Solid |
| Configuration | 8/10 | 9/10 | Solid |
| Orchestration | 7/10 | 9/10 | Needs composition |
| Monitoring | 5/10 | 8/10 | Missing metrics hub |
| Logging | 5/10 | 8/10 | Needs unification |
