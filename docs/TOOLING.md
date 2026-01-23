# Internal Tooling Reference

Quick reference for the internal utilities and patterns in the GPT-Trader codebase.

## Tool Categories

```
FOUNDATION TIER
├── Async Tools (retry, rate limiting, caching, batch processing)
├── Error Handling (TradingError hierarchy, CircuitBreaker, RecoveryStrategy)
└── Configuration (LiveTradeConfig, ConfigBaselinePayload)

ENGINE TIER
├── CoordinatorContext (immutable snapshots)
├── BaseEngine (lifecycle contract)
├── RuntimeEngine, TradingEngine
└── Telemetry helpers (health + streaming)

MONITORING TIER
├── Guards (ConfigurationGuardian, RuntimeGuardManager)
├── Health (HealthCheckRunner, HealthSignal/HealthSummary)
└── Domain Monitors (LiquidationMonitor)

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

### Engine Lifecycle
```python
context = engine.initialize(context)
tasks = await engine.start_background_tasks()
health = engine.health_check()
await engine.shutdown()
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

### Engines
```
src/gpt_trader/features/live_trade/engines/
├── base.py (BaseEngine, CoordinatorContext)
├── runtime/ (RuntimeEngine)
└── strategy.py (TradingEngine)
src/gpt_trader/features/live_trade/strategies/
└── base.py (StrategyProtocol, MarketDataContext)
```

### Monitoring
```
src/gpt_trader/monitoring/
├── configuration_guardian/ (guardian, drift detection)
├── guards/ (base, manager, builtins)
├── health_checks.py
└── health_signals.py
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
| Engines | 7/10 | 9/10 | Needs composition |
| Monitoring | 5/10 | 8/10 | Missing metrics hub |
| Logging | 5/10 | 8/10 | Needs unification |
