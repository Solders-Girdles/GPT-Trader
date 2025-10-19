# GPT-Trader Codebase Tooling Landscape Analysis

## Executive Summary

The GPT-Trader codebase has developed a well-organized but compartmentalized tooling architecture spanning 259 Python files across utilities, orchestration, monitoring, and domain features. The system uses a **coordinator pattern** for orchestration, a comprehensive **utilities layer** for common operations, and specialized **monitoring/performance tools**. However, there are opportunities to consolidate redundancies, improve cross-component integration, and fill critical gaps in observability and resilience tooling.

---

## 1. CURRENT TOOLS AND UTILITIES INVENTORY

### 1.1 Async Utilities (`src/bot_v2/utilities/async_tools/`)

**Location**: `/Users/rj/PycharmProjects/GPT-Trader/src/bot_v2/utilities/async_tools/`

**Components**:
- **AsyncRetry** – Exponential backoff retry mechanism
- **AsyncRateLimiter** – Token-bucket rate limiting with burst support
- **AsyncCache** – TTL-based async-safe caching with cleanup
- **AsyncContextManager** – Async context management utilities
- **AsyncToSyncWrapper / SyncToAsyncWrapper** – Async/sync boundary bridging
- **AsyncBatchProcessor** – Batch processing with concurrency control
- **gather_with_concurrency** – Controlled parallelism with semaphores
- **wait_for_first** – First-completion semantics
- **run_in_thread** – Thread-based blocking operations

**Usage Pattern**: Decorators and class-based interfaces
```python
@async_retry(max_attempts=3, base_delay=1.0)
@async_rate_limit(rate_limit=10.0, burst_limit=5)
@async_cache(ttl=300.0)
async def expensive_operation(): ...
```

### 1.2 Logging and Monitoring (`src/bot_v2/utilities/`)

**Components**:
- **ConsoleLogger** – Rich console output with domain-specific categories
- **StructuredLogger** – Standardized structured logging with context fields
- **log_operation** – Context manager for operation timing and logging
- **emit_metric** – Telemetry emission
- **PerformanceMonitor** – Execution time and resource tracking
- **PerformanceProfiler** – CPU/memory profiling
- **ResourceMonitor** – System resource tracking (psutil-based)
- **PerformanceTimer** – Granular timing utilities
- **PerformanceCollector** – Metrics aggregation

**Key Files**:
- `logging_patterns.py` (100+ lines of structured logging)
- `console_logging.py` (Rich terminal output)
- `telemetry.py` (Metric emission)
- `performance/` subdirectory (5 modules)

### 1.3 Trading Operations (`src/bot_v2/utilities/trading_operations.py`)

**Classes**:
- **TradingOperations** – Order placement, position management, unified error handling
- **PositionManager** – Position tracking and lifecycle
- **create_position_manager()** – Factory function
- **create_trading_operations()** – Factory function

**Features**:
- Integrated error handling with recovery strategies
- Retry logic with circuit breaker patterns
- Standardized validation and logging

### 1.4 Error Handling (`src/bot_v2/errors/`)

**Error Hierarchy**:
- Base: `TradingError` (with context, timestamps, tracebacks)
- Domain-specific: `DataError`, `ConfigurationError`, `ValidationError`, `ExecutionError`, `NetworkError`, `InsufficientFundsError`, `StrategyError`, `BacktestError`, `OptimizationError`, `RiskLimitExceeded`, `TimeoutError`, `SliceIsolationError`, `AggregateError`

**Recovery Infrastructure** (`handler.py`):
- **RecoveryStrategy** enum (RETRY, FALLBACK, CIRCUIT_BREAK, FAIL_FAST, DEGRADE)
- **RetryConfig** – Exponential backoff configuration
- **CircuitBreakerConfig** – Failure thresholds and recovery timeouts
- **CircuitBreaker** implementation (state machine pattern)

**Error Handler Function**:
- `get_error_handler()` – Singleton error handler
- `log_error()` – Structured error logging
- `handle_error()` – Exception normalization

### 1.5 Strategy Tools (`src/bot_v2/features/strategy_tools/`)

**Components**:
- **MarketConditionFilters** – Spread, depth, volume, RSI checks
- **RiskGuards** – Liquidation distance, slippage impact validation
- **StrategyEnhancements** – Strategy-specific feature support
- Factory functions: `create_conservative_filters()`, `create_aggressive_filters()`, `create_standard_risk_guards()`

**Pattern**: Dataclass-based configuration with boolean-returning validation methods

### 1.6 Orchestration Coordinators (`src/bot_v2/orchestration/coordinators/`)

**Base Pattern** (`base.py`):
- **CoordinatorContext** – Immutable dependency snapshot
- **Coordinator** protocol – Lifecycle contract
- **BaseCoordinator** – Reusable base implementation
- **HealthStatus** – Health check result

**Coordinator Implementations**:
- **RuntimeCoordinator** – Runtime initialization and settings
- **StrategyCoordinator** – Strategy lifecycle
- **ExecutionCoordinator** – Order execution and reconciliation
- **TelemetryCoordinator** – Metrics collection
- **CoordinatorRegistry** – Lifecycle orchestration

**Key Pattern**: Context propagation between coordinators
```python
context = CoordinatorContext(config=..., registry=..., ...)
registry = CoordinatorRegistry(context)
for coord in coordinators:
    context = coord.initialize(context)  # Returns updated context
```

### 1.7 Monitoring and Guards (`src/bot_v2/monitoring/`)

**Components**:
- **ConfigurationGuardian** – Configuration drift detection
- **HealthChecker** – Component health checks
- **HealthCheckEndpoint** – HTTP health endpoints
- **RuntimeGuardManager** – Runtime guard lifecycle
- **Alert** / **AlertSeverity** – Alert types
- **PerpsLiquidationMonitor** – Perps-specific liquidation tracking
- **PerpsMarginMonitor** – Margin tracking

### 1.8 Configuration Utilities (`src/bot_v2/config/`, `src/bot_v2/utilities/config.py`)

**Components**:
- **ConfigBaselinePayload** – Configuration snapshots with diffing
- **EnvUtilParser** – Environment variable parsing (boolean, env, dynamic)
- **live_trade_config.py** – Pydantic models with 14+ field validators
- **path_registry.py** – Path management

---

## 2. USAGE FREQUENCY AND PATTERNS

### 2.1 Import Analysis

**High-frequency utilities** (found in 30+ files):
- `from bot_v2.utilities import` – General utilities
- `from bot_v2.utilities.async_tools import` – Async patterns
- `from bot_v2.logging_patterns import log_operation, get_logger`
- Error handling classes

**Test Coverage**:
- 209 test files (81% of source files have tests)
- 2 dedicated async utility test files with comprehensive coverage
- Strong focus on error handling tests

### 2.2 Common Usage Patterns

**Pattern 1: Async Operations with Combined Decorators**
```python
@async_retry(max_attempts=3)
@async_rate_limit(rate_limit=10.0)
async def fetch_data(): ...
```
**Found in**: Market data services, API clients

**Pattern 2: Context Managers for Logging**
```python
with log_operation("place_order", logger, symbol=symbol):
    # operation code
```
**Found in**: 15+ orchestration files

**Pattern 3: Factory Pattern for Tool Creation**
```python
operations = create_trading_operations(broker, risk_manager)
filters = create_conservative_filters()
guards = create_standard_risk_guards()
```
**Found in**: Orchestration, strategy configuration

**Pattern 4: Coordinator Lifecycle**
```python
context = coord.initialize(context)
tasks = await coord.start_background_tasks()
await coord.shutdown()
health = coord.health_check()
```
**Found in**: Every coordinator implementation

### 2.3 Dependency Flow

```
┌─────────────────────────────────────────────────┐
│          Orchestration Layer                    │
│  (perps_bot.py, coordinators/)                  │
└────────────────┬────────────────────────────────┘
                 │
         ┌───────┴────────┐
         │                │
    ┌────▼─────┐    ┌────▼──────────────────┐
    │Utilities │    │Monitoring/Guards      │
    │- async   │    │- ConfigurationGuardian│
    │- logging │    │- RuntimeGuardManager  │
    │- trading │    │- HealthChecker        │
    │ ops      │    └─────┬──────────────────┘
    └────┬─────┘          │
         │                │
    ┌────┴────────────────┘
    │
    ├─ Error Handling (errors/)
    ├─ Trading Operations (trading_operations.py)
    ├─ Strategy Tools (strategy_tools/)
    ├─ Configuration (config/, utilities/config.py)
    └─ Performance (utilities/performance/)
```

---

## 3. TOOL UTILIZATION ANALYSIS

### 3.1 Heavily Used Components

**Async Utilities**:
- `gather_with_concurrency()` – Used in batch order operations, market data fetching
- `async_retry` decorator – Market data APIs, broker communication
- `async_rate_limit` – Coinbase API rate limiting
- `AsyncCache` – Market data caching

**Logging/Monitoring**:
- `log_operation()` – Every major operation has context manager
- `ConsoleLogger` – Rich terminal output during development
- `PerformanceMonitor` – API operation timing, database operation profiling
- Structured logging across 30+ files

**Error Handling**:
- All domain errors inherit from `TradingError` base class
- Circuit breaker used in broker communication
- Recovery strategies in execution layer

### 3.2 Underutilized Components

**Identified gaps**:
1. **Decorator Usage**: Limited use of decorators for cross-cutting concerns
   - Only `@lru_cache` and `@field_validator` found in current code
   - No `@monitor_trading_operation` or `@profile_performance` decorators in production use

2. **StrategyTools**: Present but not heavily integrated
   - `MarketConditionFilters` and `RiskGuards` defined but minimal active usage
   - Suggest potential expansion opportunity

3. **Performance Monitoring**: Infrastructure exists but underutilized
   - `PerformanceProfiler` and `ResourceMonitor` not heavily integrated
   - Health check infrastructure incomplete for some components

4. **Resilience Patterns**:
   - Circuit breaker pattern implemented but rarely explicit in code
   - Fallback strategies rarely used

---

## 4. AREAS BENEFITING FROM NEW TOOLS/UTILITIES

### 4.1 Critical Gaps

**1. Distributed Tracing and Request Correlation**
- **Problem**: Multi-step operations lack end-to-end tracing
- **Current State**: Individual log entries, no correlation IDs
- **Recommendation**:
  - Add `TraceContext` class with correlation IDs
  - Implement `@trace_operation` decorator
  - Integrate with structured logging

**2. Metrics Aggregation and Analytics**
- **Problem**: Scattered performance metrics, no centralized collection
- **Current State**: Individual monitor instances, no aggregation
- **Recommendation**:
  - Build `MetricsAggregator` for time-series collection
  - Create `MetricsStore` interface (pluggable: file, Redis, Prometheus)
  - Implement `@record_metric` decorator

**3. Dependency Injection Framework**
- **Problem**: Manual service registry, verbose factory functions
- **Current State**: `ServiceRegistry` exists but minimal DI capability
- **Recommendation**:
  - Add `DIContainer` class with scope support
  - Implement `@injectable` and `@inject` decorators
  - Support lifecycle management (singleton, transient, scoped)

**4. Batch Operations Framework**
- **Problem**: No unified batch processing abstraction
- **Current State**: `AsyncBatchProcessor` exists, but limited integration
- **Recommendation**:
  - Create `BatchOperation` protocol with retry/rollback
  - Implement `BatchExecutor` with transaction-like semantics
  - Add `@batch_operation` decorator

**5. Configuration Validation Pipeline**
- **Problem**: Config validation scattered across multiple validators
- **Current State**: 14 validators in `live_trade_config.py`, no composition
- **Recommendation**:
  - Build `ValidationPipeline` with composable validators
  - Create `ValidationRule` and `RuleSet` abstractions
  - Implement `@validation_rule` decorator

**6. Cache Management and Invalidation**
- **Problem**: Multiple caching strategies (AsyncCache, lru_cache), no unified invalidation
- **Current State**: Multiple cache implementations
- **Recommendation**:
  - Build `CacheManager` with invalidation policies
  - Implement cache warming strategies
  - Add `@cache_invalidate` decorator for cache busting

**7. Observability Dashboard/Export**
- **Problem**: No centralized observability dashboard or export mechanism
- **Current State**: Health checks exist, no aggregation
- **Recommendation**:
  - Create `ObservabilityExporter` for metrics/health export
  - Implement Prometheus metrics exporter
  - Add `@observable_component` decorator

### 4.2 Enhancement Opportunities

**1. State Machine Framework**
- **Current**: Circuit breaker uses manual state transitions
- **Recommendation**: Create reusable `StateMachine` class
- **Use Cases**: Order lifecycle, coordinator state, risk mode transitions

**2. Event Sourcing Utilities**
- **Current**: Basic event store exists
- **Recommendation**: Add event handlers, saga patterns, replay utilities

**3. Bulkhead Pattern Implementation**
- **Current**: Rate limiting exists
- **Recommendation**: Add thread pool and resource isolation patterns

**4. Saga/Workflow Orchestration**
- **Current**: Coordinators manage lifecycle
- **Recommendation**: Build transaction-like saga support for multi-step operations

---

## 5. COMMON INTERACTION PATTERNS

### 5.1 Orchestration Pattern (Dominant)

```python
# Pattern: Coordinator initialization chain
class MyCoordinator(BaseCoordinator):
    def initialize(self, context: CoordinatorContext) -> CoordinatorContext:
        # 1. Build dependencies
        service = MyService()

        # 2. Store in context
        return context.with_updates(my_service=service)

    async def start_background_tasks(self) -> list[Task]:
        # 3. Start async work
        task = asyncio.create_task(self._run_loop())
        self._register_background_task(task)
        return [task]

    async def shutdown(self):
        # 4. Cleanup
        pass

    def health_check(self) -> HealthStatus:
        # 5. Report health
        pass
```

### 5.2 Tool Application Pattern (Utilities)

```python
# Pattern 1: Decorator stacking
@async_retry(max_attempts=3)
@async_rate_limit(rate_limit=10.0)
async def call_external_api(): ...

# Pattern 2: Context manager
with log_operation("process_order", logger, symbol="BTC-USD"):
    # Automatic timing and logging
    result = await place_order(...)

# Pattern 3: Factory + Usage
operations = create_trading_operations(broker, risk_manager)
order = operations.place_order(symbol, side, quantity)

# Pattern 4: Error handling
try:
    data = await fetch_data()
except NetworkError as e:
    log_error(e)
    # Recovery strategy applied by handler
```

### 5.3 Configuration Pattern

```python
# Pattern: Pydantic model with validators + baseline tracking
config = LiveTradeConfig(
    day_mmr_per_symbol={"BTC-USD": Decimal("0.05")},
    max_position_usd=Decimal("10000"),
)

# Track configuration state
baseline = ConfigBaselinePayload.from_config(config, derivatives_enabled=True)

# Later: detect drift
new_config = updated_config()
new_baseline = ConfigBaselinePayload.from_config(new_config, derivatives_enabled=True)
diff = baseline.diff(new_baseline)  # Returns changed fields
```

### 5.4 Monitoring Pattern

```python
# Pattern: Guard-based runtime protection
guard_manager = RuntimeGuardManager()
health = guard_manager.check_all_guards()

# Configuration drift detection
guardian = ConfigurationGuardian(baseline_config)
drift_detected = guardian.check_drift(current_config)

# Performance monitoring
@monitor_api_operation
async def fetch_data():
    ...
```

---

## 6. BOTTLENECKS AND INEFFICIENCIES

### 6.1 Identified Bottlenecks

**1. Context Propagation Overhead**
- **Issue**: CoordinatorContext must pass through entire initialization chain
- **Impact**: Each coordinator recreates partial state
- **Recommendation**: Implement context caching or lazy initialization

**2. Logging Scattered Across Multiple Systems**
- **Issue**: ConsoleLogger, StructuredLogger, log_operation all used
- **Impact**: Inconsistent formatting, multiple logger instances
- **Recommendation**: Unify logging abstraction with pluggable formatters

**3. Validation Redundancy**
- **Issue**: 14+ validators in `live_trade_config.py`, custom validators in guards/filters
- **Impact**: Validation logic duplication, hard to test in isolation
- **Recommendation**: Extract to validation rules library

**4. Cache Invalidation Strategy Missing**
- **Issue**: AsyncCache TTL-based only, no event-driven invalidation
- **Impact**: Stale data in high-frequency scenarios
- **Recommendation**: Add invalidation hooks and refresh strategies

**5. Error Recovery Not Standardized**
- **Issue**: CircuitBreaker implemented but not universally applied
- **Impact**: Different failure modes in different subsystems
- **Recommendation**: Make recovery strategies transparent through decorators

### 6.2 Performance Hotspots

**1. Coordinator Initialization Chain**
- **Analysis**: `O(n)` context updates during initialization
- **Mitigation**: Batch updates or use immutable structure sharing

**2. Market Data Filtering**
- **Analysis**: `MarketConditionFilters` recalculated per-operation
- **Mitigation**: Implement caching for stable market conditions

**3. Guard Checks on Every Trade**
- **Analysis**: Multiple guard validations per order
- **Mitigation**: Short-circuit evaluation, cache guard results

**4. Structured Logging Overhead**
- **Analysis**: Field formatting for every log call
- **Mitigation**: Lazy field serialization, batch log writes

### 6.3 Integration Friction

**1. Async/Sync Boundary Complexity**
- **Current**: AsyncToSyncWrapper/SyncToAsyncWrapper exist
- **Issue**: Used inconsistently, can cause deadlocks
- **Recommendation**: Establish clear async-first architecture with limited sync bridges

**2. Service Discovery**
- **Current**: Manual dependency passing through context
- **Issue**: Verbose, error-prone, no compile-time safety
- **Recommendation**: Implement typed service registry or DI container

**3. Testing Infrastructure**
- **Current**: Mocks created for each coordinator/service
- **Issue**: Test setup verbose, fixtures duplicated
- **Recommendation**: Build test harness library with auto-wiring

---

## 7. RECOMMENDATIONS FOR IMPROVEMENT

### 7.1 Priority 1: Foundational (High Impact, Medium Effort)

**1. Create Unified Observability Stack**
```python
# NEW: observability/
class ObservabilityContext:
    trace_id: str
    parent_span_id: str | None

class Span:
    async def record_metric(self, name: str, value: float): ...
    async def log_event(self, event: str): ...

@observe(span_name="api_call")
async def my_operation(span: Span):
    await span.record_metric("latency", time.time())
```

**Impact**: End-to-end tracing, metrics aggregation, performance insights
**Effort**: 3-4 days
**Files**: ~500 lines (5 new modules)

**2. Implement Metrics Store Interface**
```python
# NEW: metrics/store.py
class MetricsStore(Protocol):
    async def record(self, metric: Metric): ...
    async def query(self, query: MetricQuery): ...

# Implementations: FileMetricsStore, RedisMetricsStore, PrometheusExporter
```

**Impact**: Pluggable metrics backends, production monitoring
**Effort**: 2-3 days
**Files**: ~400 lines (4 modules)

**3. Consolidate Validation Pipeline**
```python
# NEW: validation/pipeline.py
class ValidationRule:
    def validate(self, value: Any) -> tuple[bool, str]: ...

class ValidationPipeline:
    def add_rule(self, rule: ValidationRule): ...
    def validate(self, value: Any) -> ValidationResult: ...
```

**Impact**: DRY validation, easier testing, reusable rules
**Effort**: 2 days
**Files**: ~300 lines (3 modules)

### 7.2 Priority 2: Advanced (Medium Impact, High Effort)

**1. Build Dependency Injection Container**
```python
# NEW: di/container.py
@injectable(scope=Scope.SINGLETON)
class BrokerClient:
    pass

@injectable(scope=Scope.TRANSIENT)
class OrderValidator:
    broker: BrokerClient = inject()
```

**Impact**: Reduced boilerplate, better testability, type safety
**Effort**: 4-5 days
**Files**: ~600 lines (5 modules)

**2. Implement Saga/Transaction Framework**
```python
# NEW: sagas/framework.py
class Saga:
    @saga_step(name="create_order", rollback="cancel_order")
    async def create_order(self): ...

    @saga_step(name="execute_order", rollback="cancel_execution")
    async def execute_order(self): ...
```

**Impact**: Multi-step operation atomicity, rollback support, error recovery
**Effort**: 5-6 days
**Files**: ~700 lines (5 modules)

### 7.3 Priority 3: Polish (Low Impact, Low Effort)

**1. Add Developer Experience Tools**
- CLI for local monitoring dashboard
- Debug mode with enhanced logging
- Test data generators

**2. Documentation Generation**
- Auto-generate tool reference from decorators
- Create architecture diagrams
- Tool usage examples

---

## 8. SPECIFIC IMPLEMENTATION RECOMMENDATIONS

### 8.1 For Async Tools Enhancement

**Current State**: Good foundation with retry, rate limiting, caching
**Recommendations**:

```python
# 1. Add adaptive retry with jitter variation
@async_retry(
    max_attempts=3,
    backoff_strategy=BackoffStrategy.EXPONENTIAL_WITH_JITTER,
    jitter_factor=0.1
)

# 2. Add timeout wrapper to all async operations
@async_timeout(timeout=30.0)
@async_retry()
async def operation(): ...

# 3. Add circuit breaker as decorator
@circuit_breaker(failure_threshold=5, recovery_timeout=60)
async def risky_operation(): ...

# 4. Implement bulkhead pattern for resource isolation
@bulkhead(max_concurrent=10, queue_size=100)
async def heavy_operation(): ...
```

### 8.2 For Orchestration Enhancement

**Current State**: Solid coordinator pattern with context propagation
**Recommendations**:

```python
# 1. Add coordinator composition
coordinator_group = CoordinatorGroup([
    RuntimeCoordinator(context),
    StrategyCoordinator(context),
    ExecutionCoordinator(context),
])
await coordinator_group.initialize_all()

# 2. Add coordinator hooks/events
class EventAwareCoordinator(BaseCoordinator):
    def on_context_updated(self, old: Context, new: Context):
        # React to context changes
        pass

# 3. Add coordinator scheduling
@scheduled(interval=60)
async def periodic_health_check(coord: RuntimeCoordinator):
    health = coord.health_check()
    emit_metric("coordinator_health", health.score)
```

### 8.3 For Strategy Tools Enhancement

**Current State**: Filter and guard dataclasses exist but underutilized
**Recommendations**:

```python
# 1. Make filters composable
filters = (
    MarketConditionFilters.depth_filter(min_l1=50000)
    .volume_filter(min_1m=100000)
    .spread_filter(max_bps=10)
)

# 2. Add strategy recommendation engine
class StrategyRecommender:
    def recommend_filters(self, market_conditions: Dict) -> MarketConditionFilters:
        # Auto-select filter settings based on market
        pass

# 3. Add backtest integration
strategy = StrategyWithTools(filters, guards)
backtest_result = await strategy.backtest(
    historical_data=data,
    filter_recommendations=True
)
```

---

## 9. TOOLING MATURITY ASSESSMENT

### Current Maturity Levels:

| Tool Category | Maturity | Status | Gap |
|---|---|---|---|
| Async Tools | Production | Solid retry, rate limit, cache | Missing: bulkhead, timeout decorators, adaptive backoff |
| Logging | Development | Good structured logging | Missing: Log aggregation, structured exporters |
| Error Handling | Production | Rich error hierarchy, circuit breaker | Missing: Error recovery policies, automatic retry selection |
| Orchestration | Production | Coordinator pattern working | Missing: Hot-reload, dynamic composition |
| Monitoring | Development | Health checks, config guardian | Missing: Metrics aggregation, alerts, dashboards |
| Strategy Tools | Prototype | Filters and guards exist | Missing: Composition, recommendations, testing framework |
| Performance | Development | Profiler exists | Missing: Real-time dashboard, alerting on regressions |
| Configuration | Production | Pydantic-based validation | Missing: Hot-reload, versioning, schema validation |

---

## 10. CONCLUSION

The GPT-Trader codebase has built a solid foundation with well-organized utilities, a promising coordinator pattern for orchestration, and comprehensive error handling. The main opportunities lie in:

1. **Consolidating** scattered observability and logging
2. **Extending** the async tool decorator framework for resilience patterns
3. **Filling gaps** in distributed tracing, metrics aggregation, and DI
4. **Enhancing** strategy tools with composability and recommendations
5. **Improving** performance with targeted caching and bulkhead patterns

The recommended roadmap prioritizes foundational improvements (observability, validation pipeline, metrics store) that will provide immediate value, followed by advanced capabilities (DI, Sagas) for longer-term scalability.

---

## APPENDIX: File Structure Reference

```
src/bot_v2/
├── utilities/
│   ├── __init__.py (exports 50+ utilities)
│   ├── async_tools/
│   │   ├── __init__.py
│   │   ├── retry.py (AsyncRetry, async_retry)
│   │   ├── rate_limit.py (AsyncRateLimiter)
│   │   ├── cache.py (AsyncCache)
│   │   ├── context.py (AsyncContextManager)
│   │   ├── wrappers.py (Sync/Async bridges)
│   │   ├── helpers.py (gather_with_concurrency, etc.)
│   │   └── __pycache__/
│   ├── performance/
│   │   ├── monitor.py (PerformanceMonitor)
│   │   ├── metrics.py (PerformanceCollector)
│   │   ├── reporter.py (PerformanceReporter)
│   │   ├── profiler.py (PerformanceProfiler)
│   │   ├── timing.py (PerformanceTimer)
│   │   ├── resource.py (ResourceMonitor)
│   │   ├── health.py (Performance health)
│   │   └── decorators.py
│   ├── logging_patterns.py
│   ├── console_logging.py
│   ├── trading_operations.py
│   ├── telemetry.py
│   ├── common_patterns.py
│   ├── config.py
│   └── [15 other utility files]
├── errors/
│   ├── __init__.py (Error hierarchy)
│   ├── handler.py (RecoveryStrategy, CircuitBreaker)
│   └── error_patterns.py
├── monitoring/
│   ├── __init__.py
│   ├── configuration_guardian.py
│   ├── runtime_guards.py
│   ├── health_checks.py
│   ├── guards/ (base, manager, builtins)
│   ├── health/ (registry, checks, endpoint)
│   ├── system/ (engine, collectors, alerting)
│   └── domain/ (perps liquidation, margin)
├── orchestration/
│   ├── coordinators/
│   │   ├── base.py (CoordinatorContext, BaseCoordinator)
│   │   ├── registry.py (CoordinatorRegistry)
│   │   ├── runtime.py (RuntimeCoordinator)
│   │   ├── strategy.py (StrategyCoordinator)
│   │   ├── execution.py (ExecutionCoordinator)
│   │   └── telemetry.py (TelemetryCoordinator)
│   └── [17 other orchestration files]
├── features/
│   └── strategy_tools/
│       ├── __init__.py
│       ├── filters.py (MarketConditionFilters)
│       ├── guards.py (RiskGuards)
│       └── enhancements.py
├── config/
│   ├── live_trade_config.py (14+ field validators)
│   ├── env_utils.py
│   ├── config_utilities.py
│   └── [5 other config files]
└── [other modules: security, validation, cli, etc.]

Total: 259 Python files, ~50,000 lines of production code
```
