# Logger Refactoring Plan

**File**: `src/bot_v2/monitoring/system/logger.py`
**Current Size**: 638 lines
**Target**: Split into 4-5 focused modules (~100-150 lines each)

---

## Current Structure Analysis

### Responsibilities Identified

1. **Core Logging Infrastructure** (~80 lines)
   - `__init__()`, correlation/buffer/emitter setup
   - `_create_log_entry()`, `_emit_log()`
   - Correlation ID management

2. **General Purpose Logging** (~160 lines)
   - `log_event()` - general system events
   - `log_trade()` - trading activity
   - `log_ml_prediction()` - ML predictions
   - `log_performance()` - performance metrics
   - `log_error()` - error handling

3. **Trading Domain Loggers** (~130 lines)
   - `log_order_submission()`
   - `log_order_status_change()`
   - `log_position_change()`
   - `log_balance_update()`
   - `log_order_round_trip()`

4. **Market & Infrastructure Loggers** (~75 lines)
   - `log_market_heartbeat()`
   - `log_ws_latency()`
   - `log_rest_response()`
   - `log_strategy_duration()`

5. **Risk & PnL Loggers** (~65 lines)
   - `log_pnl()`
   - `log_funding()`
   - `log_risk_breach()`
   - `log_auth_event()`

6. **Global Instance & Convenience Functions** (~70 lines)
   - Global `_logger` instance
   - `get_logger()`
   - Convenience wrapper functions

---

## Proposed Module Structure

### 1. `trading_event_logger.py` (Trading Domain)
**Size**: ~140 lines
**Responsibility**: Specialized logging for trading operations

```python
class TradingEventLogger:
    """Specialized logger for trading domain events."""

    def __init__(self, core_logger: "ProductionLogger"):
        self._logger = core_logger

    def log_order_submission(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log order submission."""
        entry = self._logger._create_log_entry(
            level=LogLevel.INFO,
            event_type="order_submission",
            message=f"submit {side} {quantity} {symbol} @{price if price is not None else 'mkt'}",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            **kwargs,
        )
        self._logger._emit_log(entry)

    def log_order_status_change(
        self,
        order_id: str,
        client_order_id: str | None,
        from_status: str | None,
        to_status: str,
        exchange_error_code: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log order status transitions."""
        # Implementation...

    def log_position_change(
        self,
        symbol: str,
        side: str,
        size: float,
        avg_entry_price: float | None = None,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log position changes."""
        # Implementation...

    def log_balance_update(
        self,
        currency: str,
        available: float,
        total: float,
        change: float | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log balance updates."""
        # Implementation...

    def log_order_round_trip(
        self,
        order_id: str,
        client_order_id: str | None,
        round_trip_ms: float,
        submitted_ts: str | None = None,
        filled_ts: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log order round-trip timing."""
        # Implementation...
```

**Benefits**:
- Consolidates all order/position/balance logging
- Clear trading domain boundary
- Easy to extend with new trading events

---

### 2. `market_event_logger.py` (Market & Infrastructure)
**Size**: ~90 lines
**Responsibility**: Market data and infrastructure event logging

```python
class MarketEventLogger:
    """Specialized logger for market and infrastructure events."""

    def __init__(self, core_logger: "ProductionLogger"):
        self._logger = core_logger

    def log_market_heartbeat(
        self,
        source: str,
        last_update_ts: str,
        latency_ms: float | None = None,
        staleness_ms: float | None = None,
        threshold_ms: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Log market data heartbeat."""
        # Implementation...

    def log_ws_latency(self, stream: str, latency_ms: float, **kwargs: Any) -> None:
        """Log WebSocket latency."""
        # Implementation...

    def log_rest_response(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        """Log REST API response timing."""
        # Implementation...

    def log_strategy_duration(self, strategy: str, duration_ms: float, **kwargs: Any) -> None:
        """Log strategy execution duration."""
        # Implementation...
```

**Benefits**:
- Focused on market data and network operations
- Infrastructure monitoring consolidated
- Easy to add new market data event types

---

### 3. `risk_event_logger.py` (Risk & PnL)
**Size**: ~80 lines
**Responsibility**: Risk management and PnL event logging

```python
class RiskEventLogger:
    """Specialized logger for risk and PnL events."""

    def __init__(self, core_logger: "ProductionLogger"):
        self._logger = core_logger

    def log_pnl(
        self,
        symbol: str,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        fees: float | None = None,
        funding: float | None = None,
        position_size: float | None = None,
        transition: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log PnL updates."""
        # Implementation...

    def log_funding(
        self,
        symbol: str,
        funding_rate: float,
        payment: float,
        period_start: str | None = None,
        period_end: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log funding payments."""
        # Implementation...

    def log_risk_breach(
        self,
        limit_type: str,
        limit_value: float,
        current_value: float,
        **kwargs: Any,
    ) -> None:
        """Log risk limit breaches."""
        # Implementation...

    def log_auth_event(
        self,
        action: str,
        provider: str,
        success: bool,
        error_code: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log authentication events."""
        # Implementation...
```

**Benefits**:
- Risk-related events in one place
- Easy to add compliance/audit requirements
- Clear separation from trading execution

---

### 4. `logger.py` (REFACTORED - Core Orchestrator)
**Size**: ~250 lines
**Responsibility**: Core logging infrastructure + general-purpose logging

```python
class ProductionLogger:
    """
    High-performance production logger with structured JSON output.

    Delegates specialized domain logging to focused sub-loggers while
    maintaining core infrastructure and general-purpose logging methods.
    """

    def __init__(self, service_name: str = "bot_v2", enable_console: bool = True) -> None:
        """Initialize production logger."""
        self.service_name = service_name

        # Infrastructure components
        self._correlation = CorrelationContext()
        self._buffer = LogBuffer(max_size=1000)
        self._performance = PerformanceTracker()

        # Configure emitter
        min_level = os.getenv("PERPS_MIN_LOG_LEVEL", "info").lower()
        if os.getenv("PERPS_DEBUG") in ("1", "true", "yes", "on"):
            min_level = "debug"

        py_logger = logging.getLogger(f"{service_name}.json")
        if not py_logger.handlers:
            py_logger = logging.getLogger("bot_v2.json")

        env_console = os.getenv("PERPS_JSON_CONSOLE")
        if env_console is not None:
            enable_console = env_console.strip().lower() in ("1", "true", "yes", "on")

        self._emitter = LogEmitter(
            service_name=service_name,
            enable_console=enable_console,
            min_level=min_level,
            py_logger=py_logger,
        )

        # Initialize specialized loggers
        self._trading = TradingEventLogger(self)
        self._market = MarketEventLogger(self)
        self._risk = RiskEventLogger(self)

    # Core infrastructure methods
    def set_correlation_id(self, correlation_id: str | None = None) -> None:
        """Set correlation ID for current thread."""
        self._correlation.set_correlation_id(correlation_id)

    def get_correlation_id(self) -> str:
        """Get current thread's correlation ID."""
        return self._correlation.get_correlation_id()

    def _create_log_entry(
        self, level: LogLevel, event_type: str, message: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Create structured log entry with minimal overhead."""
        # Implementation (unchanged)

    def _emit_log(self, entry: dict[str, Any]) -> None:
        """Emit log entry (delegates to buffer and emitter)."""
        # Implementation (unchanged)

    # General-purpose logging methods (kept in core)
    def log_event(
        self,
        level: LogLevel,
        event_type: str,
        message: str,
        component: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log a general system event."""
        # Implementation (unchanged)

    def log_trade(
        self,
        action: str,
        symbol: str,
        quantity: float,
        price: float,
        strategy: str,
        success: bool = True,
        execution_time_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log trading activity with specialized fields."""
        # Implementation (unchanged)

    def log_ml_prediction(
        self,
        model_name: str,
        prediction: Any,
        confidence: float | None = None,
        input_features: dict[str, Any] | None = None,
        inference_time_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log ML model predictions and inference."""
        # Implementation (unchanged)

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs: Any,
    ) -> None:
        """Log performance metrics for operations."""
        # Implementation (unchanged)

    def log_error(self, error: Exception, context: str | None = None, **kwargs: Any) -> None:
        """Log errors with full context."""
        # Implementation (unchanged)

    # Delegation methods to specialized loggers
    def log_order_submission(self, *args, **kwargs) -> None:
        """Delegate to trading event logger."""
        self._trading.log_order_submission(*args, **kwargs)

    def log_order_status_change(self, *args, **kwargs) -> None:
        """Delegate to trading event logger."""
        self._trading.log_order_status_change(*args, **kwargs)

    def log_position_change(self, *args, **kwargs) -> None:
        """Delegate to trading event logger."""
        self._trading.log_position_change(*args, **kwargs)

    def log_balance_update(self, *args, **kwargs) -> None:
        """Delegate to trading event logger."""
        self._trading.log_balance_update(*args, **kwargs)

    def log_order_round_trip(self, *args, **kwargs) -> None:
        """Delegate to trading event logger."""
        self._trading.log_order_round_trip(*args, **kwargs)

    def log_market_heartbeat(self, *args, **kwargs) -> None:
        """Delegate to market event logger."""
        self._market.log_market_heartbeat(*args, **kwargs)

    def log_ws_latency(self, *args, **kwargs) -> None:
        """Delegate to market event logger."""
        self._market.log_ws_latency(*args, **kwargs)

    def log_rest_response(self, *args, **kwargs) -> None:
        """Delegate to market event logger."""
        self._market.log_rest_response(*args, **kwargs)

    def log_strategy_duration(self, *args, **kwargs) -> None:
        """Delegate to market event logger."""
        self._market.log_strategy_duration(*args, **kwargs)

    def log_pnl(self, *args, **kwargs) -> None:
        """Delegate to risk event logger."""
        self._risk.log_pnl(*args, **kwargs)

    def log_funding(self, *args, **kwargs) -> None:
        """Delegate to risk event logger."""
        self._risk.log_funding(*args, **kwargs)

    def log_risk_breach(self, *args, **kwargs) -> None:
        """Delegate to risk event logger."""
        self._risk.log_risk_breach(*args, **kwargs)

    def log_auth_event(self, *args, **kwargs) -> None:
        """Delegate to risk event logger."""
        self._risk.log_auth_event(*args, **kwargs)

    # Utility methods
    def get_recent_logs(self, count: int = 100) -> list[dict[str, Any]]:
        """Get recent log entries."""
        return self._buffer.get_recent(count)

    def get_performance_stats(self) -> dict[str, float | int]:
        """Get logger performance statistics."""
        return self._performance.get_stats()
```

**Benefits**:
- Core logger remains focused on infrastructure
- Clear delegation to specialized loggers
- Backward compatible API (all methods still available)
- Easy to add new domain loggers

---

## Refactoring Strategy

### Phase 1: Extract Specialized Loggers (Bottom-Up)
1. **Create `trading_event_logger.py`** (30 min)
   - Extract order/position/balance methods
   - Add tests

2. **Create `market_event_logger.py`** (20 min)
   - Extract market/ws/rest methods
   - Add tests

3. **Create `risk_event_logger.py`** (20 min)
   - Extract pnl/funding/risk methods
   - Add tests

### Phase 2: Refactor Main Logger (Top-Down)
4. **Refactor `logger.py`** (30 min)
   - Initialize specialized loggers
   - Convert methods to delegation
   - Update existing tests

---

## Testing Strategy

### Unit Tests (New)
- `tests/unit/bot_v2/monitoring/system/test_trading_event_logger.py`
- `tests/unit/bot_v2/monitoring/system/test_market_event_logger.py`
- `tests/unit/bot_v2/monitoring/system/test_risk_event_logger.py`

### Integration Tests (Update)
- Update existing logger tests to verify delegation
- Ensure backward compatibility

---

## Benefits Summary

### Code Quality
- ✅ **Single Responsibility**: Each logger handles one domain
- ✅ **Testability**: Specialized loggers easy to test in isolation
- ✅ **Maintainability**: Clear boundaries for adding new events
- ✅ **Extensibility**: Easy to add new domain loggers

### Team Benefits
- ✅ **Domain Organization**: Trading/Market/Risk events clearly separated
- ✅ **Easier Extension**: Add new trading events to TradingEventLogger
- ✅ **Better Navigation**: Find relevant logging code faster

### Performance
- ✅ **No Regression**: Delegation adds negligible overhead
- ✅ **Same Performance**: Core logging infrastructure unchanged

---

## Migration Path

### Backward Compatibility
- Keep `ProductionLogger` public API unchanged
- All existing methods still available (via delegation)
- All existing tests should pass without modification

### Rollout
1. Create specialized logger modules
2. Update `ProductionLogger` to use specialized loggers internally
3. Run full test suite
4. Deploy with monitoring
5. Optional: Deprecate global convenience functions in favor of domain-specific imports

---

## Success Criteria

✅ **Code Metrics**:
- `logger.py`: 638 → ~250 lines (61% reduction)
- 3 new focused modules (~90-140 lines each)
- Test coverage ≥85%

✅ **Functionality**:
- All existing tests pass
- No performance regression
- Backward compatible API

✅ **Quality**:
- Clear domain boundaries
- No circular dependencies
- Type hints complete

---

## Estimated Effort

- **Phase 1 (Extract Specialized Loggers)**: 1.5 hours
- **Phase 2 (Refactor Main Logger)**: 0.5 hours
- **Testing & Validation**: 0.5 hours
- **Total**: ~2.5 hours

---

## Next Steps

1. Review this plan
2. Create feature branch: `refactor/logger-decomposition`
3. Execute Phase 1 (bottom-up extraction)
4. Execute Phase 2 (top-down refactoring)
5. Code review and merge
