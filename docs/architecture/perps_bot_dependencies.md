# PerpsBot Architecture & Dependencies

**Status**: Living document – updated for builder-default construction
**Purpose**: Map ALL dependencies before refactoring to prevent data loss
**Last Updated**: 2025-04-27 (Builder default + service rebinding helper)

## Overview

PerpsBot is the main orchestration class for the trading system. This document maps its dependencies, side effects, and data flows to ensure safe refactoring.

## Initialization Sequence

```
PerpsBot.__init__(config, registry=None)
  ├─ builder = PerpsBotBuilder(config)
  │   └─ if registry provided → builder.with_registry(registry)
  └─ builder.build_into(self)
      ├─ _build_configuration_state(bot)
      ├─ _build_runtime_state(bot)
      ├─ _build_storage(bot)
      ├─ _build_core_services(bot)
      ├─ bot.runtime_coordinator.bootstrap()
      ├─ _build_market_data_service(bot)
      ├─ _build_accounting_services(bot)
      ├─ _build_market_services(bot)
      ├─ _build_streaming_service(bot)
      ├─ _start_streaming_if_configured(bot)
      └─ service_rebinding.rebind_bot_services(bot)

# Streaming toggles at runtime re-use: PerpsBot._start_streaming_if_configured()
```

**Rebinding helper:** `rebind_bot_services(bot)` runs at the end of `build_into`, walking every
coordinator-style attribute and reassigning cached `_bot` references to the live runtime instance.
Adding a new service that stores a `_bot` attribute automatically opts in to this rebind step.

## MarketDataService Extraction - Dependency Checklist

*Builder hook:* `PerpsBotBuilder._build_market_data_service`

### Inputs (Constructor Dependencies)
- `symbols: list[str]` - from config
- `broker: IBrokerage` - from registry
- `risk_manager: LiveRiskManager` - from registry
- `long_ma: int` - from config (for window trimming)
- ~~`event_store: EventStore`~~ - NOT NEEDED (update_marks doesn't write to event_store)

### Outputs (Side Effects to Preserve)
```python
async def update_marks(self):
    for symbol in symbols:
        # 1. Fetch quote
        quote = await asyncio.to_thread(self.broker.get_quote, symbol)

        # 2. Extract price
        last_price = getattr(quote, "last", getattr(quote, "last_price", None))
        mark = Decimal(str(last_price))

        # 3. Extract timestamp
        ts = getattr(quote, "ts", datetime.now(UTC))

        # 4. Update mark window (CRITICAL: thread-safe)
        self._update_mark_window(symbol, mark)

        # 5. SIDE EFFECT: Update risk manager timestamp
        self.risk_manager.last_mark_update[symbol] = (
            ts if isinstance(ts, datetime) else datetime.utcnow()
        )
        # ⚠️ Exception handling here logs debug but continues
```

**Side effects that MUST be preserved:**
1. ✅ Updates `mark_windows[symbol]` (thread-safe via `_mark_lock`)
2. ✅ Updates `risk_manager.last_mark_update[symbol]` with timestamp
3. ✅ Trims mark_windows to max(long_ma, short_ma) + 5
4. ✅ Logs errors but continues processing other symbols
5. ✅ No telemetry hooks (verified: only streaming uses MarketActivityMonitor)

## StreamingService Extraction (Phase 2) - Dependency Checklist

*Builder hook:* `PerpsBotBuilder._build_streaming_service`

### Inputs (Constructor Dependencies)
- `symbols: list[str]` - from PerpsBot
- `broker: IBrokerage` - from registry
- `market_data_service: MarketDataService` - for mark window updates and lock
- `risk_manager: LiveRiskManager` - from registry
- `event_store: EventStore` - for metrics
- `market_monitor: MarketActivityMonitor` - for heartbeats
- `bot_id: str` - for event logging

### Outputs (Side Effects to Preserve)
```python
def _stream_loop(self, symbols, level):
    # 1. Try orderbook stream, fallback to trades
    stream = broker.stream_orderbook(symbols, level=level)
    # OR: stream = broker.stream_trades(symbols)

    for msg in stream:
        # 2. Extract mark price from bid/ask or last
        mark = (bid + ask) / 2  # OR: mark = last

        # 3. Update mark window (CRITICAL: thread-safe via MarketDataService)
        market_data_service._update_mark_window(symbol, mark)

        # 4. SIDE EFFECTS to preserve:
        market_monitor.record_update(symbol)
        risk_manager.last_mark_update[symbol] = datetime.utcnow()
        event_store.append_metric(bot_id, {"event_type": "ws_mark_update", ...})
```

**Side effects that MUST be preserved:**
1. ✅ Updates `mark_windows[symbol]` via `market_data_service._update_mark_window()`
2. ✅ Updates `risk_manager.last_mark_update[symbol]` with timestamp
3. ✅ Records `market_monitor.record_update(symbol)` for each update
4. ✅ Writes event_store metrics: `ws_mark_update`, `ws_stream_error`, `ws_stream_exit`
5. ✅ Falls back from orderbook → trades on error
6. ✅ Respects stop event for graceful shutdown
7. ✅ Thread-safe via shared `_mark_lock` from MarketDataService

### Lock Sharing (CRITICAL)

**Current state after builder default:**
```python
# PerpsBotBuilder creates lock
bot._mark_lock = threading.RLock()

# MarketDataService shares the lock
bot._market_data_service = MarketDataService(
    ...,
    mark_lock=bot._mark_lock,  # ← SAME instance
    mark_windows=bot.mark_windows,
)

# StreamingService uses lock via MarketDataService
bot._streaming_service = StreamingService(
    market_data_service=bot._market_data_service,  # ← Lock accessed here
    ...
)

# Streaming calls:
bot._market_data_service._update_mark_window(symbol, mark)
# Which internally uses:
with self._mark_lock:  # ← SAME lock as PerpsBot._mark_lock
    ...
```

**Verification:**
- ✅ Unit tests verify lock usage via instrumentation
- ✅ Characterization tests verify lock sharing: `bot._mark_lock is bot._streaming_service.market_data_service._mark_lock`
- ✅ Concurrent update_marks + streaming don't race

### Feature Flag Rollback Path

**Flag: `USE_NEW_STREAMING_SERVICE` *(retired Oct 2025)***

StreamingService is now always active:
- All streaming handled via `_streaming_service.start(level)`
- Shutdown via `_streaming_service.stop()`
- Config changes via service restart
- Legacy methods (`_start_streaming_background_legacy()`, `_run_stream_loop()`) removed
- No rollback path available - StreamingService is the only implementation

### Data Flow Map

```
┌─────────────────────────────────────────────────┐
│ External Data Sources                           │
├─────────────────────────────────────────────────┤
│ broker.get_quote()                              │
│ broker.stream_orderbook()                       │
│ broker.stream_trades()                          │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│ PerpsBot.update_marks() [async polling]         │
│ PerpsBot._run_stream_loop() [background thread] │
└────────────┬────────────────────────────────────┘
             │
             ▼ (via _update_mark_window)
┌─────────────────────────────────────────────────┐
│ mark_windows: dict[str, list[Decimal]]          │
│ [Protected by _mark_lock: threading.RLock]      │
└────────────┬────────────────────────────────────┘
             │
             ├──────────────────────────┐
             │                          │
             ▼                          ▼
    ┌────────────────┐      ┌──────────────────────┐
    │ risk_manager   │      │ strategy_orchestrator │
    │ .last_mark     │      │ .process_symbol()     │
    │ _update[sym]   │      │   uses mark_windows  │
    └────────────────┘      └──────────────────────┘
```

## Property Access Patterns

### Critical Properties (Must Not Break)
```python
# These may use descriptors - builder shim must preserve
@property
def broker(self) -> IBrokerage:
    # Raises RuntimeError if None

@property
def risk_manager(self) -> LiveRiskManager:
    # Raises RuntimeError if None

@property
def exec_engine(self) -> LiveExecutionEngine | AdvancedExecutionEngine:
    # Raises RuntimeError if None
```

**Implication for Builder:**
- Using `self.__dict__ = bot.__dict__` bypasses `@property` descriptors
- Safer: Use `@classmethod from_builder(builder)` pattern
- Or: Override `__getattribute__` to proxy to built instance

## Shared Mutable State (Coupling Points)

| State | Writers | Readers | Lock |
|-------|---------|---------|------|
| `mark_windows` | update_marks, streaming | strategy, risk | `_mark_lock` |
| `last_decisions` | strategy | monitoring, logging | None |
| `_product_map` | get_product (cache) | execution | None |
| `order_stats` | execution | monitoring | None |
| `risk_manager.last_mark_update` | update_marks, streaming | risk checks | None |

**Refactoring risks:**
- Breaking `_mark_lock` sharing → race conditions
- Copying state instead of sharing → stale data
- Removing side effects → silent failures

## Background Tasks (Async Lifecycle)

### Spawned by `run()` (only if not dry_run and not single_cycle)
1. `execution_coordinator.run_runtime_guards()` - continuous task
2. `execution_coordinator.run_order_reconciliation()` - continuous task
3. `system_monitor.run_position_reconciliation()` - continuous task
4. `account_telemetry.run()` - continuous task (if supports_snapshots)

### Spawned by `_start_streaming_if_configured()`
- `_ws_thread` - background thread running `_run_stream_loop()`
- Uses `_ws_stop: threading.Event` for shutdown signaling

**Refactoring requirement:**
- Ensure all tasks are canceled in `shutdown()`
- Preserve task ordering and dependencies
- Test: `shutdown()` must not hang or leak tasks

## Testing Strategy

### Characterization Test Coverage

**Must verify after each extraction:**

1. **Initialization invariants:**
   - [ ] All services exist after `__init__`
   - [ ] mark_windows initialized for all symbols
   - [ ] Locks are created (RLock type)
   - [ ] Registry has broker, risk_manager
   - [ ] Event/orders stores exist

2. **update_marks behavior:**
   - [ ] mark_windows updated for each symbol
   - [ ] risk_manager.last_mark_update updated
   - [ ] Error on one symbol doesn't stop others
   - [ ] Thread-safe when called concurrently

3. **Streaming behavior:**
   - [ ] _ws_thread starts when configured
   - [ ] mark_windows updated from stream
   - [ ] Uses same lock as update_marks
   - [ ] Stops cleanly on shutdown

4. **Property access:**
   - [ ] broker raises RuntimeError if None
   - [ ] risk_manager raises RuntimeError if None
   - [ ] exec_engine raises RuntimeError if None

5. **Delegation:**
   - [ ] process_symbol → strategy_orchestrator
   - [ ] execute_decision → execution_coordinator
   - [ ] write_health_status → system_monitor

## Rollback Feature Flags

### Environment Variables for Gradual Rollout

```bash
# Phase 1: MarketDataService ✅ COMPLETE (2025-10-01)
# Flag retired Oct 2025 — MarketDataService always active

# Phase 2: StreamingService
# USE_NEW_STREAMING_SERVICE — Flag retired Oct 2025, StreamingService always active

# Phase 3: BotBuilder
USE_BUILDER_PATTERN=true          # Default after extraction
```

**Rollback procedure:**
1. Set flag to `false` in production config
2. Restart service
3. Monitor for 24h
4. If stable, remove old code in next release

## Open Questions - ANSWERED ✅

### 1. Does update_marks write to event_store?
**Answer**: ❌ NO - `update_marks` does NOT write to event_store.
- event_store is only used in `_run_stream_loop` (streaming path)
- Lines 474, 482, 490 in perps_bot.py
- update_marks only updates: mark_windows, risk_manager.last_mark_update
- **Implication**: MarketDataService does NOT need event_store dependency

### 2. Are there telemetry hooks beyond heartbeat_logger?
**Answer**: ❌ NO - Only heartbeat_logger exists for market data.
- Defined in `_init_market_services()` line 119
- Calls `_get_plog().log_market_heartbeat(**payload)`
- Only used by MarketActivityMonitor, not update_marks
- **Implication**: MarketDataService only needs MarketActivityMonitor

### 3. Do any external systems read mark_windows directly?
**Answer**: ❌ NO - mark_windows only accessed via bot instance.
- Grep found 3 files accessing mark_windows:
  - perps_bot.py (owner)
  - strategies/perps_baseline.py
  - strategies/perps_baseline_enhanced.py
- Strategies access via `bot.mark_windows[symbol]`, not direct reference
- **Implication**: Safe to move mark_windows to MarketDataService, expose via property

### 4. Is _product_map cache thread-safe?
**Answer**: ⚠️ NO - _product_map has NO lock protection.
- Only read in `get_product()` line 323
- NEVER written to (only returns new Product on cache miss)
- get_product creates Product on-the-fly, doesn't cache result
- **Implication**: Actually NOT a cache! Just a lookup dict, never populated
- **BUG FOUND**: _product_map is initialized but never used as intended
- **RESOLVED** ✅: Removed in Phase 1 (2025-10-01) - dead code cleanup

### 5. What happens if streaming updates mark while update_marks is trimming?
**Answer**: ✅ SAFE - Both use same _mark_lock (RLock).
- `_update_mark_window` acquires lock at line 496
- Both `update_marks` and `_run_stream_loop` call `_update_mark_window`
- RLock allows reentrant access from same thread
- Trimming logic is atomic within lock (line 501-502)
- **Implication**: Lock sharing MUST be preserved in refactor

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2025-10-01 | Initial dependency mapping | Phase 0 |
| 2025-10-01 | Answered all open questions | Phase 0 |
| 2025-10-01 | Fixed characterization tests (18 passed, 3 skipped) | Phase 0 |
