# TUI Performance Audit Report

**Date:** 2026-01-05
**Scope:** Reactive update paths, throttling, redundancy detection

---

## Update Flow Map

```
StatusReporter.notify_observers()
    │
    ▼
TraderApp._on_status_update(status: BotStatus)           [app_status.py:43]
    │
    ▼ (thread-safe dispatch via call_from_thread if needed)
UICoordinator.apply_observer_update(status)              [ui_coordinator.py:60]
    │
    ├─► [If throttling enabled] → UpdateThrottler.queue_full_status()
    │       └─► flush after min_interval → _apply_status_update()
    │
    └─► [No throttling] → _apply_status_update()         [ui_coordinator.py:81]
            │
            ├─► TuiState.update_from_bot_status()        [state.py:147]
            │       └─► Updates reactive properties (market_data, position_data, etc.)
            │
            ├─► update_main_screen()                     [ui_coordinator.py:252]
            │       │
            │       ├─► MainScreen.update_ui(state)      [main_screen.py:186]
            │       │       └─► self.state = state
            │       │               └─► watch_state() → StateRegistry.broadcast() ◄── BROADCAST #1
            │       │
            │       └─► StateRegistry.broadcast(state)   [ui_coordinator.py:272] ◄── BROADCAST #2 (REDUNDANT!)
            │
            └─► FrameMetrics recorded to PerformanceService
```

---

## Findings

### 🔴 Critical: Double Broadcast per Update Cycle

**Location:** `ui_coordinator.py:252-273`

**Issue:** Every status update triggers `StateRegistry.broadcast()` **twice**:
1. Via `MainScreen.update_ui()` → `watch_state()` → `broadcast()`
2. Via direct call in `update_main_screen()` line 272

**Impact:** All StateObserver widgets (`AccountWidget`, `SystemMonitorWidget`, `PositionCardWidget`, `MarketPulseWidget`, etc.) have their `on_state_updated()` called twice per cycle.

**Code:**
```python
# ui_coordinator.py:252-273
def update_main_screen(self) -> None:
    main_screen = self.app.query_one(MainScreen)
    main_screen.update_ui(self.app.tui_state)  # → watch_state() → broadcast() #1

    # ...

    # This is REDUNDANT - broadcast already happened via watch_state()
    if hasattr(self.app, "state_registry"):
        self.app.state_registry.broadcast(self.app.tui_state)  # broadcast() #2
```

**Root Cause:** The comment on line 268 explains the intent: "even when the active screen doesn't change (TuiState is mutated in-place)". This was added because `self.state = state` may not trigger `watch_state()` if the object reference is the same. However, this creates double updates on every cycle.

**Fix:** Remove the second broadcast. If the mutation-in-place issue needs solving, replace `TuiState` mutation with immutable updates or use a version counter.

---

### 🟡 Medium: Throttler Not Instantiated by Default

**Location:** `ui_coordinator.py:41`, app initialization

**Issue:** `UICoordinator` accepts an optional `throttler` parameter, but no code actually creates and passes an `UpdateThrottler` instance. The throttling infrastructure exists but is **unused**.

**Impact:** High-frequency market data updates are not batched, causing unnecessary UI refreshes.

**Recommendation:** Instantiate `UpdateThrottler(min_interval=0.1)` in `TraderApp._initialize_with_bot()` and pass it to `UICoordinator`.

---

### 🟡 Medium: watch_* Methods Lack No-Op Guards

**Location:** Various widgets in `widgets/status.py`, `widgets/dashboard_system.py`

**Issue:** Most `watch_*` methods unconditionally update their widgets even when the value hasn't changed.

**Examples:**
```python
# status.py:149 - No guard against same equity value
def watch_equity(self, equity: str) -> None:
    label.update(f"${equity}")
    self._flash_value(label)  # Flashes even if value unchanged!

# dashboard_system.py:226 - No guard
def watch_cpu_usage(self, val: float) -> None:
    self.query_one("#pb-cpu", ProgressBarWidget).percentage = val
```

**Impact:** Unnecessary DOM updates and potential visual flicker.

**Fix:** Add early returns when value equals previous:
```python
def watch_equity(self, equity: str) -> None:
    if hasattr(self, '_prev_equity') and self._prev_equity == equity:
        return
    self._prev_equity = equity
    # ... rest of method
```

---

### 🟢 Good: Performance Infrastructure Already Exists

**Location:** `services/performance_service.py`

The TUI already has comprehensive performance monitoring:
- `TuiPerformanceService` with `time_operation()` context manager
- `FrameMetrics` tracking (state update + render durations)
- `record_slow_operation()` for detecting > 50ms operations
- `PerformanceSnapshot` for dashboard display
- Slow broadcast detection (> 20ms threshold)

**No additional probes needed** - the infrastructure captures the key metrics.

---

### 🟢 Good: Heartbeat Loop is Optimized

**Location:** `ui_coordinator.py:303-360`

The heartbeat loop:
- Only pulses when bot is running AND status widget is visible
- Only broadcasts on connection health **transitions** (not every loop)
- Uses 2-second interval (appropriate for visual heartbeat)

---

## Quick Wins (Recommended Fixes)

### Fix 1: Remove Double Broadcast (High Impact, Low Risk) ✅ APPLIED

**Files Modified:**
- `src/gpt_trader/tui/screens/main_screen.py` - `watch_state()` now a no-op
- `src/gpt_trader/tui/managers/ui_coordinator.py` - Updated comment

**Change:** Made `MainScreen.watch_state()` a no-op pass statement. The single broadcast
now happens only in `UICoordinator.update_main_screen()`, eliminating double broadcasts.

**Result:** All StateObserver widgets now receive exactly one `on_state_updated()` call
per update cycle instead of two.

---

### Fix 2: Enable Update Throttler (Medium Impact, Low Risk)

**File:** `src/gpt_trader/tui/app_lifecycle.py` (in `_initialize_with_bot`)

```python
# After creating WorkerService, before creating UICoordinator:
from gpt_trader.tui.services import UpdateThrottler

throttler = UpdateThrottler(min_interval=0.1)  # 100ms batching

self.lifecycle_manager = BotLifecycleManager(self, worker_service=self.worker_service)
self.ui_coordinator = UICoordinator(self, throttler=throttler)  # Pass throttler
```

**Risk:** Low. The throttler code is already tested and handles edge cases.

---

## Metrics to Monitor Post-Fix

After implementing fixes, verify improvement using existing performance dashboard:

1. **FPS** should remain stable (target: 0.5 FPS for TUI)
2. **Avg Frame Time** should decrease (target: < 50ms)
3. **Slow Operations** count should decrease
4. **Throttler Batch Size** should show > 1.0 if throttler is active

Access via: `F9` key in TUI to open Performance Dashboard

---

## Test Commands

```bash
# Run TUI with performance tracing (existing infrastructure)
TUI_PERF_TRACE=1 uv run gpt-trader tui --demo

# Profile specific update paths
uv run pytest tests/tui -v -k "performance"
```
