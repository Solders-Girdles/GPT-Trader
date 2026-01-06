# TUI Performance Follow-up Tickets

**Created:** 2026-01-06
**Source:** TUI_PERFORMANCE_AUDIT.md

---

## Ticket 1: Enable Update Throttler by Default ✅ COMPLETED

**Priority:** Medium
**Effort:** Small (< 1 hour)
**Risk:** Low
**Status:** Completed (2026-01-06)

### Description
The `UpdateThrottler` infrastructure exists but is never instantiated. Enable it to batch high-frequency market data updates.

### Files Modified
- `src/gpt_trader/tui/app_lifecycle.py` - Added throttler creation in `_initialize_with_bot()`

### Implementation
```python
# In _initialize_with_bot(), before UICoordinator creation:
from gpt_trader.tui.services import UpdateThrottler

throttler = UpdateThrottler(min_interval=0.1)  # 100ms batching
self.ui_coordinator = UICoordinator(self, throttler=throttler)
```

### Acceptance Criteria
- [x] Throttler instantiated with 100ms interval
- [ ] Performance dashboard shows throttler stats (batch size > 1.0 under load)
- [x] No regression in TUI responsiveness (1034 tests pass)

---

## Ticket 2: Add No-Op Guards to watch_* Methods ✅ COMPLETED

**Priority:** Low
**Effort:** Medium (2-3 hours)
**Risk:** Low
**Status:** Completed (2026-01-06)

### Description
Most `watch_*` methods update widgets unconditionally even when the value hasn't changed, causing unnecessary DOM updates and potential flicker.

### Files Modified
- `src/gpt_trader/tui/widgets/status.py` - 11 watch methods guarded
- `src/gpt_trader/tui/widgets/slim_status.py` - 7 watch methods guarded
- `src/gpt_trader/tui/widgets/dashboard_system.py` - 13 watch methods guarded

### Implementation Pattern
```python
def watch_equity(self, equity: str) -> None:
    _last = getattr(self, "_last_equity", object())
    if equity == _last:
        return
    self._last_equity = equity
    # ... rest of method
```

Using `object()` as sentinel ensures the first call always runs (no value equals
a fresh object instance).

### Acceptance Criteria
- [x] No visual flicker on unchanged values
- [x] Flash animations only trigger on actual changes
- [x] All 244 widget tests pass

---

## Ticket 3: Add TUI_PERF_TRACE Environment Variable

**Priority:** Low
**Effort:** Small (< 1 hour)
**Risk:** None

### Description
Add an environment variable toggle for verbose performance tracing, useful for debugging without code changes.

### Files to Modify
- `src/gpt_trader/tui/services/performance_service.py`
- `src/gpt_trader/tui/managers/ui_coordinator.py`

### Implementation
```python
# In performance_service.py:
import os
PERF_TRACE_ENABLED = os.environ.get("TUI_PERF_TRACE", "").lower() in ("1", "true")

# In key methods, add trace logging:
if PERF_TRACE_ENABLED:
    logger.info(f"perf: {operation_name} {duration*1000:.1f}ms")
```

### Trace Points
- `UICoordinator._apply_status_update()` total time
- `TuiState.update_from_bot_status()` time
- `StateRegistry.broadcast()` time + observer count
- Individual widget `on_state_updated()` times (top 3 slowest)

### Acceptance Criteria
- [ ] `TUI_PERF_TRACE=1 uv run gpt-trader tui --demo` produces timing logs
- [ ] No overhead when env var is unset
- [ ] Logs include operation name + duration in ms

---

## Ticket 4: Investigate on_state_updated Efficiency

**Priority:** Low
**Effort:** Medium (2-3 hours)
**Risk:** Low

### Description
Some `on_state_updated()` implementations do expensive operations (string parsing, try/except chains) on every broadcast. Profile and optimize the heaviest ones.

### Files to Investigate
- `src/gpt_trader/tui/widgets/dashboard_system.py:119` - SystemMonitorWidget
- `src/gpt_trader/tui/widgets/dashboard_position.py:87` - PositionCardWidget
- `src/gpt_trader/tui/widgets/account.py:133` - AccountWidget

### Potential Optimizations
1. Cache parsed values instead of re-parsing each cycle
2. Use early returns when relevant data section unchanged
3. Batch multiple label updates into single render

### Acceptance Criteria
- [ ] Profile before/after with `TUI_PERF_TRACE`
- [ ] Top 3 widgets show measurable improvement
- [ ] No functional regressions

---

## Ticket 5: Document Performance Budget in TUI_STYLE_GUIDE ✅ COMPLETED

**Priority:** Low
**Effort:** Small (< 30 min)
**Risk:** None
**Status:** Completed (2026-01-06)

### Description
Add performance budget section to TUI_STYLE_GUIDE.md documenting the thresholds already defined in `PerformanceBudget`.

### Files Modified
- `docs/TUI_STYLE_GUIDE.md` - Enhanced Performance Budgets section

### Changes Made
- Added reference to `PerformanceBudget` class location
- Added `F9` access instructions
- Added CPU % to Resource Usage table
- Updated "command palette" reference to "F9"

Note: TUI_STYLE_GUIDE.md already had comprehensive performance budget tables;
this ticket added the class reference and access method.

### Acceptance Criteria
- [x] Budget table added to TUI_STYLE_GUIDE.md (already existed, enhanced)
- [x] References `PerformanceBudget` class location

---

## Summary

| Ticket | Priority | Effort | Impact | Status |
|--------|----------|--------|--------|--------|
| 1. Enable Throttler | Medium | Small | High - Reduces update frequency | ✅ Done |
| 2. No-Op Guards | Low | Medium | Medium - Reduces DOM updates | ✅ Done |
| 3. PERF_TRACE Env | Low | Small | Low - Debugging aid | Pending |
| 4. Optimize on_state_updated | Low | Medium | Medium - Faster cycles | Pending |
| 5. Document Budget | Low | Small | Low - Developer awareness | ✅ Done |

**Recommended order:** ~~1~~ → ~~5~~ → ~~2~~ → 3 → 4
