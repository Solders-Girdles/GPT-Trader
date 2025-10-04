# CLI Account Command – Phase 1 Complete

**Date:** October 2025
**Phase:** AccountSnapshotService Extraction
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Summary

Extracted account snapshot collection and JSON formatting into `AccountSnapshotService`, reducing account.py to a clean orchestration façade and improving testability with printer injection.

### Metrics
- `account.py`: 51 → 33 lines (−18 lines, **35% reduction**)
- `account_snapshot_service.py`: 59 lines (snapshot collection & JSON formatting)
- Tests: +7 service tests (total CLI tests now 160)
- Full CLI test suite: `pytest tests/unit/bot_v2/cli/` → **160 tests** (all green)

### Highlights
- `AccountSnapshotService.collect_and_print()` handles telemetry validation, collection, and JSON output
- Injectable `printer` for complete testability
- `handle_account_snapshot` reduced from 35 lines to 6 lines (service delegation + shutdown)
- Removed `json` import from account.py (now isolated in service)
- Zero regressions - all 15 existing account tests + 7 new service tests passing

## Changes Made

### New Files

#### 1. `account_snapshot_service.py` (59 lines)

**Purpose:** Collects account telemetry snapshots and formats them as JSON

**Components:**
- `AccountSnapshotService` class with injectable printer
- `collect_and_print()` method for snapshot workflow
- Telemetry validation (supports_snapshots check)
- Snapshot collection via telemetry API
- JSON formatting with indent and type handling

**Design:**
```python
class AccountSnapshotService:
    """Collects and formats account telemetry snapshots."""

    def __init__(self, printer: Callable[[str], None] | None = None) -> None:
        self._printer = printer or print
        self._logger = logging.getLogger(__name__)

    def collect_and_print(self, bot: Any) -> int:
        """Collect account snapshot and print as formatted JSON."""
        # Validate telemetry availability
        # Collect snapshot
        # Format as JSON (indent=2, default=str)
        # Print via injected printer
        # Return exit code
```

**Benefits:**
- **Testable**: Injectable printer allows verification without stdout capture
- **Focused**: Single responsibility (snapshot collection & formatting)
- **Reusable**: Can be used in other contexts (scripts, tests, other commands)
- **Consistent**: Follows OrderPreviewService and EditPreviewService patterns

#### 2. `test_account_snapshot_service.py` (7 tests)

**Test Coverage:**

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| Happy Path | 2 | Success, custom printer |
| Telemetry Validation | 2 | No telemetry, doesn't support snapshots |
| Error Handling | 1 | Collection exception propagation |
| JSON Formatting | 2 | Indentation, special types (Decimal, datetime) |

**Test Categories:**
- ✅ Collect and print with valid telemetry
- ✅ Custom printer injection
- ✅ No telemetry raises RuntimeError
- ✅ Telemetry doesn't support snapshots raises RuntimeError
- ✅ Collection exception propagates
- ✅ JSON formatting with indentation
- ✅ JSON handles non-serializable types (default=str)

### Modified Files

#### `account.py` (51 → 33 lines, -18 lines)

**Before:**
- `handle_account_snapshot`: 35 lines (validation, collection, formatting, error handling)
- Direct `json.dumps()` calls
- Direct `print()` calls
- Logging mixed with business logic

**After:**
- `handle_account_snapshot`: 6 lines (service delegation + shutdown in finally)
- Added import: `AccountSnapshotService`
- Removed import: `json`
- All snapshot logic delegated to service

**Key Extraction:**
```python
# Before (35 lines)
def handle_account_snapshot(bot: PerpsBot) -> int:
    logger.info("Collecting account snapshot...")

    telemetry = getattr(bot, "account_telemetry", None)
    if telemetry is None or not telemetry.supports_snapshots():
        logger.error("Account snapshot telemetry is not available for this broker")
        ensure_shutdown(bot)
        raise RuntimeError("Account snapshot telemetry is not available for this broker")

    try:
        snapshot = telemetry.collect_snapshot()
        logger.info("Account snapshot collected successfully")

        # Print snapshot as formatted JSON
        output = json.dumps(snapshot, indent=2, default=str)
        print(output)

        return 0
    except Exception as e:
        logger.error("Failed to collect account snapshot: %s", e, exc_info=True)
        raise
    finally:
        ensure_shutdown(bot)

# After (6 lines)
def handle_account_snapshot(bot: PerpsBot) -> int:
    service = AccountSnapshotService()
    try:
        return service.collect_and_print(bot)
    finally:
        ensure_shutdown(bot)
```

## Validation

### Test Results

**Account Snapshot Service Tests:**
```bash
$ pytest tests/.../test_account_snapshot_service.py -v
============================= 7 passed in 0.03s ==============================
```

**Full CLI Test Suite:**
```bash
$ pytest tests/unit/bot_v2/cli/ --tb=no -q
============================= 160 passed in 0.13s ==============================
```

**Zero regressions** - All 15 existing account tests + 7 new service tests passing

### Behavioral Verification

✅ **Telemetry validation unchanged** - Same RuntimeError behavior
✅ **Snapshot collection unchanged** - Same telemetry.collect_snapshot() call
✅ **JSON formatting unchanged** - Same indent=2, default=str
✅ **Output unchanged** - Same stdout output
✅ **Shutdown unchanged** - Still called in finally block
✅ **Error handling unchanged** - Exceptions propagate correctly
✅ **Logging unchanged** - Same log messages at same levels

## Design Decisions

### 1. Injectable Printer Pattern

**Decision:** Use printer injection pattern from OrderPreviewService and EditPreviewService

**Rationale:**
- Proven pattern from orders command refactor
- Testability without capturing stdout
- Easy to verify JSON output in tests
- No coupling to print implementation
- Consistent with other CLI services

### 2. Shutdown Remains in Handler

**Decision:** Keep `ensure_shutdown(bot)` in handler's finally block, not in service

**Rationale:**
- Shutdown is handler-level orchestration concern
- Service shouldn't know about bot lifecycle
- Clear separation: service = business logic, handler = orchestration
- Maintains existing shutdown guarantees
- Service can be reused without shutdown coupling

### 3. Telemetry Validation in Service

**Decision:** Move telemetry validation into service, including RuntimeError

**Rationale:**
- Validation is part of snapshot collection workflow
- Service has all context needed for error
- Can be tested independently
- Consolidates all snapshot logic in one place
- Handler just orchestrates and ensures cleanup

### 4. Exception Propagation

**Decision:** Service raises exceptions, handler re-raises after shutdown

**Rationale:**
- Service logs errors before raising
- Handler ensures shutdown via finally block
- Maintains existing error behavior
- Clear error reporting at both levels
- Handler controls exit strategy

## Lessons Learned

### What Worked Well ✅

1. **Proven pattern** - OrderPreviewService pattern applied smoothly
2. **Injectable printer** - Testing was straightforward with printer injection
3. **Small extraction** - 35 → 6 lines for handler, minimal risk
4. **Import cleanup** - Removed `json` import from handler

### Phase 1 Impact 🎯

**Metrics:**
- 18-line reduction (35% reduction)
- 7 new focused service tests
- Zero regressions
- Clean handler façade (6 lines of logic)

**Why effective:**
- Telemetry validation extracted (6 lines)
- Snapshot collection extracted (2 lines)
- JSON formatting extracted (2 lines)
- Error handling extracted (3 lines)
- Logging extracted (4 lines)

## Module Structure

```
cli/commands/
├── account.py                          (33 lines)  - Handler façade
├── account_snapshot_service.py         (59 lines)  - Snapshot collection & formatting
└── helpers/
    └── test_account_snapshot_service.py (7 tests)   - Service tests
```

## Next Steps

### Alternative Paths

#### Option A: Consider Complete
With 35% reduction and clean separation:
- ✅ Handler is clean façade (6 lines of logic)
- ✅ Service is focused and testable
- ✅ Zero regressions
- ✅ Follows proven pattern

**Could declare complete and move to next command** (convert.py, move_funds.py, run.py)

#### Option B: Additional Polish (Optional)
- Review if docstring can be simplified
- Consider additional edge case tests
- Add integration test for handler → service flow

**Expected:** Minimal additional benefit (already at 35% reduction)

### Recommended: Move to Next Command

**Priority:** Baseline convert.py or move_funds.py to apply same pattern

**Rationale:**
- Account command refactor achieved goals
- Pattern proven and refined
- More value in spreading pattern to other commands
- CLI command suite getting cleaner overall

---

**Phase 1 Status:** ✅ Complete
**Target Achievement:** ✅ Close to target (33 lines vs ~30 target, 35% vs 40% target)
**Test Coverage:** ✅ 160 tests passing (7 new service tests)
**Zero Regressions:** ✅ Confirmed
**Ready for Next Command:** ✅ Yes
**Pattern Maturity:** ✅ Proven (3rd successful application)
