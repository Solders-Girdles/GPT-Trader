# CLI Account Command – Phase 0 Baseline

**Date:** October 2025
**Status:** 📋 Baseline
**Next Phase:** AccountSnapshotService Extraction

## Current State

### File Metrics
- **account.py**: 51 lines
- **test_account.py**: 247 lines, 15 tests
- **Single function**: `handle_account_snapshot()`
- **Test coverage**: Comprehensive (success, errors, edge cases, shutdown)

### Responsibilities Analysis

#### `handle_account_snapshot()` (51 lines)
Located in `src/bot_v2/cli/commands/account.py:16-51`

**Current Responsibilities:**
1. **Telemetry Validation** (lines 31-35)
   - Check `bot.account_telemetry` existence via `getattr`
   - Verify `supports_snapshots()` capability
   - Error handling with logging and shutdown

2. **Snapshot Collection** (lines 37-39)
   - Call `telemetry.collect_snapshot()`
   - Success logging

3. **JSON Formatting** (lines 41-43)
   - Format snapshot as JSON with `indent=2`
   - Handle non-serializable types with `default=str`
   - Print to stdout

4. **Error Handling** (lines 46-48)
   - Exception catching and logging
   - Re-raise exceptions

5. **Shutdown Management** (lines 34, 50)
   - Call `ensure_shutdown(bot)` on error
   - Call `ensure_shutdown(bot)` in finally block

6. **Logging** (lines 29, 33, 39, 47)
   - Info: "Collecting account snapshot..."
   - Error: Telemetry unavailable
   - Info: "Account snapshot collected successfully"
   - Error: Collection failure with exc_info

### Code Structure

```python
def handle_account_snapshot(bot: PerpsBot) -> int:
    logger.info("Collecting account snapshot...")

    # Telemetry validation (5 lines)
    telemetry = getattr(bot, "account_telemetry", None)
    if telemetry is None or not telemetry.supports_snapshots():
        logger.error("Account snapshot telemetry is not available for this broker")
        ensure_shutdown(bot)
        raise RuntimeError("Account snapshot telemetry is not available for this broker")

    try:
        # Snapshot collection (2 lines)
        snapshot = telemetry.collect_snapshot()
        logger.info("Account snapshot collected successfully")

        # JSON formatting and output (3 lines)
        output = json.dumps(snapshot, indent=2, default=str)
        print(output)

        return 0
    except Exception as e:
        # Error handling (2 lines)
        logger.error("Failed to collect account snapshot: %s", e, exc_info=True)
        raise
    finally:
        # Shutdown (1 line)
        ensure_shutdown(bot)
```

### Test Coverage Analysis

#### Existing Tests (15 tests)

**Happy Path (3 tests):**
- ✅ Successful snapshot collection
- ✅ JSON output validation
- ✅ JSON formatting (indentation)

**Error Cases (5 tests):**
- ✅ No telemetry available (getattr returns None)
- ✅ Telemetry doesn't support snapshots
- ✅ Collection exception
- ✅ Telemetry attribute check (bot with no attributes)
- ✅ supports_snapshots returns False

**Edge Cases (4 tests):**
- ✅ Empty positions list
- ✅ Multiple positions
- ✅ Nested data structures
- ✅ Non-JSON-serializable types (Decimal, datetime)

**Shutdown Behavior (3 tests):**
- ✅ Shutdown called on success
- ✅ Shutdown called on error before collection
- ✅ Shutdown called on error during collection

### Test Coverage Gaps

**Current gaps identified:**
1. **No isolated tests for snapshot service logic** - All tests go through handler
2. **No printer injection** - Tests rely on capsys for output verification
3. **No service-level telemetry validation tests** - Validation mixed with handler
4. **No separate JSON formatting tests** - Formatting logic coupled to handler

## Refactor Plan

### Phase 1: AccountSnapshotService Extraction

**Goal:** Extract snapshot collection and JSON formatting into dedicated service

#### Target Structure

```python
# account_snapshot_service.py (new)
class AccountSnapshotService:
    def __init__(self, printer: Callable[[str], None] | None = None):
        self._printer = printer or print
        self._logger = logging.getLogger(__name__)

    def collect_and_print(self, bot: PerpsBot) -> int:
        """Collect account snapshot and print as formatted JSON."""
        # Telemetry validation
        # Snapshot collection
        # JSON formatting
        # Output printing
        # Return exit code
```

```python
# account.py (modified)
def handle_account_snapshot(bot: PerpsBot) -> int:
    service = AccountSnapshotService()
    try:
        return service.collect_and_print(bot)
    finally:
        ensure_shutdown(bot)
```

#### Extraction Details

**Lines to Extract (20-25 lines):**
- Telemetry validation logic (5 lines)
- Snapshot collection (2 lines)
- JSON formatting (3 lines)
- Logging statements (4 lines)

**Lines to Remain in Handler (5-10 lines):**
- Service instantiation (1 line)
- Service call in try block (1 line)
- Finally block with shutdown (2 lines)
- Function signature and docstring (4 lines)

**New Service Tests (~5-7 tests):**
- ✅ Collect and print with valid telemetry
- ✅ Printer override (inject mock printer)
- ✅ Telemetry validation (no telemetry)
- ✅ Telemetry validation (doesn't support snapshots)
- ✅ Collection exception handling
- ✅ JSON formatting verification
- ✅ Logging verification (info + error)

### Expected Outcomes

#### Metrics
- **account.py**: 51 → ~30 lines (**~40% reduction**)
- **account_snapshot_service.py**: ~50 lines (new)
- **test_account_snapshot_service.py**: ~5-7 tests (new)
- **Total CLI tests**: 153 → ~160 tests

#### Benefits
- ✅ **Testability**: Service can be tested in isolation with printer injection
- ✅ **Separation of Concerns**: Handler focuses on shutdown, service handles snapshot logic
- ✅ **Reusability**: Service can be used in other contexts
- ✅ **Consistency**: Follows OrderPreviewService and EditPreviewService patterns

### Design Decisions

#### 1. Injectable Printer Pattern

**Decision:** Use same printer injection pattern as OrderPreviewService

**Rationale:**
- Proven pattern from orders command refactor
- Testability without capturing stdout
- Easy to verify JSON output in tests
- No coupling to print implementation

#### 2. Keep Shutdown in Handler

**Decision:** Leave `ensure_shutdown(bot)` in handler's finally block

**Rationale:**
- Shutdown is handler-level concern (CLI orchestration)
- Service shouldn't know about shutdown lifecycle
- Clear separation: service = business logic, handler = orchestration
- Maintains existing shutdown guarantees

#### 3. Telemetry Validation in Service

**Decision:** Move telemetry validation into service

**Rationale:**
- Validation is part of snapshot collection workflow
- Service needs to handle RuntimeError anyway
- Can be tested independently with mock telemetry
- Consolidates all snapshot logic in one place

#### 4. Exception Propagation

**Decision:** Let service raise exceptions, handler re-raises

**Rationale:**
- Service logs errors before raising
- Handler ensures shutdown via finally block
- Maintains existing error behavior
- Clear error reporting at both levels

## Risk Assessment

### Low Risk Factors ✅
- Small, focused function (51 lines)
- Comprehensive test coverage (15 tests)
- Clear separation of concerns (validation → collection → formatting)
- Proven extraction pattern (orders command)

### Considerations
- Shutdown must remain in handler's finally block
- RuntimeError behavior must be preserved
- JSON formatting (indent=2, default=str) must be exact
- All logging messages must be preserved

## Next Steps

1. **Create AccountSnapshotService** (~50 lines)
   - `collect_and_print(bot)` method
   - Telemetry validation
   - Snapshot collection
   - JSON formatting and printing
   - Injectable printer

2. **Add Service Tests** (~5-7 tests)
   - Happy path with telemetry
   - Printer injection
   - Telemetry validation errors
   - Collection exceptions
   - Logging verification

3. **Integrate into Handler** (~30 lines)
   - Instantiate service
   - Call service in try block
   - Keep shutdown in finally
   - Update imports

4. **Validate**
   - Run all 15 existing account tests (should pass)
   - Run 5-7 new service tests (should pass)
   - Verify zero regressions in CLI suite

---

**Phase 0 Status:** ✅ Baseline Complete
**Current Lines:** 51
**Target Lines:** ~30 (40% reduction)
**Pattern:** Extract → Test → Integrate → Validate
**Risk Level:** Low ✅
**Ready for Phase 1:** ✅ Yes
