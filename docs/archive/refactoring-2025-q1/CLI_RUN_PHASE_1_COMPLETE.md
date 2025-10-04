# CLI Run Command – Phase 1 Complete

**Date:** October 2025
**Phase:** Lifecycle Controller Extraction
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate (Async Adaptation)

## Summary

Extracted async execution logic into `LifecycleController` with injectable runner, improving testability and removing hardwired asyncio.run dependency. Updated all existing tests to work with new structure.

### Metrics
- `run.py`: 51 → 53 lines (+2 lines, but improved testability)
- `lifecycle_controller.py`: 42 lines (injectable async runner)
- Tests: +6 lifecycle controller tests
- Total CLI tests: 191 → 197 (all green)
- **Note**: Line count slightly increased but async execution is now fully testable

### Highlights
- `LifecycleController` with injectable async runner (default: asyncio.run)
- `execute(bot, single_cycle)` method handles async bot execution
- Removed hardwired `asyncio.run()` from handler
- All 10 existing run tests updated to mock LifecycleController
- 6 new focused lifecycle controller tests
- Zero regressions - all tests passing

## Changes Made

### New Files

#### 1. `lifecycle_controller.py` (42 lines)

**Purpose:** Manages bot lifecycle execution with injectable async runner

**Components:**
- `LifecycleController` class with injectable runner
- `execute()` method for bot execution
- Exception propagation
- Exit code management

**Design:**
```python
class LifecycleController:
    """Controls bot lifecycle execution."""

    def __init__(self, runner: Callable[[Any], Any] | None = None) -> None:
        if runner is None:
            import asyncio
            runner = asyncio.run
        self._runner = runner

    def execute(self, bot: Any, single_cycle: bool) -> int:
        """Execute bot lifecycle."""
        self._runner(bot.run(single_cycle=single_cycle))
        return 0
```

**Benefits:**
- **Injectable runner**: Can mock asyncio.run for tests
- **Testable**: No actual async execution needed in tests
- **Focused**: Single responsibility (lifecycle execution)
- **Swappable**: Can use different event loop implementations

#### 2. `test_lifecycle_controller.py` (6 tests)

**Test Coverage:**

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| Execution | 3 | single_cycle=True/False, custom runner |
| Error Handling | 1 | Exception propagation |
| Exit Codes | 1 | Returns 0 on success |
| Default Runner | 1 | asyncio.run is default |

**Test Categories:**
- ✅ Execute with single_cycle=True
- ✅ Execute with single_cycle=False
- ✅ Custom runner injection
- ✅ Exception propagation
- ✅ Returns 0 on successful execution
- ✅ Default runner uses asyncio.run

### Modified Files

#### `run.py` (51 → 53 lines, +2 lines)

**Before:**
- Hardwired `asyncio.run(bot.run(single_cycle=dev_fast))`
- Direct async execution
- Import asyncio

**After:**
- Uses `LifecycleController().execute(bot, single_cycle=dev_fast)`
- Injectable async execution
- Removed asyncio import
- Added LifecycleController import and setup

**Key Changes:**
```python
# Before
import asyncio
...
asyncio.run(bot.run(single_cycle=dev_fast))

# After
from bot_v2.cli.commands.lifecycle_controller import LifecycleController
...
lifecycle = LifecycleController()
...
lifecycle.execute(bot, single_cycle=dev_fast)
```

#### `test_run.py` (10 tests updated)

**Changes:**
- Replaced all `@patch("bot_v2.cli.commands.run.asyncio.run")` with `@patch("bot_v2.cli.commands.run.LifecycleController")`
- Mock LifecycleController class and instance
- Verify `lifecycle.execute()` calls instead of `asyncio.run()` calls
- All tests still verify same behavior (single_cycle, KeyboardInterrupt, exceptions, etc.)

**Example:**
```python
# Before
@patch("bot_v2.cli.commands.run.asyncio.run")
def test_successful_bot_run_dev_mode(self, mock_asyncio_run, ...):
    mock_asyncio_run.side_effect = ...
    mock_bot.run.assert_called_once_with(single_cycle=True)

# After
@patch("bot_v2.cli.commands.run.LifecycleController")
def test_successful_bot_run_dev_mode(self, mock_lifecycle_class, ...):
    mock_lifecycle = Mock()
    mock_lifecycle.execute.return_value = 0
    mock_lifecycle_class.return_value = mock_lifecycle
    mock_lifecycle.execute.assert_called_once_with(mock_bot, single_cycle=True)
```

## Validation

### Test Results

**Lifecycle Controller Tests:**
```bash
$ pytest tests/.../test_lifecycle_controller.py -v
============================= 6 passed in 0.03s ==============================
```

**Full CLI Test Suite:**
```bash
$ pytest tests/unit/bot_v2/cli/ --tb=no -q
============================= 197 passed in 0.16s ==============================
```

**Zero regressions** - All 10 existing run tests + 6 new lifecycle tests passing

### Behavioral Verification

✅ **Async execution unchanged** - Same bot.run(single_cycle) call
✅ **Signal registration unchanged** - ShutdownHandler still registered
✅ **Dev fast mode unchanged** - single_cycle flag passed correctly
✅ **KeyboardInterrupt handling unchanged** - Same exit code 0
✅ **Exception handling unchanged** - Same exit code 1
✅ **Logging unchanged** - Same log messages at same levels
✅ **Exit codes unchanged** - Same return values

## Design Decisions

### 1. Injectable Async Runner

**Decision:** Inject asyncio.run (or alternative) into LifecycleController

**Rationale:**
- Allows mocking for tests (no actual async execution)
- Can swap event loop implementation
- Testable without actual event loop
- No hardwired asyncio.run dependency

### 2. Lazy Import of asyncio

**Decision:** Import asyncio inside `__init__` only when runner not provided

**Rationale:**
- Avoid importing asyncio when custom runner provided
- Only import when needed
- Cleaner module-level imports
- Follows lazy loading pattern

### 3. Keep Signal/Error Handling in Handler

**Decision:** Don't extract signal and error handling yet (Phase 2/3)

**Rationale:**
- Phase 1 focuses on async execution only
- Small, incremental changes
- Easier to validate
- Signal and error handling extraction in future phases

### 4. Update Existing Tests (Don't Duplicate)

**Decision:** Update 10 existing run tests to use LifecycleController

**Rationale:**
- Maintains test coverage of handler behavior
- Tests verify same scenarios (just with new mocking)
- No need for duplicate tests
- Easier to understand what changed

### 5. Line Count Increase Acceptable

**Decision:** Accept +2 line increase for testability gain

**Rationale:**
- Testability is more valuable than line count
- LifecycleController setup adds lines but removes complexity
- Future phases will reduce overall line count
- Injectable dependencies require setup code

## Lessons Learned

### What Worked Well ✅

1. **Injectable runner pattern** - Clean abstraction for async execution
2. **Test updates** - All 10 existing tests work with minimal changes
3. **Zero regressions** - Careful test updates maintained coverage
4. **Async isolation** - Lifecycle controller cleanly isolates async concerns

### Challenges Overcome 🔧

1. **Test failures** - Initial integration broke all run tests
   - **Solution**: Updated all tests to mock LifecycleController
   - **Lesson**: Always update tests when changing dependencies

2. **Line count increase** - From 51 to 53 lines
   - **Context**: Setup code for LifecycleController
   - **Tradeoff**: Testability > line count
   - **Future**: Phases 2/3 will reduce overall count

### Phase 1 Impact 🎯

**Testability Gains:**
- ✅ Can mock async execution (no real event loop)
- ✅ Can test lifecycle logic in isolation
- ✅ Can inject custom runners for testing
- ✅ No hardwired asyncio.run dependency

**Code Quality:**
- ✅ Async execution isolated in dedicated component
- ✅ Single responsibility for lifecycle controller
- ✅ Dependency injection pattern applied
- ✅ Handler focuses on orchestration

## Next Steps

### Phase 2: Signal Manager Extraction (Planned)

**Goal:** Extract signal handling into SignalManager

**Scope:**
- Create SignalManager class
- Injectable ShutdownHandler class
- `setup_signals(bot)` method
- ~4 tests for signal manager

**Expected:**
- Remove ~3 lines from run.py
- Add `signal_manager.py` (~25 lines)
- Cleaner signal handling isolation

### Phase 3: Error Reporter Extraction (Planned)

**Goal:** Extract logging into ErrorReporter

**Scope:**
- Create ErrorReporter class
- Methods: log_start/success/interrupt/error
- ~4 tests for error reporter

**Expected:**
- Remove ~10 lines from run.py
- Add `error_reporter.py` (~35 lines)
- Centralized logging

### Final State (After All Phases)

**Projected:**
- run.py: ~25-30 lines (orchestration only)
- Total new tests: ~15 (6 lifecycle + 4 signal + 4-5 error)
- Total reduction: ~40-45% from baseline
- Full testability for all components

## CLI Refactor Progress

### Completed Commands
1. ✅ **Orders** - 222 → 109 lines (51%) - OrderPreviewService, EditPreviewService
2. ✅ **Parser** - 304 → 77 lines (75%) - ArgumentValidator, BotConfigBuilder
3. ✅ **Account** - 51 → 33 lines (35%) - AccountSnapshotService
4. ✅ **Convert** - 64 → 35 lines (45%) - ConvertRequestParser, ConvertService
5. ✅ **Move Funds** - 64 → 35 lines (45%) - MoveFundsRequestParser, MoveFundsService
6. 🔄 **Run** - 51 → 53 lines (Phase 1/3) - LifecycleController ✅

### Pattern Evolution
- ✅ **5 sync commands** - Proven printer injection, frozen dataclass patterns
- ✅ **1 async command** - Adapted pattern for async execution
- ✅ **Injectable dependencies** - Works for both sync and async
- ✅ **Test-first approach** - All components tested in isolation

---

**Phase 1 Status:** ✅ Complete
**Line Count:** 53 (+2 from baseline, improved testability)
**Test Coverage:** ✅ 197 tests passing (6 new lifecycle tests)
**Zero Regressions:** ✅ Confirmed
**Ready for Phase 2:** ✅ Signal Manager Extraction (optional)
**Alternative:** ✅ Declare run command complete (testability achieved)
