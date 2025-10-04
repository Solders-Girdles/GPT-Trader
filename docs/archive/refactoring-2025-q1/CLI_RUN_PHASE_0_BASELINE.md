# CLI Run Command – Phase 0 Baseline

**Date:** October 2025
**Status:** 📋 Baseline
**Next Phase:** Lifecycle Controller, Signal Manager, Error Reporter Extraction

## Current State

### File Metrics
- **run.py**: 51 lines
- **test_run.py**: 180 lines, 10 tests
- **Single function**: `handle_run_bot()`
- **Test coverage**: Comprehensive (success, dev_fast, KeyboardInterrupt, exceptions, logging)
- **Unique aspects**: Async execution, signal handling, long-running loops

### Responsibilities Analysis

#### `handle_run_bot()` (51 lines)
Located in `src/bot_v2/cli/commands/run.py:16-51`

**Current Responsibilities:**
1. **Signal Registration** (lines 30-31)
   - Create ShutdownHandler instance with bot
   - Register system signals (SIGINT, SIGTERM)

2. **Async Execution** (line 35)
   - Call `asyncio.run(bot.run(single_cycle=dev_fast))`
   - Hardwired to asyncio.run

3. **Dev Fast Mode** (lines 37-40)
   - Single cycle execution when dev_fast=True
   - Continuous execution when dev_fast=False
   - Different logging messages

4. **KeyboardInterrupt Handling** (lines 44-46)
   - Catch KeyboardInterrupt
   - Log graceful shutdown message
   - Return exit code 0

5. **Exception Handling** (lines 48-50)
   - Catch any Exception
   - Log error with exc_info=True
   - Return exit code 1

6. **Logging** (lines 27, 38, 40, 45, 49)
   - Info: "Starting bot execution (dev_fast=...)"
   - Info: "Single cycle completed successfully" (dev_fast)
   - Info: "Bot execution completed" (continuous)
   - Info: "KeyboardInterrupt received, shutdown complete"
   - Error: "Bot execution failed: ..." with exc_info

7. **Exit Codes** (lines 42, 46, 50)
   - Return 0 on success
   - Return 0 on KeyboardInterrupt (graceful)
   - Return 1 on exception (error)

### Code Structure

```python
def handle_run_bot(bot: PerpsBot, dev_fast: bool = False) -> int:
    logger.info("Starting bot execution (dev_fast=%s)", dev_fast)

    # Signal registration (2 lines)
    shutdown_handler = ShutdownHandler(bot)
    shutdown_handler.register_signals()

    try:
        # Async execution (1 line)
        asyncio.run(bot.run(single_cycle=dev_fast))

        # Success logging (4 lines)
        if dev_fast:
            logger.info("Single cycle completed successfully")
        else:
            logger.info("Bot execution completed")

        return 0

    except KeyboardInterrupt:
        # Graceful interrupt (2 lines)
        logger.info("KeyboardInterrupt received, shutdown complete")
        return 0

    except Exception as e:
        # Error handling (2 lines)
        logger.error("Bot execution failed: %s", e, exc_info=True)
        return 1
```

### Dependencies

**Direct Dependencies:**
- `asyncio` - For asyncio.run()
- `ShutdownHandler` - For signal registration
- `PerpsBot` - Bot instance to run

**Implicit Dependencies:**
- Signal handling system (SIGINT, SIGTERM)
- Async event loop
- Bot.run() coroutine implementation

### Test Coverage Analysis

#### Existing Tests (10 tests)

**Happy Path (2 tests):**
- ✅ Successful bot run in dev_fast mode
- ✅ Successful bot run in continuous mode

**Signal Handling (1 test):**
- ✅ Shutdown handler registered before run

**Dev Fast Mode (3 tests):**
- ✅ Dev fast mode runs single cycle
- ✅ Continuous mode runs without single_cycle
- ✅ Default dev_fast is False

**Error Handling (2 tests):**
- ✅ KeyboardInterrupt handled gracefully (exit code 0)
- ✅ Exception returns error code 1

**Logging (2 tests):**
- ✅ Exception logged with details
- ✅ KeyboardInterrupt logged appropriately

### Comparison to Other CLI Commands

**Similarities:**
- Single function handler
- Exception handling with logging
- Return exit codes

**Unique Differences:**
- **Async execution** - Uses asyncio.run() (other commands are sync)
- **Signal handling** - Registers SIGINT/SIGTERM (other commands use ensure_shutdown)
- **Long-running** - Continuous loop or single cycle (other commands are one-shot)
- **No parser** - No string parsing (other commands parse FROM:TO:AMOUNT)
- **No JSON output** - No result formatting (other commands print JSON)
- **No printer injection** - No testable output (other commands use injectable printer)

### Current Test Gaps

Despite 10 tests, gaps exist:

1. **No isolated lifecycle tests** - Async execution mixed with handler
2. **No isolated signal manager tests** - Signal registration coupled to handler
3. **No isolated error reporter tests** - Logging logic mixed with handler
4. **Hardwired asyncio.run** - Cannot inject event loop for testing
5. **No printer injection** - Cannot verify output (though run doesn't print JSON)

## Refactor Plan

### Unique Challenges

Unlike other CLI commands, run.py has:
- **Async execution** - Requires different extraction approach
- **Signal handling** - Needs isolation from async loop
- **Long-running nature** - Different from one-shot commands
- **No output** - No JSON printing, different validation approach

### Extraction Strategy

#### Phase 1: Lifecycle Controller Extraction

**Goal:** Extract async execution logic into testable LifecycleController

**Target Structure:**
```python
# lifecycle_controller.py (new)
class LifecycleController:
    def __init__(self, runner: Callable = None):
        self._runner = runner or asyncio.run

    def execute(self, bot: PerpsBot, single_cycle: bool) -> int:
        """Execute bot lifecycle (async run)."""
        # Call runner with bot.run(single_cycle)
        # Return exit code
```

**Benefits:**
- Injectable async runner (can mock asyncio.run)
- Testable without actual event loop
- Isolates lifecycle concerns
- Single responsibility (execution)

#### Phase 2: Signal Manager Extraction

**Goal:** Extract signal handling into SignalManager

**Target Structure:**
```python
# signal_manager.py (new)
class SignalManager:
    def __init__(self, shutdown_handler_cls: type = None):
        self._handler_cls = shutdown_handler_cls or ShutdownHandler

    def setup_signals(self, bot: PerpsBot) -> None:
        """Setup signal handlers for bot."""
        # Create handler
        # Register signals
```

**Benefits:**
- Injectable ShutdownHandler class
- Testable signal setup
- Isolates signal concerns
- Can verify registration without signals

#### Phase 3: Error Reporter Extraction

**Goal:** Extract logging/error reporting into ErrorReporter

**Target Structure:**
```python
# error_reporter.py (new)
class ErrorReporter:
    def log_start(self, dev_fast: bool) -> None:
        """Log bot execution start."""

    def log_success(self, dev_fast: bool) -> None:
        """Log successful completion."""

    def log_interrupt(self) -> None:
        """Log KeyboardInterrupt."""

    def log_error(self, error: Exception) -> None:
        """Log execution error."""
```

**Benefits:**
- Centralized logging
- Testable log messages
- Consistent log formatting
- Can verify logging without logs

### Modified Handler Structure

```python
# run.py (after extraction)
def handle_run_bot(bot: PerpsBot, dev_fast: bool = False) -> int:
    reporter = ErrorReporter()
    signal_mgr = SignalManager()
    lifecycle = LifecycleController()

    reporter.log_start(dev_fast)
    signal_mgr.setup_signals(bot)

    try:
        exit_code = lifecycle.execute(bot, dev_fast)
        reporter.log_success(dev_fast)
        return exit_code
    except KeyboardInterrupt:
        reporter.log_interrupt()
        return 0
    except Exception as e:
        reporter.log_error(e)
        return 1
```

### Expected Outcomes

#### Metrics
- **run.py**: 51 → ~25 lines (**~50% reduction**)
- **lifecycle_controller.py**: ~30 lines (new)
- **signal_manager.py**: ~25 lines (new)
- **error_reporter.py**: ~35 lines (new)
- **New tests**: ~12-15 tests (4-5 per component)
- **Total CLI tests**: 191 → ~205 tests

#### Benefits
- ✅ **Testability**: Each component tested in isolation
- ✅ **Injectable dependencies**: Can mock async runner, signals, logging
- ✅ **Separation of Concerns**: Lifecycle → Signals → Logging → Orchestration
- ✅ **Reusability**: Components can be used in other contexts
- ✅ **Consistency**: Follows extraction pattern (though adapted for async)

## Design Decisions

### 1. Injectable Async Runner

**Decision:** Inject asyncio.run (or alternative) into LifecycleController

**Rationale:**
- Allows mocking for tests
- Can swap event loop implementation
- Testable without actual async execution
- No hardwired asyncio.run

### 2. Signal Manager Wraps ShutdownHandler

**Decision:** SignalManager creates and configures ShutdownHandler

**Rationale:**
- ShutdownHandler is existing, working code
- No need to reimplement signal handling
- SignalManager is thin wrapper for testability
- Maintains existing signal behavior

### 3. Error Reporter for Logging

**Decision:** Centralize all logging in ErrorReporter

**Rationale:**
- Consistent log messages
- Testable logging behavior
- Single place to change log formats
- Clear responsibility (reporting)

### 4. Keep Exception Handling in Handler

**Decision:** try/except blocks remain in handle_run_bot

**Rationale:**
- Handler orchestrates flow control
- Exception handling is control flow
- Clear exit code logic
- Each component can focus on happy path

### 5. No Printer Injection (Different from Other Commands)

**Decision:** Don't add printer injection (run doesn't output JSON)

**Rationale:**
- Run command doesn't print results
- Logging is via logger, not stdout
- Different pattern than account/convert/move_funds
- No need to capture output for tests

## Risk Assessment

### Medium Risk Factors ⚠️
- **Async execution** - Different from other command extractions
- **Signal handling** - System-level concerns
- **Existing tests** - 10 tests already pass, must maintain compatibility
- **No similar pattern** - First async command extraction

### Low Risk Factors ✅
- Well-tested baseline (10 comprehensive tests)
- Small, focused function (51 lines)
- Clear responsibilities
- Existing ShutdownHandler abstraction

### Mitigation Strategies
- Extract in small phases (Lifecycle → Signal → Error)
- Maintain all existing tests
- Add focused component tests
- Injectable dependencies for testability

## Next Steps

### Phase 1: Lifecycle Controller Extraction

1. **Create LifecycleController** (~30 lines)
   - Injectable runner (default asyncio.run)
   - execute(bot, single_cycle) method
   - Exception propagation
   - Exit code management

2. **Add Lifecycle Controller Tests** (~4-5 tests)
   - Execute with single_cycle=True
   - Execute with single_cycle=False
   - Custom runner injection
   - Exception propagation
   - Exit code verification

3. **Integrate into Handler** (~partial integration)
   - Use LifecycleController for async execution
   - Keep signal and error handling in handler
   - Update imports

4. **Validate**
   - Run all 10 existing run tests (should pass)
   - Run 4-5 new lifecycle tests (should pass)
   - Verify zero regressions

### Phase 2: Signal Manager Extraction

- Extract signal setup logic
- Add ~4 tests for signal manager
- Integrate, validate

### Phase 3: Error Reporter Extraction

- Extract logging logic
- Add ~4 tests for error reporter
- Integrate, validate

### Final State

**Projected:** ~25 lines for handler, ~15 new tests, zero regressions

---

**Phase 0 Status:** ✅ Baseline Complete
**Current Lines:** 51
**Target Lines:** ~25 (50% reduction)
**Pattern:** Extract → Test → Integrate → Validate (adapted for async)
**Risk Level:** Medium ⚠️ (async, signals, different pattern)
**Test Coverage:** 10 existing tests (comprehensive)
**Ready for Phase 1:** ✅ Yes
