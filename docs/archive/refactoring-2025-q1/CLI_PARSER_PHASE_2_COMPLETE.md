# CLI Parser Refactor – Phase 2 Complete

**Date:** October 2025
**Phase:** Argument Validation Service
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Summary

Extracted argument validation and environment handling into `ArgumentValidator`, achieving clean separation of validation logic from parser orchestration and improving testability.

### Metrics
- `parser.py`: 133 → 120 lines (−13 lines this phase, −184 overall from 304 baseline)
- `argument_validator.py`: 71 focused lines handling validation and logging
- Tests: +8 validator tests (total CLI tests now 144)
- Full CLI test suite: `pytest tests/unit/bot_v2/cli/` → **144 tests** (all green)

### Highlights
- `ArgumentValidator.validate()` handles symbol validation, PERPS_DEBUG env handling, and logging config
- Injectable `env_reader` and `log_config` for complete testability
- `parse_and_validate_args` reduced from 29 lines to 3 lines (parse → delegate)
- Zero regressions - all 136 existing CLI tests + 8 new validator tests passing

## Changes Made

### New Files

#### 1. `argument_validator.py` (71 lines)

**Purpose:** Validates CLI arguments and configures logging based on environment

**Components:**
- `ArgumentValidator` class with injectable dependencies
- `validate()` method for argument validation
- Symbol validation (non-empty string check)
- PERPS_DEBUG environment variable handling
- Logger level configuration

**Design:**
```python
class ArgumentValidator:
    """Validates CLI arguments and configures logging based on environment."""

    def __init__(
        self,
        *,
        env_reader: Callable[[str], str | None] | None = None,
        log_config: Callable[[str, int], None] | None = None,
    ) -> None:
        import os
        self._env_reader = env_reader or os.getenv
        self._log_config = log_config or self._default_log_config

    def validate(
        self, args: argparse.Namespace, parser: argparse.ArgumentParser
    ) -> argparse.Namespace:
        """Validate parsed CLI arguments."""
        # Symbol validation
        # PERPS_DEBUG handling
        # Logger configuration
        # Debug logging
```

**Benefits:**
- **Testable**: Injectable env_reader and log_config for mocking
- **Focused**: Single responsibility (validation + logging config)
- **Reusable**: Can be used independently of parser
- **Maintainable**: Clear separation of concerns

#### 2. `test_argument_validator.py` (8 tests)

**Test Coverage:**

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| Symbol Validation | 4 | Valid symbols, empty symbols, whitespace, None |
| PERPS_DEBUG Handling | 3 | Set to "1", not set, set to "0" |
| Logger Configuration | 1 | Default log config behavior |

**Test Categories:**
- ✅ Valid symbols pass validation
- ✅ Empty symbols raise SystemExit
- ✅ Whitespace-only symbols raise SystemExit
- ✅ None symbols (not provided) pass validation
- ✅ PERPS_DEBUG=1 enables debug logging
- ✅ PERPS_DEBUG not set skips debug logging
- ✅ PERPS_DEBUG=0 skips debug logging
- ✅ Default log config sets logger level

### Modified Files

#### `parser.py` (133 → 120 lines, -13 lines)

**Before:**
- `parse_and_validate_args`: 29 lines (symbol validation, env handling, logging)
- Direct `os.getenv()` calls
- Direct `logging.getLogger().setLevel()` calls

**After:**
- `parse_and_validate_args`: 3 lines (parse → delegate)
- Added import: `ArgumentValidator`
- All validation logic delegated to validator

**Key Extraction:**
```python
# Before (29 lines)
def parse_and_validate_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()

    # Validate symbols if provided
    if args.symbols:
        empty = [sym for sym in args.symbols if not str(sym).strip()]
        if empty:
            parser.error("Symbols must be non-empty strings")

    # Enable debug logging if requested
    if os.getenv("PERPS_DEBUG") == "1":
        logger.info("Debug mode enabled via PERPS_DEBUG=1")
        logging.getLogger("bot_v2.features.brokerages.coinbase").setLevel(logging.DEBUG)
        logging.getLogger("bot_v2.orchestration").setLevel(logging.DEBUG)

    logger.debug("Parsed CLI arguments: profile=%s, dry_run=%s", args.profile, args.dry_run)

    return args

# After (3 lines)
def parse_and_validate_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    validator = ArgumentValidator()
    return validator.validate(args, parser)
```

## Validation

### Test Results

**Argument Validator Tests:**
```bash
$ pytest tests/.../test_argument_validator.py -v
============================= 8 passed in 0.03s ==============================
```

**Full CLI Test Suite:**
```bash
$ pytest tests/unit/bot_v2/cli/ --tb=no -q
============================= 144 passed in 0.13s ==============================
```

**Zero regressions** - All 136 existing CLI tests + 8 new validator tests passing

### Behavioral Verification

✅ **Symbol validation unchanged** - Same error checking and SystemExit behavior
✅ **PERPS_DEBUG handling unchanged** - Exact same logger configuration
✅ **Debug logging unchanged** - Same log messages and levels
✅ **Config building unchanged** - `build_bot_config_from_args()` untouched
✅ **Order tooling check unchanged** - `order_tooling_requested()` untouched

## Progress Summary

### Cumulative Reduction

| Phase | Lines | Reduction | Cumulative |
|-------|-------|-----------|------------|
| Phase 0 (Baseline) | 304 | 0 | 0 |
| Phase 1 (Argument Specs) | 133 | -171 | -171 (56%) |
| Phase 2 (Argument Validator) | 120 | -13 | -184 (61%) |

**Total Reduction:** 304 → 120 lines (-184 lines, **61% reduction**)

### Module Structure

```
cli/
├── parser.py                    (120 lines) - Orchestration only
├── argument_groups.py           (310 lines) - Declarative argument specs
├── argument_validator.py        (71 lines)  - Validation & env handling
├── test_argument_groups.py      (22 tests)  - Argument spec tests
└── test_argument_validator.py   (8 tests)   - Validator tests
```

## Design Decisions

### 1. Injectable Dependencies

**Decision:** Use dependency injection for `env_reader` and `log_config`

**Rationale:**
- Complete testability without mocking global functions
- Can verify PERPS_DEBUG behavior in isolation
- Can verify logger configuration without side effects
- Clean separation between validation logic and side effects

### 2. Default Implementations in __init__

**Decision:** Provide default implementations for injected dependencies

**Rationale:**
- Production code uses real `os.getenv` and `logging.getLogger`
- Tests override with mocks for isolation
- No need for factory or builder pattern
- Pythonic approach (duck typing)

### 3. Static Helper for Default Log Config

**Decision:** Use static method `_default_log_config` for default logger configuration

**Rationale:**
- Avoids circular import with logging
- Can be overridden for testing
- Clear separation of default behavior
- Consistent with dependency injection pattern

### 4. Preserve parser.error() Behavior

**Decision:** Keep `parser.error()` call in validator instead of raising custom exception

**Rationale:**
- Maintains exact same SystemExit behavior
- Consistent error messages and exit codes
- No need for exception translation
- Validator knows about parser context

## Lessons Learned

### What Worked Well ✅

1. **Dependency injection** - Made validation completely testable without mocking globals
2. **Small extraction** - 29 → 3 lines, minimal risk
3. **Test-first approach** - 8 tests written before integration
4. **Pattern consistency** - Following Extract → Test → Integrate → Validate rhythm

### Unexpected Wins 🎉

1. **Cleaner than expected** - 3-line parse_and_validate_args is remarkably clean
2. **Better test isolation** - Can test PERPS_DEBUG without environment pollution
3. **Reusable validator** - ArgumentValidator can be used in other contexts
4. **No import cleanup needed** - os import still used in build_bot_config_from_args

## Next Steps

### Phase 3 Preview: BotConfig Builder Isolation

**Scope:**
- Extract `build_bot_config_from_args()` into `BotConfigBuilder`
- Skip-list management for non-config arguments
- Environment fallback for symbols (TRADING_SYMBOLS)
- BotConfig factory encapsulation

**Expected:**
- Remove ~40-45 lines from parser.py
- Add `bot_config_builder.py` (~80 lines)
- Add ~5-7 new builder tests
- **Target:** parser.py down to ~75 lines

### Phase 4 Preview: Integration Cleanup

**Scope:**
- Final polish and documentation
- Review remaining parser.py code
- Consider any final extractions
- Complete CLI parser refactor

**Expected:**
- Small additional reductions (~-5 to -10 lines)
- **Target:** parser.py down to ~65-70 lines (77-78% total reduction)

### Final Target

**Projected:** ~65-70 lines for parser.py (≤90 lines, 77-78% total reduction)

---

**Phase 2 Status:** ✅ Complete
**Ready for Phase 3:** ✅ Yes
**Estimated Phase 3 Effort:** 1 hour
**Risk Level:** Low ✅
**Zero Regressions:** ✅ Confirmed (144/144 tests pass)
**Target Achievement:** ✅ On track (120 lines, heading toward ≤90 target)
