# CLI Parser Refactor – Phase 1 Complete

**Date:** October 2025
**Phase:** Argument Specs Extraction
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Summary

Extracted repetitive argument definitions into declarative `ArgumentSpec` dataclasses and
`ArgumentGroupRegistrar`, achieving massive line reduction while improving maintainability and
testability.

### Metrics
- `parser.py`: 304 → 133 lines (−171 lines, **56% reduction**)
- `argument_groups.py`: 310 new lines (declarative specs + registrar)
- Tests: +22 argument group tests (total CLI tests now 136)
- Full CLI test suite: `pytest tests/unit/bot_v2/cli/` → **136 tests** (all green)

### Highlights
- Replaced 160+ lines of repetitive `parser.add_argument()` calls with declarative `ArgumentSpec` definitions
- Eliminated 6 `_add_*` helper functions → single `ArgumentGroupRegistrar.register_all()` call
- Added comprehensive test coverage for argument specs, registrar, and specific argument properties
- Zero regressions - all 114 existing CLI tests still passing

## Changes Made

### New Files

#### 1. `argument_groups.py` (310 lines)

**Purpose:** Declarative argument specifications for all CLI arguments

**Components:**
- `ArgumentSpec` dataclass - immutable specification for a single argument
- Six argument group constants:
  - `BOT_CONFIG_ARGS` (9 arguments)
  - `ACCOUNT_ARGS` (1 argument)
  - `CONVERT_ARGS` (1 argument)
  - `MOVE_FUNDS_ARGS` (1 argument)
  - `ORDER_TOOLING_ARGS` (13 arguments)
  - `DEV_ARGS` (1 argument)
- `ArgumentGroupRegistrar` class - applies argument specs to parser

**Design:**
```python
@dataclass(frozen=True)
class ArgumentSpec:
    """Declarative specification for a single CLI argument."""
    name: str
    action: str | None = None
    type: type | None = None
    default: Any = None
    nargs: str | int | None = None
    choices: list[str] | None = None
    metavar: str | None = None
    dest: str | None = None
    help: str = ""

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add this argument to the given parser."""
        # Builds kwargs dict and calls parser.add_argument()

class ArgumentGroupRegistrar:
    """Registers argument groups to a parser."""

    @staticmethod
    def register_all(parser: argparse.ArgumentParser) -> None:
        """Register all argument groups to the parser."""
        # Registers all six argument groups
```

**Benefits:**
- **Declarative**: Arguments defined as data, not code
- **DRY**: No repetitive `add_argument` calls
- **Testable**: Can verify specs without running parser
- **Maintainable**: Easy to add/modify arguments
- **Type-safe**: Frozen dataclass with type hints

#### 2. `test_argument_groups.py` (22 tests)

**Test Coverage:**

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| ArgumentSpec | 5 | Dataclass behavior, add_to_parser() |
| Argument Groups | 6 | Count verification for each group |
| ArgumentGroupRegistrar | 7 | Registration methods |
| Argument Properties | 4 | Specific argument validation |

**Test Categories:**
- ✅ ArgumentSpec with action (store_true)
- ✅ ArgumentSpec with type and default
- ✅ ArgumentSpec with choices validation
- ✅ ArgumentSpec with custom dest
- ✅ ArgumentSpec with nargs
- ✅ Group counts (verify no arguments lost)
- ✅ Individual registrar methods
- ✅ register_all() integration
- ✅ Profile choices validation
- ✅ Symbols nargs (multi-value)
- ✅ Leverage dest mapping
- ✅ Order quantity Decimal type

### Modified Files

#### `parser.py` (304 → 133 lines, -171 lines)

**Before:**
- 6 `_add_*` helper functions (160+ lines)
- Repetitive `parser.add_argument()` calls
- Imported `Decimal` for order_quantity type

**After:**
- Single import: `ArgumentGroupRegistrar`
- One-line registration: `ArgumentGroupRegistrar.register_all(parser)`
- Removed `Decimal` import (moved to argument_groups.py)
- Removed all 6 `_add_*` helper functions

**Key Changes:**
```python
# Before (lines 16-35)
def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Perpetuals Trading Bot")

    # Core bot configuration
    _add_bot_config_args(parser)

    # Command-specific arguments
    _add_account_args(parser)
    _add_convert_args(parser)
    _add_move_funds_args(parser)
    _add_order_tooling_args(parser)
    _add_dev_args(parser)

    return parser

# After (lines 17-26)
def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Perpetuals Trading Bot")
    ArgumentGroupRegistrar.register_all(parser)
    return parser
```

**Removed Functions:**
- `_add_bot_config_args()` (56 lines)
- `_add_account_args()` (7 lines)
- `_add_convert_args()` (7 lines)
- `_add_move_funds_args()` (7 lines)
- `_add_order_tooling_args()` (69 lines)
- `_add_dev_args()` (7 lines)

**Total removed:** 153 lines of function definitions + 7 lines of function calls = 160 lines

## Validation

### Test Results

**Argument Groups Tests:**
```bash
$ pytest tests/unit/bot_v2/cli/test_argument_groups.py -v
============================= 22 passed in 0.04s ==============================
```

**Full CLI Test Suite:**
```bash
$ pytest tests/unit/bot_v2/cli/ --tb=no -q
============================= 136 passed in 0.13s ==============================
```

**Zero regressions** - All 114 existing CLI tests + 22 new tests passing

### Behavioral Verification

✅ **Parser output unchanged** - Exact same arguments registered
✅ **All argument properties preserved** - Choices, defaults, types, dest all correct
✅ **Validation unchanged** - `parse_and_validate_args()` untouched
✅ **Config building unchanged** - `build_bot_config_from_args()` untouched
✅ **Order tooling check unchanged** - `order_tooling_requested()` untouched

## Progress Summary

### Cumulative Reduction

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| parser.py | 304 lines | 133 lines | -171 lines (-56%) |
| Argument definitions | Procedural | Declarative | ✅ Improved |
| Helper functions | 6 functions | 0 functions | -6 |
| Test coverage | 114 tests | 136 tests | +22 tests |

### Module Structure

```
cli/
├── parser.py                    (133 lines) - Validation & config building
├── argument_groups.py           (310 lines) - Declarative argument specs
├── test_argument_groups.py      (22 tests)  - Argument spec tests
└── commands/
    └── ... (unchanged)
```

## Design Decisions

### 1. Frozen Dataclass for ArgumentSpec

**Decision:** Use `@dataclass(frozen=True)` for `ArgumentSpec`

**Rationale:**
- Immutable argument specifications
- Clear intent: specs are configuration, not state
- Type safety with frozen attributes
- Hashable (could be used in sets/dicts if needed)

### 2. Explicit None Defaults

**Decision:** Use `None` defaults for optional fields, build kwargs dict in `add_to_parser()`

**Rationale:**
- Clear distinction between "not provided" and "explicitly set to default"
- Allows argparse to use its own defaults
- Avoids passing unnecessary kwargs
- Cleaner argument addition

### 3. Static Methods in ArgumentGroupRegistrar

**Decision:** Use static methods instead of instance methods

**Rationale:**
- No state needed in registrar
- Clear that registration is a pure function
- Can call methods without instantiating
- Follows command pattern (execute action)

### 4. Separate Argument Groups

**Decision:** Define six separate argument group lists instead of one big list

**Rationale:**
- Logical grouping matches domain concepts
- Easy to find specific arguments
- Can register groups individually for testing
- Documentation clarity (can reference BOT_CONFIG_ARGS vs ALL_ARGS)

## Lessons Learned

### What Worked Well ✅

1. **Declarative approach** - ArgumentSpec makes arguments data, easier to reason about
2. **Test-driven** - Writing tests first ensured specs matched behavior
3. **Incremental validation** - Testing each group separately caught issues early
4. **Pattern consistency** - Same Extract → Test → Integrate → Validate rhythm

### Unexpected Wins 🎉

1. **Exceeded target** - Achieved 56% reduction vs 50% target
2. **Better test coverage** - 22 new focused tests vs 5-8 expected
3. **Clean imports** - Removed unused `Decimal` import from parser.py
4. **Zero merge conflicts** - Declarative specs live in separate module

## Next Steps

### Phase 2 Preview: Parse & Validation Service

**Scope:**
- Extract `parse_and_validate_args()` logic into `ArgumentValidator`
- Injectable logger and environment access
- Symbol validation, PERPS_DEBUG handling
- Comprehensive validation tests

**Expected:**
- Remove ~25-30 lines from parser.py
- Add `argument_validator.py` (~60 lines)
- Add ~6-8 new validator tests
- **Target:** parser.py down to ~105 lines

### Phase 3 Preview: BotConfig Builder Isolation

**Scope:**
- Extract `build_bot_config_from_args()` into `BotConfigBuilder`
- Skip-list management
- Environment fallback for symbols
- BotConfig factory encapsulation

**Expected:**
- Remove ~40-45 lines from parser.py
- Add `bot_config_builder.py` (~80 lines)
- Add ~5-7 new builder tests
- **Target:** parser.py down to ~60 lines

### Final Target

**Projected:** ~60-70 lines for parser.py (≤80 lines, 73-77% total reduction)

---

**Phase 1 Status:** ✅ Complete
**Ready for Phase 2:** ✅ Yes
**Estimated Phase 2 Effort:** 1 hour
**Risk Level:** Low ✅
**Zero Regressions:** ✅ Confirmed (136/136 tests pass)
**Target Achievement:** ✅ Exceeded (133 lines vs ≤150 target)
