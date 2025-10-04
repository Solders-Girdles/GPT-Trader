# CLI Parser Refactor – Phase 3 Complete

**Date:** October 2025
**Phase:** BotConfig Builder Isolation
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Summary

Extracted BotConfig building logic into `BotConfigBuilder`, achieving the largest single-phase reduction in the CLI parser refactor and reducing parser.py to a clean orchestration façade.

### Metrics
- `parser.py`: 120 → 77 lines (−43 lines this phase, −227 overall from 304 baseline)
- `bot_config_builder.py`: 95 focused lines handling config construction
- Tests: +9 builder tests (total CLI tests now 153)
- Full CLI test suite: `pytest tests/unit/bot_v2/cli/` → **153 tests** (all green)

### Highlights
- `BotConfigBuilder.build()` handles skip-list filtering, TRADING_SYMBOLS env fallback, and config factory
- Injectable `env_reader` and `config_factory` for complete testability
- `build_bot_config_from_args` reduced from 59 lines to 4 lines (just builder delegation)
- Removed `os` import from parser.py (now isolated in builder)
- Zero regressions - all 144 existing CLI tests + 9 new builder tests passing

## Changes Made

### New Files

#### 1. `bot_config_builder.py` (95 lines)

**Purpose:** Builds BotConfig instances from CLI arguments with environment fallbacks

**Components:**
- `BotConfigBuilder` class with injectable dependencies
- `SKIP_KEYS` class constant defining non-config arguments
- `build()` method for config construction
- TRADING_SYMBOLS environment variable handling
- Config override filtering and logging

**Design:**
```python
class BotConfigBuilder:
    """Builds BotConfig from CLI arguments with environment fallbacks."""

    SKIP_KEYS = {
        "profile", "account_snapshot", "convert", "move_funds",
        "preview_order", "edit_order_preview", "apply_order_edit",
        "order_side", "order_type", "order_quantity", "order_price",
        "order_stop", "order_tif", "order_client_id", "order_reduce_only",
        "order_leverage", "order_symbol",
    }

    def __init__(
        self,
        *,
        env_reader: Callable[[str, str], str | None] | None = None,
        config_factory: Any = None,
    ) -> None:
        import os
        self._env_reader = env_reader or os.getenv
        self._config_factory = config_factory or self._default_config_factory()

    def build(self, args: argparse.Namespace) -> Any:
        """Build BotConfig from parsed CLI arguments."""
        # Filter to config overrides (skip command-specific args)
        # Handle TRADING_SYMBOLS env fallback
        # Log config building
        # Return config from factory
```

**Benefits:**
- **Testable**: Injectable env_reader and config_factory for mocking
- **Encapsulated**: All config building logic in one place
- **Clear separation**: Skip-list management isolated from parser
- **Maintainable**: Easy to add new skip keys or env variables

#### 2. `test_bot_config_builder.py` (9 tests)

**Test Coverage:**

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| Config Overrides | 3 | Valid overrides, skip-list filtering, None filtering |
| TRADING_SYMBOLS | 6 | Env fallback, CLI precedence, parsing, edge cases |

**Test Categories:**
- ✅ Config overrides passed to factory
- ✅ Skip keys filtered from config overrides
- ✅ None values filtered from config overrides
- ✅ TRADING_SYMBOLS loaded from env when symbols not in CLI
- ✅ CLI symbols override TRADING_SYMBOLS env
- ✅ TRADING_SYMBOLS parsed with semicolon delimiter
- ✅ TRADING_SYMBOLS whitespace stripped
- ✅ Empty TRADING_SYMBOLS skipped
- ✅ Empty symbols list triggers env fallback

### Modified Files

#### `parser.py` (120 → 77 lines, -43 lines)

**Before:**
- `build_bot_config_from_args`: 59 lines (skip-list, filtering, env handling, logging, factory)
- Direct `os.getenv()` calls
- Skip-list defined inline
- Config override logic mixed with parser

**After:**
- `build_bot_config_from_args`: 4 lines (just builder delegation)
- Added import: `BotConfigBuilder`
- Removed import: `os`
- All config building logic delegated to builder

**Key Extraction:**
```python
# Before (59 lines)
def build_bot_config_from_args(
    args: argparse.Namespace,
    *,
    config_cls=None,
):
    if config_cls is None:
        from bot_v2.orchestration.configuration import BotConfig as _ConfigFactory
    else:
        _ConfigFactory = config_cls

    skip_keys = {
        "profile", "account_snapshot", "convert", "move_funds",
        "preview_order", "edit_order_preview", "apply_order_edit",
        # ... (15 more keys)
    }

    config_overrides = {
        key: value
        for key, value in vars(args).items()
        if value is not None and key not in skip_keys
    }

    # Handle symbols from environment if not provided via CLI
    if "symbols" not in config_overrides or not config_overrides.get("symbols"):
        env_symbols = os.getenv("TRADING_SYMBOLS", "")
        if env_symbols:
            tokens = [
                tok.strip() for tok in env_symbols.replace(";", ",").split(",") if tok.strip()
            ]
            if tokens:
                config_overrides["symbols"] = tokens
                logger.info("Loaded %d symbols from TRADING_SYMBOLS env var", len(tokens))

    logger.info("Building bot config with profile=%s, overrides=%s", args.profile, config_overrides)

    return _ConfigFactory.from_profile(args.profile, **config_overrides)

# After (4 lines)
def build_bot_config_from_args(
    args: argparse.Namespace,
    *,
    config_cls=None,
):
    builder = BotConfigBuilder(config_factory=config_cls)
    return builder.build(args)
```

## Validation

### Test Results

**BotConfig Builder Tests:**
```bash
$ pytest tests/.../test_bot_config_builder.py -v
============================= 9 passed in 0.03s ==============================
```

**Full CLI Test Suite:**
```bash
$ pytest tests/unit/bot_v2/cli/ --tb=no -q
============================= 153 passed in 0.14s ==============================
```

**Zero regressions** - All 144 existing CLI tests + 9 new builder tests passing

### Behavioral Verification

✅ **Skip-list filtering unchanged** - Same arguments excluded from config
✅ **TRADING_SYMBOLS handling unchanged** - Exact same env fallback and parsing
✅ **Config factory unchanged** - Same BotConfig.from_profile() behavior
✅ **Logging unchanged** - Same log messages and levels
✅ **Argument validation unchanged** - `parse_and_validate_args()` untouched
✅ **Order tooling check unchanged** - `order_tooling_requested()` untouched

## Progress Summary

### Cumulative Reduction

| Phase | Lines | Reduction | Cumulative |
|-------|-------|-----------|------------|
| Phase 0 (Baseline) | 304 | 0 | 0 |
| Phase 1 (Argument Specs) | 133 | -171 | -171 (56%) |
| Phase 2 (Argument Validator) | 120 | -13 | -184 (61%) |
| Phase 3 (BotConfig Builder) | 77 | -43 | -227 (75%) |

**Total Reduction:** 304 → 77 lines (-227 lines, **75% reduction**)

### Module Structure

```
cli/
├── parser.py                    (77 lines)  - Thin orchestration façade
├── argument_groups.py           (310 lines) - Declarative argument specs
├── argument_validator.py        (71 lines)  - Validation & env handling
├── bot_config_builder.py        (95 lines)  - Config construction
├── test_argument_groups.py      (22 tests)  - Argument spec tests
├── test_argument_validator.py   (8 tests)   - Validator tests
└── test_bot_config_builder.py   (9 tests)   - Builder tests
```

## Design Decisions

### 1. Skip Keys as Class Constant

**Decision:** Define `SKIP_KEYS` as class constant instead of function parameter

**Rationale:**
- Skip-list is stable configuration, not runtime data
- Centralized definition makes maintenance easier
- Can be referenced in tests or other modules
- Clear intent: these keys never change per instance

### 2. Injectable Config Factory

**Decision:** Allow config_factory injection while providing default

**Rationale:**
- Tests can inject mock factory without imports
- Maintains compatibility with existing `config_cls` parameter
- Default factory loaded lazily to avoid import cycles
- Consistent with dependency injection pattern

### 3. Combined Env Reader Signature

**Decision:** Use `env_reader(key, default)` signature matching `os.getenv`

**Rationale:**
- Direct replacement for os.getenv
- Tests can mock easily
- Supports default value pattern
- Standard Python convention

### 4. SKIP_KEYS in Builder Not Parser

**Decision:** Move skip-list definition from parser to builder

**Rationale:**
- Skip-list is config building concern, not parsing concern
- Parser shouldn't know about BotConfig internals
- Easier to maintain and extend
- Encapsulation of config-specific logic

## Lessons Learned

### What Worked Well ✅

1. **Skip-list as class constant** - Made tests clearer and maintenance easier
2. **Largest reduction yet** - 43 lines removed vs 13 in Phase 2
3. **Comprehensive env tests** - 6 tests for TRADING_SYMBOLS edge cases
4. **Pattern consistency** - Following same Extract → Test → Integrate rhythm

### Unexpected Wins 🎉

1. **Exceeded expectations** - 43-line reduction vs expected 15-20 lines
2. **Import cleanup** - Removed `os` import from parser.py
3. **Cleaner parser** - parser.py now just orchestrates 3 components
4. **Better encapsulation** - Config concerns fully isolated

### Phase 3 Impact 🎯

**Why so effective:**
- Large skip-list definition extracted (18 lines)
- Complex TRADING_SYMBOLS logic extracted (9 lines)
- Config override filtering extracted (4 lines)
- Multiple logging statements extracted (2 lines)
- Factory resolution extracted (4 lines)
- Comments and blank lines reduced

## Next Steps

### Phase 4 Preview: Parser Façade Cleanup

**Scope:**
- Review remaining parser.py code (77 lines)
- Add integration test wiring parser → validator → builder
- Final polish and documentation
- Consider any remaining extractions

**Expected:**
- Small additional reductions (~-5 to -10 lines)
- Integration test ensuring end-to-end parity
- **Target:** parser.py down to ~65-70 lines

### Final Target

**Current:** 77 lines (75% reduction, **exceeded ≤90 line target**)
**Projected:** ~65-70 lines (77-78% total reduction)

### Alternative: Declare Complete

With 75% reduction achieved (77 lines from 304 baseline) and clean separation of concerns:
- ✅ Exceeded target (≤90 lines)
- ✅ Clean orchestration façade
- ✅ All concerns separated (specs, validation, config building)
- ✅ Comprehensive test coverage (153 tests)

**CLI Parser refactor could be considered complete.** The proven Extract → Test → Integrate → Validate pattern is ready to apply to remaining CLI commands.

---

**Phase 3 Status:** ✅ Complete
**Target Achievement:** ✅ Exceeded (77 lines vs ≤90 target, 75% reduction)
**Test Coverage:** ✅ 153 tests passing (9 new builder tests)
**Zero Regressions:** ✅ Confirmed
**Ready for Phase 4:** ✅ Optional (already exceeded goals)
**Alternative:** ✅ Declare complete and move to CLI commands baseline
