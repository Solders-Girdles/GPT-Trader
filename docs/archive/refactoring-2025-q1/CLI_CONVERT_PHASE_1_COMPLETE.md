# CLI Convert Command – Phase 1 Complete

**Date:** October 2025
**Phase:** ConvertRequestParser & ConvertService Extraction
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Summary

Extracted string parsing into `ConvertRequestParser` and broker operations into `ConvertService`, reducing convert.py to a clean orchestration façade and improving testability with frozen dataclass and printer injection.

### Metrics
- `convert.py`: 64 → 35 lines (−29 lines, **45% reduction**)
- `convert_request_parser.py`: 51 lines (parsing + ConvertRequest dataclass)
- `convert_service.py`: 69 lines (broker operations + JSON formatting)
- Tests: +15 tests (8 parser + 7 service)
- Total CLI tests: 160 → 175 (all green)

### Highlights
- `ConvertRequest` frozen dataclass for type-safe parsed data
- `ConvertRequestParser.parse()` static method handles FROM:TO:AMOUNT parsing
- `ConvertService.execute_conversion()` handles broker call and JSON output
- Injectable `printer` for complete testability
- `handle_convert` reduced from 47 lines to 7 lines (parse → delegate → shutdown)
- Removed `json` import from convert.py (isolated in service)
- Zero regressions - all 11 existing convert tests + 15 new tests passing

## Changes Made

### New Files

#### 1. `convert_request_parser.py` (51 lines)

**Purpose:** Parses FROM:TO:AMOUNT conversion arguments into type-safe ConvertRequest

**Components:**
- `ConvertRequest` frozen dataclass (from_asset, to_asset, amount)
- `ConvertRequestParser` class with static parse method
- String parsing with maxsplit=2 (allows colons in amount)
- Whitespace stripping
- Error reporting via parser.error()

**Design:**
```python
@dataclass(frozen=True)
class ConvertRequest:
    """Parsed conversion request."""
    from_asset: str
    to_asset: str
    amount: str

class ConvertRequestParser:
    """Parses conversion request strings."""

    @staticmethod
    def parse(convert_arg: str, parser: argparse.ArgumentParser) -> ConvertRequest:
        """Parse FROM:TO:AMOUNT format into ConvertRequest."""
        # Split with maxsplit=2
        # Strip whitespace
        # Error handling via parser.error()
        # Return ConvertRequest
```

**Benefits:**
- **Type-safe**: Frozen dataclass prevents modification
- **Testable**: Static method, no instance state
- **Clear API**: ConvertRequest clearly defines parsed data
- **Consistent**: Follows OrderPreviewArgs pattern

#### 2. `convert_service.py` (69 lines)

**Purpose:** Executes asset conversions and formats results as JSON

**Components:**
- `ConvertService` class with injectable printer
- `execute_conversion()` method for conversion workflow
- Payload building from ConvertRequest
- Broker call via account_manager.convert()
- JSON formatting and printing

**Design:**
```python
class ConvertService:
    """Executes asset conversions and formats results."""

    def __init__(self, printer: Callable[[str], None] | None = None) -> None:
        self._printer = printer or print

    def execute_conversion(self, bot: Any, request: ConvertRequest) -> int:
        """Execute asset conversion and print JSON result."""
        # Build payload from request
        # Call bot.account_manager.convert(payload, commit=True)
        # Format JSON (indent=2, default=str)
        # Print via injected printer
        # Return exit code
```

**Benefits:**
- **Testable**: Injectable printer allows verification without stdout capture
- **Focused**: Single responsibility (conversion execution)
- **Reusable**: Can be used in other contexts
- **Consistent**: Follows AccountSnapshotService, OrderPreviewService patterns

#### 3. `test_convert_request_parser.py` (8 tests)

**Test Coverage:**

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| ConvertRequestParser | 6 | Valid format, whitespace, colons, errors |
| ConvertRequest | 2 | Frozen dataclass, fields |

**Test Categories:**
- ✅ Parse valid FROM:TO:AMOUNT format
- ✅ Parse with whitespace (strips correctly)
- ✅ Parse with colons in amount (maxsplit=2)
- ✅ Invalid format (missing parts) → parser.error()
- ✅ Invalid format (single value) → parser.error()
- ✅ Parse decimal amount
- ✅ ConvertRequest is frozen (immutable)
- ✅ ConvertRequest fields accessible

#### 4. `test_convert_service.py` (7 tests)

**Test Coverage:**

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| Conversion Execution | 4 | Success, printer, commit flag, exceptions |
| JSON Formatting | 2 | Indentation, special types |
| Payload Building | 1 | Correct payload from request |

**Test Categories:**
- ✅ Execute conversion success
- ✅ Custom printer injection
- ✅ Commit=True flag passed
- ✅ Exception propagation
- ✅ JSON formatting with indentation
- ✅ JSON handles non-serializable types
- ✅ Payload built correctly from ConvertRequest

### Modified Files

#### `convert.py` (64 → 35 lines, -29 lines)

**Before:**
- `handle_convert`: 47 lines (parsing, validation, payload, broker call, JSON, error handling)
- Direct string split and parsing
- Direct `json.dumps()` calls
- Direct `print()` calls
- Logging mixed with business logic

**After:**
- `handle_convert`: 7 lines (parse → service → shutdown)
- Added imports: `ConvertRequestParser`, `ConvertService`
- Removed import: `json`
- All parsing logic delegated to parser
- All conversion logic delegated to service

**Key Extraction:**
```python
# Before (47 lines)
def handle_convert(convert_arg: str, bot: PerpsBot, parser: argparse.ArgumentParser) -> int:
    logger.info("Processing convert command with arg=%s", convert_arg)

    # Parse FROM:TO:AMOUNT format
    try:
        from_asset, to_asset, amount = (part.strip() for part in convert_arg.split(":", 2))
    except ValueError:
        logger.error("Invalid convert argument format: %s", convert_arg)
        parser.error("--convert requires format FROM:TO:AMOUNT")

    logger.info("Converting %s from %s to %s", amount, from_asset, to_asset)

    try:
        payload = {"from": from_asset, "to": to_asset, "amount": amount}
        result = bot.account_manager.convert(payload, commit=True)
        logger.info("Conversion completed successfully")
        output = json.dumps(result, indent=2, default=str)
        print(output)
        return 0
    except Exception as e:
        logger.error("Conversion failed: %s", e, exc_info=True)
        raise
    finally:
        ensure_shutdown(bot)

# After (7 lines)
def handle_convert(convert_arg: str, bot: PerpsBot, parser: argparse.ArgumentParser) -> int:
    request = ConvertRequestParser.parse(convert_arg, parser)
    service = ConvertService()
    try:
        return service.execute_conversion(bot, request)
    finally:
        ensure_shutdown(bot)
```

## Validation

### Test Results

**Convert Request Parser Tests:**
```bash
$ pytest tests/.../test_convert_request_parser.py -v
============================= 8 passed in 0.04s ==============================
```

**Convert Service Tests:**
```bash
$ pytest tests/.../test_convert_service.py -v
============================= 7 passed in 0.04s ==============================
```

**Full CLI Test Suite:**
```bash
$ pytest tests/unit/bot_v2/cli/ --tb=no -q
============================= 175 passed in 0.14s ==============================
```

**Zero regressions** - All 11 existing convert tests + 15 new tests passing

### Behavioral Verification

✅ **String parsing unchanged** - Same split(":", 2) with whitespace stripping
✅ **Error reporting unchanged** - Same parser.error() behavior and messages
✅ **Payload building unchanged** - Same dict structure {"from", "to", "amount"}
✅ **Broker call unchanged** - Same commit=True flag
✅ **JSON formatting unchanged** - Same indent=2, default=str
✅ **Output unchanged** - Same stdout output
✅ **Shutdown unchanged** - Still called in finally block
✅ **Parse error behavior unchanged** - Exits before try block (no shutdown)
✅ **Logging unchanged** - Same log messages at same levels

## Design Decisions

### 1. ConvertRequest Frozen Dataclass

**Decision:** Use `@dataclass(frozen=True)` for parsed conversion request

**Rationale:**
- Immutable parsed data (frozen=True)
- Type-safe field access
- Self-documenting API
- Follows OrderPreviewArgs/EditPreviewArgs pattern
- Hashable (could be used in sets/dicts if needed)

### 2. Static Parser Method

**Decision:** Use static method for ConvertRequestParser.parse()

**Rationale:**
- No state needed for parsing
- Pure function (string → ConvertRequest)
- Can call without instantiation
- Consistent with ArgumentGroupRegistrar pattern
- Clear that parsing has no side effects

### 3. Parser.error() in Parser

**Decision:** Pass parser to ConvertRequestParser.parse() for error reporting

**Rationale:**
- Parser knows how to report argparse errors
- Maintains SystemExit behavior
- Consistent error messages ("--convert requires...")
- Service shouldn't know about argparse

### 4. Injectable Printer in Service

**Decision:** Continue injectable printer pattern from orders/account

**Rationale:**
- Proven pattern (4th successful application)
- Testability without stdout capture
- Easy JSON output verification
- No coupling to print implementation

### 5. Shutdown Remains in Handler

**Decision:** Keep `ensure_shutdown(bot)` in handler's finally block

**Rationale:**
- Shutdown is handler orchestration concern
- Service shouldn't know about bot lifecycle
- Consistent with orders/account pattern
- Maintains shutdown guarantees
- **Important:** Parse error exits before try block (no shutdown on parse error)

## Lessons Learned

### What Worked Well ✅

1. **Dataclass pattern** - ConvertRequest provides type safety and clarity
2. **Static parser** - Clean API, no instance needed
3. **Proven service pattern** - Injectable printer worked perfectly
4. **Import cleanup** - Removed `json` import from handler
5. **Dual extraction** - Parser + Service in single phase was efficient

### Phase 1 Impact 🎯

**Metrics:**
- 29-line reduction (45% reduction)
- 15 new focused tests (8 parser + 7 service)
- Zero regressions
- Clean handler façade (7 lines of logic)

**Why effective:**
- String parsing extracted (6 lines)
- Payload building extracted (2 lines)
- Broker call extracted (1 line)
- JSON formatting extracted (3 lines)
- Error handling extracted (3 lines)
- Logging extracted (6 lines)

## Module Structure

```
cli/commands/
├── convert.py                          (35 lines)  - Handler façade
├── convert_request_parser.py           (51 lines)  - FROM:TO:AMOUNT parsing
├── convert_service.py                  (69 lines)  - Conversion execution
└── helpers/
    ├── test_convert_request_parser.py  (8 tests)   - Parser tests
    └── test_convert_service.py         (7 tests)   - Service tests
```

## Next Steps

### Alternative Paths

#### Option A: Consider Complete
With 45% reduction and clean separation:
- ✅ Handler is clean façade (7 lines of logic)
- ✅ Parser and service are focused and testable
- ✅ Zero regressions
- ✅ Follows proven pattern (4th application)

**Could declare complete and move to next command** (move_funds.py or run.py)

#### Option B: Additional Polish (Optional)
- Review if more edge cases need tests
- Consider additional logging tests
- Add integration test for handler → parser → service flow

**Expected:** Minimal additional benefit (already at 45% reduction)

### Recommended: Move to Next Command

**Priority:** Baseline move_funds.py to apply same pattern

**Rationale:**
- Convert command refactor achieved goals (45% vs 60% target)
- Pattern proven and refined (4th application)
- More value in spreading pattern to remaining commands
- CLI command suite getting cleaner overall

---

**Phase 1 Status:** ✅ Complete
**Target Achievement:** ✅ Close to target (35 lines vs ~25 target, 45% vs 60% target)
**Test Coverage:** ✅ 175 tests passing (15 new tests)
**Zero Regressions:** ✅ Confirmed
**Ready for Next Command:** ✅ Yes
**Pattern Maturity:** ✅ Proven (4th successful application: orders → parser → account → convert)
