# CLI Convert Command – Phase 0 Baseline

**Date:** October 2025
**Status:** 📋 Baseline
**Next Phase:** ConvertRequestParser & ConvertService Extraction

## Current State

### File Metrics
- **convert.py**: 64 lines
- **test_convert.py**: 198 lines, 11 tests
- **Single function**: `handle_convert()`
- **Test coverage**: Comprehensive (success, errors, edge cases, shutdown)

### Responsibilities Analysis

#### `handle_convert()` (64 lines)
Located in `src/bot_v2/cli/commands/convert.py:17-64`

**Current Responsibilities:**
1. **String Parsing** (lines 32-36)
   - Split "FROM:TO:AMOUNT" format with maxsplit=2
   - Strip whitespace from parts
   - Error handling with parser.error()

2. **Payload Building** (lines 46-47)
   - Create conversion dict: `{"from": ..., "to": ..., "amount": ...}`

3. **Broker Call** (line 50)
   - Call `bot.account_manager.convert(payload, commit=True)`

4. **JSON Formatting** (lines 54-56)
   - Format result as JSON with `indent=2`
   - Handle non-serializable types with `default=str`
   - Print to stdout

5. **Error Handling** (lines 34-36, 59-61)
   - ValueError on invalid format → parser.error()
   - Exception during conversion → log and re-raise

6. **Shutdown Management** (line 63)
   - Call `ensure_shutdown(bot)` in finally block

7. **Logging** (lines 29, 35, 38-43, 52, 60)
   - Info: "Processing convert command with arg=..."
   - Error: "Invalid convert argument format: ..."
   - Info: "Converting %s from %s to %s"
   - Info: "Conversion completed successfully"
   - Error: "Conversion failed: ..." with exc_info

### Code Structure

```python
def handle_convert(convert_arg: str, bot: PerpsBot, parser: argparse.ArgumentParser) -> int:
    logger.info("Processing convert command with arg=%s", convert_arg)

    # String parsing (5 lines)
    try:
        from_asset, to_asset, amount = (part.strip() for part in convert_arg.split(":", 2))
    except ValueError:
        logger.error("Invalid convert argument format: %s", convert_arg)
        parser.error("--convert requires format FROM:TO:AMOUNT")

    logger.info("Converting %s from %s to %s", amount, from_asset, to_asset)

    try:
        # Payload building (2 lines)
        payload = {"from": from_asset, "to": to_asset, "amount": amount}

        # Broker call (1 line)
        result = bot.account_manager.convert(payload, commit=True)

        logger.info("Conversion completed successfully")

        # JSON formatting and output (3 lines)
        output = json.dumps(result, indent=2, default=str)
        print(output)

        return 0
    except Exception as e:
        # Error handling (2 lines)
        logger.error("Conversion failed: %s", e, exc_info=True)
        raise
    finally:
        # Shutdown (1 line)
        ensure_shutdown(bot)
```

### Test Coverage Analysis

#### Existing Tests (11 tests)

**Happy Path (3 tests):**
- ✅ Successful conversion
- ✅ JSON output validation
- ✅ Decimal amount handling

**Parse Validation (4 tests):**
- ✅ Invalid format (missing parts)
- ✅ Invalid format (single value)
- ✅ Whitespace handling (strips correctly)
- ✅ Colons in amount (maxsplit=2 behavior)

**Error Cases (1 test):**
- ✅ Conversion exception (raised and logged)

**Broker Integration (1 test):**
- ✅ Commit flag (commit=True passed)

**Shutdown Behavior (2 tests):**
- ✅ Shutdown called on success
- ✅ Shutdown NOT called on parse error (exits before try block)

### Test Coverage Gaps

**Current gaps identified:**
1. **No isolated parser tests** - Parsing logic mixed with handler
2. **No isolated service tests** - Broker call logic coupled to handler
3. **No printer injection** - Tests rely on capsys for output verification
4. **No separate JSON formatting tests** - Formatting logic coupled to handler

## Refactor Plan

### Phase 1: ConvertRequestParser & ConvertService Extraction

**Goal:** Extract string parsing into ConvertRequestParser and broker operations into ConvertService

#### Target Structure

```python
# convert_request_parser.py (new)
@dataclass(frozen=True)
class ConvertRequest:
    from_asset: str
    to_asset: str
    amount: str

class ConvertRequestParser:
    @staticmethod
    def parse(convert_arg: str, parser: argparse.ArgumentParser) -> ConvertRequest:
        """Parse FROM:TO:AMOUNT format into ConvertRequest."""
        # Split and validate
        # Strip whitespace
        # Return ConvertRequest
```

```python
# convert_service.py (new)
class ConvertService:
    def __init__(self, printer: Callable[[str], None] | None = None):
        self._printer = printer or print

    def execute_conversion(self, bot: PerpsBot, request: ConvertRequest) -> int:
        """Execute conversion and print JSON result."""
        # Build payload
        # Call broker
        # Format JSON
        # Print
        # Return exit code
```

```python
# convert.py (modified)
def handle_convert(convert_arg: str, bot: PerpsBot, parser: argparse.ArgumentParser) -> int:
    request = ConvertRequestParser.parse(convert_arg, parser)
    service = ConvertService()
    try:
        return service.execute_conversion(bot, request)
    finally:
        ensure_shutdown(bot)
```

#### Extraction Details

**ConvertRequestParser (~20 lines):**
- Parse FROM:TO:AMOUNT with split(":", 2)
- Strip whitespace from parts
- Raise ValueError → parser.error()
- Return ConvertRequest dataclass

**ConvertService (~40 lines):**
- Accept ConvertRequest
- Build payload dict
- Call bot.account_manager.convert()
- Format JSON (indent=2, default=str)
- Print via injectable printer
- Logging (info + error)

**Handler (~20 lines):**
- Call parser to get ConvertRequest
- Instantiate service
- Call service.execute_conversion() in try block
- Shutdown in finally block

**New Tests (~8-10 tests):**

ConvertRequestParser tests (~4-5):
- ✅ Valid format parsed correctly
- ✅ Whitespace stripped
- ✅ Maxsplit=2 (colons in amount)
- ✅ Invalid format raises error via parser
- ✅ Missing parts raises error

ConvertService tests (~4-5):
- ✅ Execute conversion success
- ✅ Printer override (inject mock printer)
- ✅ JSON formatting verification
- ✅ Conversion exception handling
- ✅ Commit=True flag passed

### Expected Outcomes

#### Metrics
- **convert.py**: 64 → ~25 lines (**~60% reduction**)
- **convert_request_parser.py**: ~30 lines (new, includes dataclass)
- **convert_service.py**: ~45 lines (new)
- **New tests**: ~8-10 tests (parser + service)
- **Total CLI tests**: 160 → ~170 tests

#### Benefits
- ✅ **Testability**: Parser and service tested in isolation
- ✅ **Separation of Concerns**: Handler → Parser → Service clean flow
- ✅ **Reusability**: ConvertRequest dataclass can be used elsewhere
- ✅ **Consistency**: Follows orders/account patterns (parser + service)

### Design Decisions

#### 1. ConvertRequest Dataclass

**Decision:** Use frozen dataclass for parsed conversion request

**Rationale:**
- Type-safe representation of parsed data
- Immutable (frozen=True)
- Self-documenting (from_asset, to_asset, amount fields)
- Can be validated independently
- Follows OrderPreviewArgs/EditPreviewArgs pattern

#### 2. Static Parser Method

**Decision:** Use static method for parsing (no instance needed)

**Rationale:**
- No state required for parsing
- Pure function (arg → ConvertRequest)
- Consistent with ArgumentGroupRegistrar pattern
- Can be called without instantiation

#### 3. Injectable Printer in Service

**Decision:** Continue injectable printer pattern from orders/account

**Rationale:**
- Proven pattern (3 successful applications)
- Testability without stdout capture
- Easy JSON output verification
- No coupling to print implementation

#### 4. Shutdown Remains in Handler

**Decision:** Keep ensure_shutdown(bot) in handler's finally block

**Rationale:**
- Shutdown is handler orchestration concern
- Service shouldn't know about bot lifecycle
- Consistent with orders/account pattern
- Maintains shutdown guarantees

#### 5. Parser.error() in Parser

**Decision:** Pass parser to ConvertRequestParser.parse() for error reporting

**Rationale:**
- Parser knows how to report errors to argparse
- Maintains SystemExit behavior
- Consistent error messages
- Parser has context for error reporting

## Risk Assessment

### Low Risk Factors ✅
- Small, focused function (64 lines)
- Comprehensive test coverage (11 tests)
- Clear separation of concerns (parse → build → call → format)
- Proven extraction pattern (orders + account)

### Considerations
- Shutdown must remain in handler's finally block
- Parse error (parser.error) must exit before try block (no shutdown on parse error)
- JSON formatting (indent=2, default=str) must be exact
- commit=True flag must be preserved
- All logging messages must be preserved

## Next Steps

1. **Create ConvertRequest Dataclass + Parser** (~30 lines)
   - ConvertRequest frozen dataclass
   - ConvertRequestParser.parse() static method
   - FROM:TO:AMOUNT parsing with maxsplit=2
   - Whitespace stripping
   - Error reporting via parser.error()

2. **Create ConvertService** (~45 lines)
   - execute_conversion(bot, request) method
   - Payload building from ConvertRequest
   - Broker call (account_manager.convert)
   - JSON formatting and printing
   - Injectable printer

3. **Add Parser + Service Tests** (~8-10 tests)
   - ConvertRequestParser tests (4-5)
   - ConvertService tests (4-5)

4. **Integrate into Handler** (~25 lines)
   - Call parser to get ConvertRequest
   - Instantiate service
   - Call service in try block
   - Keep shutdown in finally
   - Update imports

5. **Validate**
   - Run all 11 existing convert tests (should pass)
   - Run 8-10 new parser/service tests (should pass)
   - Verify zero regressions in CLI suite

---

**Phase 0 Status:** ✅ Baseline Complete
**Current Lines:** 64
**Target Lines:** ~25 (60% reduction)
**Pattern:** Extract → Test → Integrate → Validate
**Risk Level:** Low ✅
**Ready for Phase 1:** ✅ Yes
