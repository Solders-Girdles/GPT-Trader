# CLI Move Funds Command – Phase 0 Baseline

**Date:** October 2025
**Status:** 📋 Baseline
**Next Phase:** MoveFundsRequestParser & MoveFundsService Extraction

## Current State

### File Metrics
- **move_funds.py**: 64 lines
- **test_move_funds.py**: 216 lines, 11 tests
- **Single function**: `handle_move_funds()`
- **Test coverage**: Comprehensive (success, errors, edge cases, shutdown)

### Responsibilities Analysis

#### `handle_move_funds()` (64 lines)
Located in `src/bot_v2/cli/commands/move_funds.py:17-64`

**Current Responsibilities:**
1. **String Parsing** (lines 32-36)
   - Split "FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT" format with maxsplit=2
   - Strip whitespace from parts
   - Error handling with parser.error()

2. **Payload Building** (line 47)
   - Create fund movement dict: `{"from_portfolio": ..., "to_portfolio": ..., "amount": ...}`

3. **Broker Call** (line 50)
   - Call `bot.account_manager.move_funds(payload)`
   - **Note:** No commit flag (unlike convert)

4. **JSON Formatting** (lines 54-56)
   - Format result as JSON with `indent=2`
   - Handle non-serializable types with `default=str`
   - Print to stdout

5. **Error Handling** (lines 34-36, 59-61)
   - ValueError on invalid format → parser.error()
   - Exception during fund movement → log and re-raise

6. **Shutdown Management** (line 63)
   - Call `ensure_shutdown(bot)` in finally block

7. **Logging** (lines 29, 35, 38-43, 52, 60)
   - Info: "Processing move-funds command with arg=..."
   - Error: "Invalid move-funds argument format: ..."
   - Info: "Moving %s from portfolio %s to portfolio %s"
   - Info: "Fund movement completed successfully"
   - Error: "Fund movement failed: ..." with exc_info

### Code Structure

```python
def handle_move_funds(move_arg: str, bot: PerpsBot, parser: argparse.ArgumentParser) -> int:
    logger.info("Processing move-funds command with arg=%s", move_arg)

    # String parsing (5 lines)
    try:
        from_uuid, to_uuid, amount = (part.strip() for part in move_arg.split(":", 2))
    except ValueError:
        logger.error("Invalid move-funds argument format: %s", move_arg)
        parser.error("--move-funds requires format FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT")

    logger.info("Moving %s from portfolio %s to portfolio %s", amount, from_uuid, to_uuid)

    try:
        # Payload building (1 line)
        payload = {"from_portfolio": from_uuid, "to_portfolio": to_uuid, "amount": amount}

        # Broker call (1 line, no commit flag)
        result = bot.account_manager.move_funds(payload)

        logger.info("Fund movement completed successfully")

        # JSON formatting and output (3 lines)
        output = json.dumps(result, indent=2, default=str)
        print(output)

        return 0
    except Exception as e:
        # Error handling (2 lines)
        logger.error("Fund movement failed: %s", e, exc_info=True)
        raise
    finally:
        # Shutdown (1 line)
        ensure_shutdown(bot)
```

### Test Coverage Analysis

#### Existing Tests (11 tests)

**Happy Path (2 tests):**
- ✅ Successful fund movement
- ✅ JSON output validation

**Parse Validation (4 tests):**
- ✅ Invalid format (missing parts)
- ✅ Invalid format (single value)
- ✅ Whitespace handling (strips correctly)
- ✅ Colons in amount (maxsplit=2 behavior)

**Edge Cases (2 tests):**
- ✅ Decimal amount handling
- ✅ UUID-formatted portfolios

**Error Cases (1 test):**
- ✅ Fund movement exception (raised and logged)

**Shutdown Behavior (2 tests):**
- ✅ Shutdown called on success
- ✅ Shutdown called on error

### Comparison to Convert Command

**Similarities:**
- Same parsing pattern (FROM:TO:AMOUNT with maxsplit=2)
- Same JSON formatting (indent=2, default=str)
- Same shutdown pattern (finally block)
- Same error handling pattern
- Same logging pattern

**Differences:**
- **No commit flag** - `move_funds(payload)` vs `convert(payload, commit=True)`
- **Different payload keys** - "from_portfolio", "to_portfolio" vs "from", "to"
- **Different variable names** - from_uuid, to_uuid vs from_asset, to_asset
- **UUID portfolios** - Tests include UUID handling

### Test Coverage Gaps

**Current gaps identified:**
1. **No isolated parser tests** - Parsing logic mixed with handler
2. **No isolated service tests** - Broker call logic coupled to handler
3. **No printer injection** - Tests rely on capsys for output verification
4. **No separate JSON formatting tests** - Formatting logic coupled to handler

## Refactor Plan

### Phase 1: MoveFundsRequestParser & MoveFundsService Extraction

**Goal:** Extract string parsing into MoveFundsRequestParser and broker operations into MoveFundsService

#### Target Structure

```python
# move_funds_request_parser.py (new)
@dataclass(frozen=True)
class MoveFundsRequest:
    from_portfolio: str
    to_portfolio: str
    amount: str

class MoveFundsRequestParser:
    @staticmethod
    def parse(move_arg: str, parser: argparse.ArgumentParser) -> MoveFundsRequest:
        """Parse FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT format into MoveFundsRequest."""
        # Split and validate
        # Strip whitespace
        # Return MoveFundsRequest
```

```python
# move_funds_service.py (new)
class MoveFundsService:
    def __init__(self, printer: Callable[[str], None] | None = None):
        self._printer = printer or print

    def execute_fund_movement(self, bot: PerpsBot, request: MoveFundsRequest) -> int:
        """Execute fund movement and print JSON result."""
        # Build payload
        # Call broker (NO commit flag)
        # Format JSON
        # Print
        # Return exit code
```

```python
# move_funds.py (modified)
def handle_move_funds(move_arg: str, bot: PerpsBot, parser: argparse.ArgumentParser) -> int:
    request = MoveFundsRequestParser.parse(move_arg, parser)
    service = MoveFundsService()
    try:
        return service.execute_fund_movement(bot, request)
    finally:
        ensure_shutdown(bot)
```

#### Extraction Details

**MoveFundsRequestParser (~20 lines):**
- Parse FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT with split(":", 2)
- Strip whitespace from parts
- Raise ValueError → parser.error()
- Return MoveFundsRequest dataclass

**MoveFundsService (~40 lines):**
- Accept MoveFundsRequest
- Build payload dict (from_portfolio, to_portfolio, amount)
- Call bot.account_manager.move_funds() (NO commit flag)
- Format JSON (indent=2, default=str)
- Print via injectable printer
- Logging (info + error)

**Handler (~20 lines):**
- Call parser to get MoveFundsRequest
- Instantiate service
- Call service.execute_fund_movement() in try block
- Shutdown in finally block

**New Tests (~8-10 tests):**

MoveFundsRequestParser tests (~4-5):
- ✅ Valid format parsed correctly
- ✅ Whitespace stripped
- ✅ Maxsplit=2 (colons in amount)
- ✅ Invalid format raises error via parser
- ✅ UUID portfolio format handled

MoveFundsService tests (~4-5):
- ✅ Execute fund movement success
- ✅ Printer override (inject mock printer)
- ✅ JSON formatting verification
- ✅ Fund movement exception handling
- ✅ No commit flag (unlike convert)

### Expected Outcomes

#### Metrics
- **move_funds.py**: 64 → ~25 lines (**~60% reduction**)
- **move_funds_request_parser.py**: ~30 lines (new, includes dataclass)
- **move_funds_service.py**: ~45 lines (new)
- **New tests**: ~8-10 tests (parser + service)
- **Total CLI tests**: 175 → ~185 tests

#### Benefits
- ✅ **Testability**: Parser and service tested in isolation
- ✅ **Separation of Concerns**: Handler → Parser → Service clean flow
- ✅ **Reusability**: MoveFundsRequest dataclass can be used elsewhere
- ✅ **Consistency**: Follows convert/orders/account patterns

### Design Decisions

#### 1. MoveFundsRequest Dataclass

**Decision:** Use frozen dataclass for parsed fund movement request

**Rationale:**
- Type-safe representation of parsed data
- Immutable (frozen=True)
- Self-documenting (from_portfolio, to_portfolio, amount fields)
- Follows ConvertRequest/OrderPreviewArgs pattern

#### 2. No Commit Flag

**Decision:** Do NOT pass commit flag to move_funds() (unlike convert)

**Rationale:**
- Current implementation doesn't use commit flag
- Preserve existing behavior exactly
- move_funds() API may not support commit parameter
- Tests don't verify commit flag for move_funds

#### 3. Injectable Printer in Service

**Decision:** Continue injectable printer pattern from convert/account

**Rationale:**
- Proven pattern (5th application)
- Testability without stdout capture
- Easy JSON output verification
- No coupling to print implementation

#### 4. Shutdown Remains in Handler

**Decision:** Keep ensure_shutdown(bot) in handler's finally block

**Rationale:**
- Shutdown is handler orchestration concern
- Service shouldn't know about bot lifecycle
- Consistent with all previous patterns
- Maintains shutdown guarantees

## Risk Assessment

### Low Risk Factors ✅
- Small, focused function (64 lines)
- Comprehensive test coverage (11 tests)
- Nearly identical to convert.py (proven pattern)
- Clear separation of concerns

### Considerations
- Shutdown must remain in handler's finally block
- Parse error (parser.error) must exit before try block (no shutdown on parse error)
- JSON formatting (indent=2, default=str) must be exact
- **No commit flag** - different from convert
- All logging messages must be preserved
- UUID portfolio format must be handled

## Next Steps

1. **Create MoveFundsRequest Dataclass + Parser** (~30 lines)
   - MoveFundsRequest frozen dataclass
   - MoveFundsRequestParser.parse() static method
   - FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT parsing with maxsplit=2
   - Whitespace stripping
   - Error reporting via parser.error()

2. **Create MoveFundsService** (~45 lines)
   - execute_fund_movement(bot, request) method
   - Payload building from MoveFundsRequest
   - Broker call (account_manager.move_funds, NO commit flag)
   - JSON formatting and printing
   - Injectable printer

3. **Add Parser + Service Tests** (~8-10 tests)
   - MoveFundsRequestParser tests (4-5)
   - MoveFundsService tests (4-5)

4. **Integrate into Handler** (~25 lines)
   - Call parser to get MoveFundsRequest
   - Instantiate service
   - Call service in try block
   - Keep shutdown in finally
   - Update imports

5. **Validate**
   - Run all 11 existing move_funds tests (should pass)
   - Run 8-10 new parser/service tests (should pass)
   - Verify zero regressions in CLI suite

---

**Phase 0 Status:** ✅ Baseline Complete
**Current Lines:** 64
**Target Lines:** ~25 (60% reduction)
**Pattern:** Extract → Test → Integrate → Validate (5th application)
**Risk Level:** Low ✅ (nearly identical to convert.py)
**Ready for Phase 1:** ✅ Yes
