# CLI Orders Command Refactor – Phase 3 Complete

**Date:** October 2025
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Summary

Extracted edit preview and apply edit workflows into `EditPreviewService`, achieving the largest
single-phase reduction yet and simplifying `orders.py` to a clean orchestration façade.

### Metrics
- `orders.py`: 163 → 109 lines (−54 lines this phase, −113 overall from 222 baseline)
- `edit_preview_service.py`: 72 focused lines handling both edit preview and apply edit flows
- Tests: +5 service tests (total CLI command tests now 98)
- Full CLI command suite: `pytest tests/unit/bot_v2/cli/commands` → **98 tests** (all green)

### Highlights
- `EditPreviewService.edit_preview()` handles logging, broker edit preview call, and JSON printing
- `EditPreviewService.apply_edit()` handles logging, broker apply edit call, and JSON printing
- Both methods use injectable printer for testability
- `_handle_edit_order_preview` and `_handle_apply_order_edit` now just parse args and delegate
- Removed unused imports: `json`, `asdict`, `Decimal`, `OrderSide`, `OrderType`, `TimeInForce`
- Added comprehensive service coverage in `test_edit_preview_service.py`:
  - Edit preview success with payload verification
  - Edit preview with printer override
  - Edit preview logging verification
  - Apply edit success with broker call verification
  - Apply edit with printer override

## Changes Made

### New Files

#### 1. `edit_preview_service.py` (72 lines)

**Purpose:** Handles edit order preview and apply edit CLI workflows

**Components:**
- `EditPreviewService` class
  - `edit_preview()`: Previews order edits
  - `apply_edit()`: Applies previewed edits
  - Injectable printer for testing
  - Logging for all operations

**Design:**
```python
class EditPreviewService:
    def __init__(self, printer: Callable[[str], None] | None = None) -> None:
        self._printer = printer or print
        self._logger = logging.getLogger(__name__)

    def edit_preview(self, bot: Any, args: EditPreviewArgs) -> int:
        """Execute edit order preview flow and print JSON response."""
        # Logging, broker call, JSON output

    def apply_edit(self, bot: Any, args: ApplyEditArgs) -> int:
        """Execute apply order edit flow and print JSON response."""
        # Logging, broker call, JSON output
```

#### 2. `test_edit_preview_service.py` (5 tests)

**Test Coverage:**
- Edit preview success (verifies broker call, payload, JSON output)
- Edit preview with custom printer override
- Edit preview logging (info + debug)
- Apply edit success (verifies broker call, JSON output)
- Apply edit with custom printer override

### Modified Files

#### `orders.py` (163 → 109 lines, -54 lines)

**Before:**
- `_handle_edit_order_preview`: 51 lines (logging, broker call, JSON formatting)
- `_handle_apply_order_edit`: 28 lines (logging, broker call, JSON formatting)
- Imports: `json`, `asdict`, `Decimal`, `OrderSide`, `OrderType`, `TimeInForce`

**After:**
- `_handle_edit_order_preview`: 3 lines (parse → delegate)
- `_handle_apply_order_edit`: 3 lines (parse → delegate)
- Removed 6 unused imports
- Added 1 import: `EditPreviewService`

**Key Extraction:**
```python
# Before (51 lines)
def _handle_edit_order_preview(...):
    parsed = OrderArgumentsParser.parse_edit_preview(args, parser)
    logger.info(...)
    logger.debug(...)
    preview = bot.broker.edit_order_preview(...)
    logger.info(...)
    output = json.dumps(preview, indent=2, default=str)
    print(output)
    return 0

# After (3 lines)
def _handle_edit_order_preview(...):
    parsed = OrderArgumentsParser.parse_edit_preview(args, parser)
    service = EditPreviewService()
    return service.edit_preview(bot, parsed)
```

## Validation

### Test Results

**Edit Preview Service Tests:**
```bash
$ pytest tests/.../test_edit_preview_service.py -v
============================= 5 passed in 0.03s ==============================
```

**Full CLI Command Suite:**
```bash
$ pytest tests/unit/bot_v2/cli/commands/ --tb=no -q
============================= 98 passed in 0.10s ==============================
```

**Zero regressions** - All 98 tests passing

## Progress Summary

### Cumulative Reduction

| Phase | Lines | Reduction | Cumulative |
|-------|-------|-----------|------------|
| Phase 0 (Baseline) | 222 | 0 | 0 |
| Phase 1 (Argument Parser) | 188 | -34 | -34 (15%) |
| Phase 2 (Preview Service) | 163 | -25 | -59 (27%) |
| Phase 3 (Edit Preview Service) | 109 | -54 | -113 (51%) |

**Total Reduction:** 222 → 109 lines (-113 lines, **51% reduction**)

### Module Structure

```
cli/commands/
├── orders.py                    (109 lines) - Clean orchestration façade
├── edit_preview_service.py      (72 lines)  - Edit preview & apply workflows
├── order_preview_service.py     (50 lines)  - Order preview workflow
├── order_args.py                (120+ lines) - Argument parsing & validation
└── helpers/
    ├── test_edit_preview_service.py   (5 tests)
    ├── test_order_preview_service.py  (3 tests)
    └── test_order_args_parser.py      (8 tests)
```

## Design Decisions

### 1. Single Service for Edit Preview + Apply

**Decision:** Combine edit preview and apply edit in `EditPreviewService`

**Rationale:**
- Both workflows are related (apply depends on preview)
- Shared printer and logger
- Consistent with user's mental model (edit workflow)
- Avoids over-fragmentation

### 2. Injectable Printer Pattern

**Decision:** Continue injectable printer pattern from Phase 2

**Rationale:**
- Testability without capturing stdout
- Consistent with `OrderPreviewService`
- Easy to verify JSON output in tests
- No coupling to print implementation

### 3. Minimal Service Methods

**Decision:** Keep service methods simple (just logging + broker call + output)

**Rationale:**
- Single responsibility
- No business logic in service
- All validation done in parser (Phase 1)
- Easy to test and maintain

## Lessons Learned

### What Worked Well ✅

1. **Pattern consistency** - Following OrderPreviewService pattern made implementation trivial
2. **Injectable printer** - Testing was straightforward with custom printer
3. **Import cleanup** - Removing 6 unused imports improved clarity
4. **Dataclass args** - EditPreviewArgs extending PreviewOrderArgs provided type safety

### Phase 3 Impact 🎯

**Largest reduction yet:** -54 lines in single phase (vs -34 Phase 1, -25 Phase 2)

**Why so effective:**
- Two workflows extracted (edit preview + apply edit)
- Heavy JSON formatting logic moved out
- Multiple imports cleaned up
- Delegation pattern proven and refined

## Next Steps

### Phase 4 Preview (Optional): Final Polish

**Potential improvements:**
- Review remaining code in `orders.py` (109 lines)
- Consider any final extractions
- Code cleanup and documentation

**Expected:**
- Small additional reductions (~-10 to -15 lines)
- Final polish
- **Target:** ~95-100 lines (≤100 line stretch goal)

**Note:** At 109 lines (51% reduction), orders.py is already very clean. Phase 4 would be
optional polish rather than necessary refactoring.

### Alternative: Move to Next Subsystem

With 51% reduction achieved and clean separation of concerns, the Orders CLI refactor could be
considered complete. The proven Extract → Test → Integrate → Validate pattern is ready to apply
to other CLI commands or subsystems.

---

**Phase 3 Status:** ✅ Complete
**Target Achievement:** ✅ Exceeded (109 lines vs ~120-130 target)
**Test Coverage:** ✅ 98 tests passing (5 new service tests)
**Zero Regressions:** ✅ Confirmed
**Ready for Phase 4:** ✅ Optional (already exceeded goals)
