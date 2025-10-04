# CLI Orders Command Refactor – Phase 2 Complete

**Date:** October 2025
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Summary

Moved broker preview execution and JSON formatting into `OrderPreviewService`, further shrinking
`orders.py` and improving testability.

### Metrics
- `orders.py`: 188 → 163 lines (−25 lines this phase, −59 overall)
- `order_preview_service.py`: 50 focused lines
- Tests: +3 service tests (total CLI command tests now 96)
- Full CLI command suite: `pytest tests/unit/bot_v2/cli/commands` → **93 tests** (all green)

### Highlights
- `OrderPreviewService.preview()` handles logging, broker invocation, and JSON printing with an
  injectable printer (supports testing).
- `_handle_preview_order` now parses args via `OrderArgumentsParser` and delegates to the service.
- Added service coverage (`tests/unit/bot_v2/cli/commands/helpers/test_order_preview_service.py`)
  ensuring payload construction, printer override, and logging behaviour.

## Next Steps

Phase 3 will extract the edit-preview flow, applying the same pattern for `edit_order_preview` so
`orders.py` continues to slim down toward the ≤120‑line target.
