# CLI Orders Command Refactor – Phase 1 Complete

**Date:** October 2025
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Summary

Extracted argument parsing and validation into `order_args.py`, replacing the duplicated logic
in `orders.py`. The command file now focuses on orchestration while the sanitizer provides typed
payloads for preview, edit-preview, and apply-edit flows.

### Metrics
- `orders.py`: 222 → 188 lines (−34 lines, 15% reduction)
- New module `order_args.py`: 136 lines (Preview/Edit/Apply dataclasses + parser)
- Tests: 9 → 17 (added 8 parser-focused tests)
- Full CLI command suite: `pytest tests/unit/bot_v2/cli/commands` → **43 tests** (all green)

### Highlights
- `OrderArgumentsParser` centralizes:
  - Symbol validation
  - Enum parsing with `parser.error` feedback
  - Decimal/int conversion for quantity/price/stop/leverage
  - ORDER_ID:PREVIEW_ID parsing for apply-edit
- Helper functions (`_handle_preview_order`, etc.) now operate on sanitized payloads, keeping old
  behaviour while reducing duplication.
- Backward compatibility preserved – tests still interact with existing helper signatures.

## Next Steps

Phase 2 will extract the broker preview interaction and JSON output into a dedicated service so
`orders.py` can fully delegate operational logic.
