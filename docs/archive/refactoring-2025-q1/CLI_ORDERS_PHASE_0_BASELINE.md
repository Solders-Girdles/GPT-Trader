# CLI Orders Command Refactor – Phase 0 Baseline

**Date:** October 2025
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## File Overview

- `src/bot_v2/cli/commands/orders.py` — **222 lines**
  - Single module containing argument validation, preview execution, edit preview, apply edit, and output formatting.

## Current Responsibilities

| Section | Lines | Description |
|---------|-------|-------------|
| Entry (`handle_order_tooling`) | 55 | Parses flags, validates presence of symbol, dispatches to helper, handles exceptions, ensures shutdown |
| Preview helper (`_handle_preview_order`) | 68 | Validates required args, parses enums/decimals, calls broker preview, prints JSON |
| Edit preview helper (`_handle_edit_order_preview`) | 80 | Similar parsing/validation, calls broker edit preview, prints JSON |
| Apply edit helper (`_handle_apply_order_edit`) | 60 | Parses parameters, calls broker edit, prints result or error |
| Utility `_build_order_payload` | 30 | Shared payload assembly from args |
| Logger usage | Throughout | Manual logging strings |

## Dependencies

- `argparse` Namespace from CLI parser.
- `PerpsBot` (expects `.broker` with `preview_order`, `edit_order_preview`, `edit_order` methods).
- `OrderSide`, `OrderType`, `TimeInForce` enums; `Decimal` for numeric parsing.
- Shutdown handler to ensure bot is stopped cleanly.

## Tests Baseline

- `tests/unit/bot_v2/cli/commands/test_orders.py` — **9 tests**, all passing.
  - Preview success/failure, edit preview success/failure, apply edit success/failure, validation error paths.

## Pain Points

1. **Validation Duplication:** Required flag checks repeated across helper functions.
2. **Parsing Logic Scattered:** Enum/Decimal parsing duplicated for preview and edit flows.
3. **Formatting Inline:** JSON printing handled inline; harder to adjust output formats/tests.
4. **Tightly Coupled Helpers:** Private functions rely directly on `argparse.Namespace`, making unit testing richer logic awkward.
5. **Error Handling Mixed with Business Logic:** `parser.error` sprinkled throughout; limited reuse outside CLI.

## Extraction Candidates

1. **Argument Sanitizer** — builds a typed payload from `Namespace`, handles shared validation.
2. **Preview Service** — encapsulates broker preview calls and JSON serialization.
3. **Edit Service** — handles edit preview and apply edit flows with consistent logging/formatting.
4. **Output Printer** — formats dictionaries for console output (JSON/pretty). [Optional if Phase 1 scope needs simpler start.]
5. **Command Dispatcher** — small coordinator using above components to mirror `handle_order_tooling`.

## Proposed Phase Plan

1. **Phase 1 – Input Parsing & Validation Module**
   - Extract flag validation and enum/decimal parsing into `orders_arg_parser.py` (e.g., `OrderArgsSanitizer`).
   - Add focused tests for parsing combinations (side/type/tif, price/stop, reduce_only, leverage).
   - Target reduction: ~40 lines in main file.

2. **Phase 2 – Preview Service Extraction**
   - Create `order_preview_service.py` handling preview request & JSON formatting.
   - Inject service into command handler (supports dependency override in tests).
   - Target reduction: ~50 lines.

3. **Phase 3 – Edit Preview Service**
   - Extract `_handle_edit_order_preview` logic into dedicated service method.
   - Shared payload builder reused from Phase 1.
   - Target reduction: ~60 lines.

4. **Phase 4 – Apply Edit Service**
   - Encapsulate `edit_order` call and success/error output.
   - Provide consistent logging & error mapping.
   - Target reduction: ~40 lines.

5. **Phase 5 – Handler Facade Cleanup**
   - `handle_order_tooling` becomes thin dispatcher composing sanitizer + services + shutdown.
   - Aim for ≤120 lines in the command module.

6. **Phase 6 – Documentation & Final Validation**
   - Add architecture doc summarizing components and tests.
   - Run full CLI suite to ensure no regressions.

Each phase will maintain backwards compatibility with existing CLI behaviour and update tests accordingly (expect total to grow from 9 to ~40 targeted tests).
