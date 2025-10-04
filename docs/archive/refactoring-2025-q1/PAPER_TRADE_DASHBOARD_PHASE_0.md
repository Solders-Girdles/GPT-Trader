# Paper Trade Dashboard Refactor – Phase 0 Baseline

**Date:** October 2025
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Files & Line Counts

- `src/bot_v2/features/paper_trade/dashboard.py` — **401 lines**

## Existing Responsibilities

| Responsibility | Lines | Notes |
|----------------|-------|-------|
| CLI setup (`__init__`, refresh interval, initial equity) | 20 | Straightforward configuration |
| Screen clearing (`clear_screen`) | 4 | Platform-dependent call |
| Formatting (`format_currency`, `format_pct`) | 11 | Pure helpers |
| Metrics calculation (`calculate_metrics`) | 47 | Reads engine state, computes returns, drawdown, win rate, exposure |
| Rendering header/summary/performance/positions/trades | 131 | Mixed print/logging formatting |
| Dashboard loop (`display_once`, `display_continuous`) | 51 | Loops, handles Ctrl+C, sleeps |
| HTML report export | 139 | Builds HTML string, writes files, ensures directories |
| Utilities (figlet banner import, etc.) | 5 | Simple wrappers for optional dependencies |

## Dependencies

- `engine`: requires attributes `initial_capital`, `calculate_equity`, `cash`, `positions`, `trades`, `bot_id`, `strategy_name`, `symbols`.
- Uses standard libs: `logging`, `os`, `time`, `datetime`, `pathlib.Path`.
- `bot_v2.config.path_registry.RESULTS_DIR` for HTML output directory.

## Test Baseline

- `tests/unit/bot_v2/features/paper_trade/test_dashboard.py` — **45 tests**, all green (`pytest tests/unit/bot_v2/features/paper_trade/test_dashboard.py`).

### Coverage Highlights
- Initialization defaults and custom refresh interval.
- Formatting helpers (currency, percentage).
- Screen clearing (POSIX vs Windows).
- Metrics calculation (initial, profit, loss, no trades, edge cases).
- Rendering functions (portfolio summary, positions, performance, trades).
- `display_once`/`display_continuous` loops (with duration, KeyboardInterrupt).
- HTML report generation (file content, empty state, directory creation).

## Observed Gaps / Risks

- Metrics calculation uses simplified drawdown and win-rate (no issue but keep behaviour).
- HTML generation embeds formatting logic; no tests for error handling if writing fails.
- `format_pct` applies identical rendering regardless of sign (legacy behaviour).
- Dashboard blends data preparation, presentation, and export logic in single class.

## Extraction Candidates

1. **Formatting helpers** (`format_currency`, `format_pct`) → `dashboard/formatters.py`.
2. **Metrics assembler** (`calculate_metrics`) → `dashboard/metrics.py`.
3. **Renderer** (console printing) → `dashboard/console_renderer.py`.
4. **HTML report builder** → `dashboard/html_report.py`.
5. **Dashboard runner** (loop control) → `dashboard/loop.py`.

## Next Steps (Phase Plan)

1. Phase 1 – Formatters module (helpers, tests for formatting, colours/logging optional). Target: −20 lines.
2. Phase 2 – Metrics assembler (clean data snapshot, tested against legacy behaviour). Target: −40 lines.
3. Phase 3 – Console renderer (header/summary/performance/positions/trades printing). Target: −100 lines.
4. Phase 4 – HTML report writer (string building, file output, path prep). Target: −120 lines.
5. Phase 5 – Dashboard runner (display_once/continuous, CLI start). Target: −70 lines.
6. Phase 6 – Facade cleanup & documentation (≤200 line target).

Each phase will add dedicated unit tests mirroring existing coverage and keep API backwards compatible.
