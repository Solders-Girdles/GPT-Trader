# Issue 530 — Clock/TimeProvider abstraction

## Summary
- Introduce a minimal Clock/TimeProvider interface with `now_utc()`, `time()`, and `monotonic()`.
- Provide a real `SystemClock` plus a `FakeClock` that can be advanced/reset for deterministic tests.
- Wire the abstraction into a monitoring component (suggest `src/gpt_trader/monitoring/status_reporter.py`) that currently calls `time.time()` or `datetime.now()` directly.
- Add tests that demonstrate deterministic timestamps/durations while keeping production behavior unchanged.

## Acceptance Criteria
- Add the Clock/TimeProvider interface and a production-grade `SystemClock` implementation.
- Provide a `FakeClock` (with `advance()`/`set_time()` helpers) and `get_clock()/set_clock()/reset_clock()` helpers for dependency injection.
- Update at least one monitoring component to use this clock abstraction instead of direct `time`/`datetime` calls.
- Cover the utilities and the monitoring change with new unit tests that rely on the fake clock for deterministic assertions.

## Plan
1. Create `src/gpt_trader/utilities/time_provider.py` with:
   - A `TimeProvider` protocol (methods: `now_utc()`, `time()`, `monotonic()`).
   - `SystemClock` returning real UTC timestamps and monotonic time.
   - `FakeClock` storing `_time`/`_monotonic` and implementing `advance()`, `set_time()`, `set_datetime()`.
   - Module helpers `get_clock()`, `set_clock(clock)`, `reset_clock()` and a `_default_clock` singleton.
2. In `src/gpt_trader/monitoring/status_reporter.py` (or another monitoring file), import `get_clock()` and replace direct `time.time()`/`datetime.now()` calls with the clock helper. Keep the existing behavior for production (use `SystemClock` by default) and slide in the fake clock through the module helper.
3. Create `tests/unit/gpt_trader/utilities/test_time_provider.py` to cover `SystemClock`, `FakeClock`, and the module helpers (ensure reset restores the system clock).
4. Add a monitoring test file (e.g., `tests/unit/gpt_trader/monitoring/test_status_reporter_time.py`) showing how the fake clock makes reporting deterministic.
5. Add a short summary at the end of the work explaining what changed, why the fake clock exists, and how the monitoring component now uses `get_clock()`.
