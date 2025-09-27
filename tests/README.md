Test Suite Overview
====================

The GPT-Trader repository now ships a single, hermetic unit test suite. All
active coverage lives under `tests/unit/` with shared helpers in `tests/utils/`.
Historical integration, performance, and experimental suites were archived in
September 2025; consult git history if you need those references.

Suite Layout
------------
- `tests/unit/bot_v2/` – orchestration, broker adapters, risk, and CLI coverage
- `tests/unit/features/` – legacy slice checks retained for regression safety
- `tests/unit/coinbase/` – quantisation and spec validation utilities
- `tests/utils/` – deterministic broker doubles and helper fixtures reused across
  the suite

Conventions
-----------
- Markers
  - `integration`, `real_api`, `perf`, and `uses_mock_broker` remain declared in
    `pytest.ini` for continuity, but no active tests currently use them.
  - `xfail` placeholders document planned risk work in
    `tests/unit/bot_v2/features/live_trade/test_risk_backlog.py`.
- Async tests continue to rely on `anyio`; prefer injecting clocks or fixtures
  (e.g. `fake_clock`) over `time.sleep`.
- The `test-hygiene` pre-commit hook warns when modules exceed ~240 lines or use
  real sleeps.

Running the Suite
-----------------
```bash
# Full unit suite (default deselection already applied)
poetry run pytest -q

# Inspect collection counts (current: 452 collected / 445 selected / 7 deselected / 1 skipped)
poetry run pytest --collect-only -q

# Focus on orchestration coverage
poetry run pytest tests/unit/bot_v2 -q
```

Deselection & Skips
-------------------
- Deselection in `pytest.ini` keeps the historical markers (`integration`,
  `real_api`, `perf`, `performance`, `uses_mock_broker`) filtered even though no
  matching tests are present today. Retain the markers so downstream tooling
  continues to behave when those suites are reintroduced.
- The only explicit skips/xfails are the TODO placeholders in
  `tests/unit/bot_v2/features/live_trade/test_risk_backlog.py`.

Hygiene Checklist
-----------------
- Keep new tests under `tests/unit/…` alongside the code slice they cover.
- Co-locate fixtures in the nearest `conftest.py` or reuse helpers from
  `tests/utils/` when they remain deterministic.
- Update this README (and `pytest.ini`) if you add new markers or resurrect the
  integration suite.

Counts Snapshot (September 2025)
--------------------------------
- 445 selected tests (unit)
- 7 deselected by markers (integration/perf placeholders)
- 1 skipped (credential-gated API smoke)
- 4 xfails tracking risk backlog items

Use `poetry run pytest --collect-only -q` after structural changes to keep this
snapshot accurate.
