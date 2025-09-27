Test Suite Overview

This project organizes tests into unit and integration layers and uses explicit pytest markers to keep intent clear and behavior realistic.

- Unit: Fast, isolated tests under `tests/unit/`. No network or filesystem side‑effects. External services are mocked at the transport boundary.
- Integration: Slice and end‑to‑end tests under `tests/integration/`. These exercise CLI flows, adapters, and orchestration with in‑memory transports and fixtures.

Conventions

- Markers:
  - `@pytest.mark.integration` for integration tests.
  - `@pytest.mark.perf` or `@pytest.mark.performance` for opt‑in performance checks.
- Async tests: Use `@pytest.mark.asyncio` — the suite maps these to anyio so no extra plugin is required.
- Time control: Prefer injecting a clock (e.g., `_now()` helper) and patch that in tests instead of patching `datetime` directly. The shared `fake_clock` fixture patches `time.time`, `time.sleep`, and `asyncio.sleep` while allowing manual advancement.
- Hygiene: The `test-hygiene` pre-commit hook (`scripts/ci/check_test_hygiene.py`) warns when a test module grows beyond ~240 lines or calls `time.sleep` without the `fake_clock` fixture.
- Skips/Xfails: Place skip/xfail markers in the test file with a clear reason. Avoid centralized skip rules in `conftest.py`.

Running

- All tests: `pytest` (see pytest.ini for defaults)
- Only unit tests: `pytest tests/unit -q`
- Only integration tests: `pytest -m integration -q`

Notes

- Placeholder or demo scripts that are not true tests are archived under `archived/tests/` and excluded from collection.

Deprecations and Consolidation

- Week 1–3 prototype integration tests have been moved to `archived/tests/integration/bot_v2/`.
- Early bot_v2 adapter/unit experiments formerly under `tests/bot_v2/` have either been:
  - moved into `tests/unit/bot_v2/...` (kept as active coverage), or
  - archived under `archived/tests/bot_v2/...` when redundant with newer unit coverage.
- Legacy JSON fixtures under `tests/fixtures/` have been removed to reduce noise. Prefer test‑local factories or the behavioral Python fixtures under `tests/fixtures/behavioral/` when static data is needed (see `tests/fixtures/DEPRECATED.md`).

- Dev-iteration test scripts under `src/` that duplicated integration coverage have been archived to `archived/tests/dev_iterations/`:
  - `src/test_coinbase_api.py` (see `tests/integration/real_api/test_coinbase_connectivity.py` and `tests/integration/test_cdp_comprehensive.py`)
  - `src/bot_v2/scripts/test_all_slices.py` (see `tests/integration/bot_v2/test_slice_isolation.py` and `test_vertical_slice.py`)
  - `src/bot_v2/test_api_integration.py`, `test_workflow_adapter.py`, `test_workflow_engine_fixes.py` (see `tests/integration/bot_v2/test_workflows.py` and `test_orchestration.py`)

- CDP tests consolidated: older `tests/integration/test_cdp_connection.py` and `test_official_cdp.py` were redundant with `tests/integration/test_cdp_comprehensive.py` and have been archived under `archived/tests/dev_iterations/obsolete_cdp/`.

Only `tests/unit/` and `tests/integration/` are considered active for CI.

Deselected & Skipped Backlog (2024-11-24)

The default `pytest.ini` selection (`-m "not integration and not real_api and not perf and not performance and not uses_mock_broker"`) leaves 58 tests out of CI. The tables below capture why each group is excluded today and how we plan to address it.

Integration marker (deselected by default):

| Module | Tests | Blocker | Disposition |
| --- | --- | --- | --- |
| tests/integration/bot_v2/features/brokerages/coinbase/test_sandbox_smoke.py | 2 | Requires live sandbox credentials and optional order placement env toggles | Keep opt-in (run only with explicit sandbox vars) |
| tests/integration/bot_v2/perps/test_perps_e2e.py | 1 | Perps cycle still gated behind `COINBASE_ENABLE_DERIVATIVES` + INTX access | Keep opt-in; document INTX requirement |
| tests/integration/bot_v2/test_adaptive_portfolio.py | 6 | Script-style suite; no external deps but mixes concerns and includes CLI shim | Refactor into smaller unit slices and delete script harness |
| tests/integration/bot_v2/test_slice_isolation.py | 10 | Acts as static analysis for slice hygiene; better enforced via tooling | Replace with lint check; remove from pytest once tool exists |
| tests/integration/bot_v2/test_workflows.py | 7 | Exercises `bot_v2.workflows` (flagged `__experimental__ = True`) | Leave deselected until workflows graduate from experimental |
| tests/integration/test_canary_profile.py | 15 | Heavy guard simulations but only use in-memory doubles | Split into focused unit tests and keep a slim integration smoke |
| tests/integration/test_current_setup.py | 0 | Legacy diagnostic script, not a real pytest module | Remove (move instructions to docs) |
| tests/integration/test_e2e_smoke.py | 1 | Full CLI spin-up; slower but valuable regression | Keep opt-in (nightly/scheduled) |
| tests/integration/test_risk_integration.py | 3 | Bot-level risk checks duplicating unit behavior | Migrate assertions into `tests/unit/bot_v2/live_trade/` |

Recently migrated: `tests/integration/perps/test_execution_preflight.py` and `tests/integration/test_auth_selection.py` now live under `tests/unit/bot_v2/orchestration/` and `tests/unit/bot_v2/features/brokerages/coinbase/` respectively, so their checks run in the default unit suite.

`uses_mock_broker` marker (deprecated path):

| Module | Tests | Blocker | Disposition |
| --- | --- | --- | --- |
| tests/unit/bot_v2/orchestration/test_mock_broker.py | 3 | Relies on deprecated `MockBroker`; marker keeps it out of CI | Decide whether to retire MockBroker or swap in DeterministicBroker, then re-enable |
| tests/unit/bot_v2/orchestration/test_perps_bot.py (`@pytest.mark.uses_mock_broker`) | 2 | Specific behaviors still stub via MockBroker | Convert to deterministic broker fixture and drop marker |
| tests/unit/bot_v2/orchestration/test_bot_streaming.py | 1 | Streaming path currently tied to MockBroker internals | Extract interface-friendly fixture and re-enable |

Explicit skips (`@pytest.mark.skip`):

| Location | Scope | Reason | Disposition |
| --- | --- | --- | --- |
| tests/unit/bot_v2/features/brokerages/coinbase/test_api_permissions.py::test_live_permission_check | test | Needs real Advanced Trade credentials; opt-in only | Marked `real_api`; runs in scheduled suites, otherwise skips when creds absent |
| tests/unit/bot_v2/features/live_trade/test_risk_backlog.py::test_circuit_breakers_placeholder | test | Circuit breaker APIs not yet implemented | `xfail` placeholder with TODO(2025-01-31) to implement |
| tests/unit/bot_v2/features/live_trade/test_risk_backlog.py::test_impact_cost_placeholder | test | Impact cost methods missing | `xfail` placeholder with TODO(2025-01-31) to implement |
| tests/unit/bot_v2/features/live_trade/test_risk_backlog.py::test_position_sizing_placeholder | test | Dynamic position sizing backlog | `xfail` placeholder with TODO(2025-01-31) to implement |
| tests/unit/bot_v2/features/live_trade/test_risk_backlog.py::test_risk_metrics_placeholder | test | Risk metrics aggregation missing | `xfail` placeholder with TODO(2025-01-31) to implement |

Conditional skips via `pytest.skip(...)` remain, but they only trigger when environment fixtures (e.g., sandbox templates) are absent. Track these locally while refactoring the related modules.
