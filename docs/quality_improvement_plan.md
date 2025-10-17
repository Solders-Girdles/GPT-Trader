# GPT-Trader Quality Improvement Plan

## Overview
This document captures notable opportunities to improve maintainability, streamline workflows, and tighten organization across the GPT-Trader code base. The items below focus on changes that improve code structure, configuration hygiene, and test coverage without altering trading behaviour.

## High-Impact Opportunities

### 1. Break up the `PerpsBot` bootstrap sequence
*Observation.* `PerpsBot.__init__` mixes symbol normalization, filesystem setup, registry wiring, broker/risk initialization, and telemetry configuration inside one 100+ line constructor, making the orchestration layer hard to test and extend.【F:src/bot_v2/orchestration/perps_bot.py†L79-L200】

*Why it matters.* Centralising so much logic in the constructor couples unrelated responsibilities (environment parsing, storage layout, dependency creation) and complicates dependency injection for tests and alternative runtimes.

*First steps.*
- Introduce a thin factory/builder that prepares runtime paths and registry objects before instantiating `PerpsBot`.
- Extract symbol list normalisation and storage path selection into dedicated helpers with focused unit tests.
- Allow `PerpsBot` to accept pre-built collaborators (risk manager, execution engine) so integration tests can substitute fakes without touching environment variables.

### 2. Consolidate configuration parsing helpers
*Status.* Completed. `RiskConfig.from_env` now delegates to a `RiskConfigModel` Pydantic schema that draws inputs from `RuntimeSettings.snapshot_env`, keeping defaults and aliases aligned with the runtime snapshot.

*Highlights.*
- Validators normalise mapping payloads, enforce percentage bounds, and raise descriptive errors that reference the originating env var or JSON key.
- Shared unit tests capture the env key list via `RuntimeSettings.snapshot_env` and cover legacy aliases, invalid mappings, and out‑of‑range percentages to guard against regression.

*Next opportunities.*
- Extend the schema-driven approach to other configuration surfaces (e.g., spot risk, orchestration profiles) so loaders share the same validation guarantees.

### 3. Source mock product metadata from structured fixtures
*Observation.* `MockBroker._init_products` hardcodes extensive spot/perp metadata inline, duplicating market constants and forcing edits in code for simple test data tweaks.【F:src/bot_v2/orchestration/mock_broker.py†L35-L200】

*Why it matters.* Keeping large literal dictionaries in code bloats diffs, hides shared defaults (e.g., price increments), and makes it difficult to reuse the same fixtures in other tests.

*First steps.*
- Move the product specifications to JSON/YAML fixtures under `tests/fixtures/` or `config/brokers/`, and load them via a lightweight helper.
- Provide defaults (e.g., shared min size) in one place and allow overrides per symbol to reduce duplication.
- Expose a deterministic fixture factory that both the mock broker and tests can consume to ensure consistency.

### 4. Align experimental feature slices with optional dependencies
*Observation.* Experimental slices under `archived/experimental/features/*` expose `__experimental__ = True`, yet the optional dependency groups in `pyproject.toml` do not clearly map to those slices, leaving unused packages in the default install and complicating dependency trimming.【F:archived/experimental/features/backtest/__init__.py†L1-L14】【F:pyproject.toml†L36-L78】

*Why it matters.* Clear boundaries between core and experimental features make it easier to ship a lean runtime image and avoid unnecessary security patching.

*First steps.*
- Define extras that match the experimental slices (e.g., `backtest`, `monitoring`) and gate optional imports behind feature flags.
- Update installation docs to recommend `poetry install --without` for unused slices.
- Move experimental modules into a dedicated namespace or package subdirectory to emphasise their opt-in nature.

### 5. Restore lint and test visibility for the `tests/` tree
*Observation.* The Ruff configuration excludes the entire `tests` directory, so style regressions or typing issues in test utilities are never surfaced. The testing status doc also highlights legacy suites that remain skipped, hinting at a gap between documentation and automation.【F:pyproject.toml†L116-L150】【F:tests/TESTING_STATUS.md†L1-L22】

*Why it matters.* Allowing tests to drift stylistically lowers confidence in fixtures, while stale skip markers make it harder to evaluate suite health.

*First steps.*
- Narrow Ruff's `extend-exclude` to specific problematic files (or use per-file ignores) so the broader test tree is linted again.
- Add a CI check that fails when new `pytest.skip` markers are introduced without an entry in `TESTING_STATUS.md`.
- Evaluate whether any skipped suites can be archived or updated, reducing the maintenance burden of documenting their status.

## Additional Maintenance Ideas
- **Runtime path management.** Encapsulate the logic that chooses between `EVENT_STORE_ROOT`, `GPT_TRADER_RUNTIME_ROOT`, and default directories in a reusable `RuntimePaths` helper to avoid inconsistencies across services.【F:src/bot_v2/orchestration/perps_bot.py†L145-L169】
- **Dependency hygiene.** Review core dependencies listed in `pyproject.toml` to confirm each is needed for the default trading path; large packages like `scipy` or `ta-lib` might move to extras if not required at runtime.【F:pyproject.toml†L8-L34】

## Next Steps
1. Prioritise the high-impact items above based on effort and upcoming release timelines.
2. Convert selected opportunities into GitHub issues with acceptance criteria (e.g., "PerpsBot constructor refactored to ≤50 lines and dependencies injectable").
3. Schedule the work into maintenance sprints, ensuring each item gains automated tests to guard against regression.
