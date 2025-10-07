# GPT-Trader Naming Refactor Plan

This plan complements the draft naming standards (`docs/agents/naming_standards_outline.md`) and outlines the concrete rename waves we will drive during Sprints 0–2. Use it to prepare backlog entries, coordinate execution, and track verification needs.

## Wave 0 – Inventory & Policy (Sprint 0)
- **Objectives:** Finish the naming inventory (T-002, T-007) and finalize the standards document.
- **Key Tasks:**
  - Run automation to flag banned abbreviations (`cfg`, `svc`, `qty`, etc.) across `src/` and `tests/`.
  - Generate reports via `python scripts/agents/naming_inventory.py` (JSON → `var/agents/naming_inventory.json`, summary → `docs/agents/naming_inventory.md`).
  - Classify each finding by subsystem (config, orchestration, CLI/tests) and attach risk notes.
  - Capture doc updates needed in `docs/archive/agents/templates.md`.
- **Exit Criteria:** Standards ratified, prioritized rename backlog drafted, kickoff checklist signed off.

## Wave 1 – Config & Orchestration Cleanups (Late Sprint 0 / Early Sprint 1)
- **Scope:**
  - Rename short-form config variables to descriptive names:
    - `cfg` → `config` in `src/bot_v2/persistence/config_store.py` and `src/bot_v2/orchestration/broker_factory.py`.
    - `svc_registry` → `service_registry` in `src/bot_v2/orchestration/bootstrap.py`.
  - Update related tests or fixtures once they are discovered by the inventory scripts.
  - Convert helper scripts and validation utilities to use `quantity` naming ahead of core protocol changes (e.g., `scripts/validation/validate_perps_e2e.py`, `scripts/perps_dashboard.py`).
- **Risk:** Low; changes are local variables/parameters but touch initialization paths.
- **Verification:**
  - Run targeted tests covering config load (`tests/bot_v2/config`) and orchestration bootstrap.
  - `poetry run perps-bot run --profile dev --dev-fast` smoke test.
  - Confirm docs referencing these names (if any) are updated.
- **Backlog Seeds:**
  - `R-001` ConfigStore variable rename.
  - `R-002` Brokerage factory variable rename.
  - `R-003` Orchestration bootstrap rename.
  - `R-004` Script helper terminology alignment (`quantity` naming in validation/monitoring tools).

## Wave 1.5 – Naming Consolidation (Sprint 1 kickoff)
- **Scope:**
  - Collapse duplicate error hierarchies by promoting `src/bot_v2/errors.py` to `src/bot_v2/errors/base.py` and updating imports; preserve a re-export shim for one sprint.
  - Normalize root-level helpers into packages (`src/bot_v2/logging_setup.py` → `src/bot_v2/logging/setup.py`, `src/bot_v2/system_paths.py` → `src/bot_v2/config/path_registry.py`, `src/bot_v2/validate_calculations.py` → `src/bot_v2/validation/calculation_validator.py`).
  - Remove sprint/version suffixes from live-trade modules (`execution_v3.py` → `advanced_execution.py`, `perps_baseline_v2.py` → `perps_baseline_enhanced.py`, `week2_filters.py` → `liquidity_filters.py`) and add import shims with deprecation warnings.
  - Replace Coinbase helper modules with descriptive names (e.g., promote the market data helper to `market_data_features.py` and consolidate generic helpers under `utilities.py`) and update unit tests referencing the old names.
- **Risks:** Medium; import paths across orchestration, live trade, and tests will shift. Alias modules should ship alongside renames to keep the CLI stable.
- **Verification:**
- Run targeted suites: `tests/unit/bot_v2/orchestration/test_bootstrap.py`, `tests/unit/bot_v2/features/live_trade/test_advanced_execution.py`, Coinbase adapter/unit suites under `tests/unit/bot_v2/features/brokerages/coinbase/`.
  - Execute `poetry run pytest tests/unit/bot_v2/errors` (once curated) to validate re-export shims.
  - Manual smoke: `poetry run perps-bot run --profile dev --dev-fast` to confirm orchestrator imports remain healthy.
- **Backlog Seeds:**
  - `R-050` Error hierarchy reshuffle (`errors.py` → `errors/base.py`).
  - `R-051` Logging/config helper package moves.
  - `R-052` Live-trade module rename (advanced execution + baseline strategy).
  - `R-053` Coinbase helper module rename and test alignment.
  - `R-054` Author import alias/deprecation shims for renamed modules.
- **Execution Checklist:**
  - **Preparation (Before coding):**
    1. Extend `scripts/agents/naming_inventory.py` patterns to catch the legacy helper suffix and `logging_setup` (word-boundary tweak or explicit terms) and re-run inventory; drop results in `docs/agents/naming_inventory.md` for traceability (historical Sprint 0 exports remain in git history).
    2. Draft design notes for each rename batch (errors, helpers, live trade, Coinbase) summarizing public API touch points and alias requirements; link notes in the backlog items above.
    3. Align on glossary acceptance for `perps` vs `perpetuals` to avoid mid-wave reversions; record decision alongside the current docs (wave-one notes live in repository history).
  - **Implementation (Per PR cadence):**
    1. Execute one rename batch at a time, starting with error hierarchy (least external exposure) → helper modules → live-trade modules → Coinbase helper package.
    2. Introduce alias modules (`errors/__init__.py`, `features/live_trade/execution_v3.py`, etc.) emitting `DeprecationWarning` and schedule their removal in Wave 3 (`R-020`).
    3. Update imports, docs, and scripts in the same PR to avoid broken references (search for old filenames with `rg` and update in-place).
    4. Run targeted unit tests listed above plus `poetry run pytest tests/unit/bot_v2/features/live_trade/test_risk_validation.py` when touching live-trade risk dependencies.
    5. Capture manual verification notes (CLI smoke, docs build) in the PR description and link back to this checklist.
  - **Follow-Up (After merge):**
    1. Re-run naming inventory to confirm the old file names disappear and alias warnings behave as expected.
    2. Record completion notes alongside the project docs (Wave 1 weekly notes are preserved via git history).
    3. Set reminders in the backlog for alias removal (Wave 3) and ensure automation checks flag reintroductions of banned names.

## Wave 2 – Quantity Terminology Alignment (Sprint 1 core effort)
- **Scope:** Replace `qty`/`order_qty` naming with `quantity` across the trading stack.
  - Data models (`Order.qty`, `Position.qty`, etc.) in `src/bot_v2/features/brokerages/core/interfaces.py` and downstream consumers.
  - CLI flags and arguments now expose only `--order-quantity`; the legacy `--order-qty` alias was removed when the CLI moved to `src/bot_v2/cli/`.
  - Strategy modules, deterministic broker, backtests, and tests relying on `qty`.
  - Serialization/logging emitters to prefer `quantity` while accepting legacy fields during transition.
- **Risk:** High; affects public CLI, persisted state representations, and broad strategy code.
- **Mitigations:**
  - Ship temporary adapters/aliases (e.g., accept both `qty` and `quantity` in payloads) and surface deprecation warnings.
  - Stage renames per subsystem (core data models → orchestration → strategies → tests) with green CI between each PR.
  - Update documentation (`README`, guides) and sample configs simultaneously.
  - Provide helper conversion utilities so dashboards/logs emit both keys until consumers migrate.
- **Compatibility Plan:**
  1. Introduce `quantity` fields alongside existing `qty` attributes in dataclasses/serializers; emit both values and mark `qty` as deprecated in docstrings.
  2. Ship `--order-quantity` as the sole CLI flag; the former `--order-qty` alias was retired once downstream scripts confirmed adoption. ✅ Completed 2025-03.
  3. Update REST/WS payload builders to populate both keys when interacting with external APIs, ensuring downstream systems remain stable.
  4. ✅ After one full sprint, drop `qty` aliases once consumers confirm adoption; naming inventory now enforces the canonical field set (completed Sprint 1 wrap-up).
  5. Track rollout status and external dependencies in a current status log (Wave 1 timeline is available via git history) and update README/CLI docs once consumers confirm adoption.
- **Verification:**
  - Full unit/integration suite.
  - Regenerate or replay sample trades to confirm no serialization drift.
  - CLI smoke tests accepting both old and new flags during the alias period.
- **Backlog Seeds:**
  - `R-010` Core data model rename (`qty` → `quantity`).
  - `R-011` CLI flag alignment and alias support.
  - `R-012` Strategy/backtest updates.
  - `R-013` Documentation refresh for quantity terminology.

## Wave 3 – Follow-Up & Governance (Sprint 2)
- **Scope:**
  - Remove transitional aliases (e.g., legacy CLI flags) once downstream teams confirm adoption. (Wave 1.5 shims already retired in Sprint 1.)
  - Integrate naming enforcement into automation (extend T-002/T-007 outputs into CI checks).
  - Publish rename changelog entries and dashboard metrics.
- **Backlog Seeds:**
  - `R-020` Remove deprecated aliases and finalize terminology.
  - `R-054A` ✅ Retired live-trade strategy shims.
  - `R-054B` ✅ Retired execution shim (`execution_v3`).
  - `R-054C` ✅ Retired Coinbase utility shims.
  - `R-054D` ✅ Retired helper module shims (`logging_setup`, `system_paths`, `validate_calculations`).
  - `R-021` Promote naming checks into preflight/CI.
  - `R-022` Update governance docs with final standards and migration notes.

## Tracking & Reporting
- Log progress in weekly status notes stored with the active documentation set (Wave 1 snapshots remain accessible through git history).
- Mirror backlog IDs in the Kanban board and reference this plan in PR descriptions.
- Update `CONTRIBUTING.md` once renames stabilize to embed the final standard.

---

**Next Actions**
1. Circulate this plan for feedback alongside the naming standards outline.
2. Expand the backlog (`T-008+`, `R-001+`) in the shared tracker once inventory results land.
3. Review the latest naming inventory summary (recommended path: `docs/agents/naming_inventory.md`; Sprint 0 exports are in git history) to prioritize initial rename candidates.
4. Draft the pilot rename PR (Wave 1) with checklists covering code, tests, and docs.
