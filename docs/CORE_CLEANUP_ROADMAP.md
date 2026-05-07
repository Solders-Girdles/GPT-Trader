# Core Cleanup Roadmap

---
status: current
last-updated: 2026-05-06
---

This roadmap tracks cleanup work that prepares GPT-Trader for larger rebuilds.
It is not a feature backlog. Each item should either remove drift, clarify the
canonical path, or identify a behavior decision that must happen before more
automation is added.

## Operating Rules

- Keep cleanup passes small enough to verify and commit independently.
- Start each pass from a clean working tree and end with `git status --short --branch`
  showing only the branch line.
- Prefer removing or rehoming stale surfaces before rewriting core behavior.
- Treat broker/profile availability as implementation state, not product approval.
  Use [Pre-Migration Decision Framework](PRE_MIGRATION_DECISION_FRAMEWORK.md)
  before adding or enabling execution paths.
- Keep generated inventories current with `uv run agent-regenerate --verify` when a
  pass moves, removes, or changes generated-artifact inputs.
- Do not preserve compatibility shims only because they exist. Either keep them
  intentionally, deprecate them with a target, or remove them with tests.

## Cleanup Lanes

| Lane | Goal | Current State | Next Useful Pass |
|------|------|---------------|------------------|
| Stale surfaces | Remove commands, config, docs, and scripts that advertise unsupported or retired flows. | Legacy paper runner, retired script metadata, and duplicate agent Make targets have been removed. | Continue removing stale surfaces as they are found during focused lanes. |
| Canonical entrypoints | Make CLI, TUI, preflight, scripts, and profiles point to one maintained path per workflow. | [Script taxonomy](../scripts/README.md) documents script directories and root exceptions. | Keep future scripts aligned with the taxonomy. |
| Profile truth | Keep runtime profiles tied to explicit `Profile` values and readiness gates. | `ProfileLoader` is the profile gate; unregistered profile configs were removed. | Review docs and tests for language that treats `canary` or `prod` as sufficient live-trading approval. |
| Compatibility shims | Collapse old aliases and fallback shapes that keep obsolete behavior alive. | TUI `active_guards` and `EventStore.events`/`path` compatibility have been removed; `docs/DEPRECATIONS.md` lists the remaining inventory. | Continue with `RiskConfig.daily_loss_limit`. |
| Test truth | Delete or modernize tests that protect behavior the project no longer wants. | Legacy-test triage tooling is active, CI-checked, and has no current untriaged candidates. | Re-run triage after future test moves or compatibility-shim removals. |
| Generated artifacts | Keep `var/agents/**` aligned with source truth without making every pass noisy. | `agent-regenerate --verify` is the current guardrail. | Tighten docs around when regeneration is required and when verify-only is enough. |
| Product and execution scope | Keep venue/API/account capability ahead of execution-path expansion. | Human-approved execution is the first AI-assisted target; broker-neutral records are canonical. | Inventory execution paths that could bypass human approval or readiness evidence. |

## Ready Queue

These are candidates for the next small cleanup passes:

1. **Deprecation inventory pass**
   - Continue with one bounded item from [Deprecations](DEPRECATIONS.md).
   - Suggested next target: `RiskConfig.daily_loss_limit`.
   - Leave `BotConfig.from_dict` profile mapping until profile migration decisions
     are clearer.

2. **Live-operation language audit**
   - Search docs for claims that `canary`, `prod`, futures, perps, or spot support
     imply automated execution approval.
   - Reframe those claims around readiness evidence and human-approved execution.

## Needs Decision

These should not be handled as drive-by cleanup:

| Topic | Decision Needed |
|-------|-----------------|
| Coinbase legacy credential support | Keep indefinitely, deprecate, or split UI detection from runtime acceptance. |
| `prod` and `canary` profile meaning | Whether they remain live-operation assets or become labels under a newer approval ladder. |
| `RiskConfig.daily_loss_limit` | Whether absolute-dollar loss limits remain supported alongside percentage risk budgets. |
| Event JSONL compatibility | Whether JSONL remains an accepted fallback or becomes import-only historical data. |
| Perps/futures execution paths | Which paths are retained as gated adapters before broader AI-assisted execution work. |

## Recent Cleanup Baseline

- Preserved pre-migration decision docs as local cleanup history.
- Aligned migration framing around human-approved execution and broker-neutral records.
- Routed TUI profile selection through `ProfileLoader`.
- Removed unregistered profile configs.
- Removed the legacy paper trading runner.
- Renamed operator smoke probes and moved them under `scripts/ops`.
- Moved the product catalog probe under `scripts/ops`.
- Rehomed analysis, monitoring, and readiness scripts under the script taxonomy.
- Documented script directories and root exceptions in `scripts/README.md`.
- Pruned stale cleanup surfaces from `pyproject.toml` and the Makefile.
- Closed review-only legacy-test triage false positives; triage now reports no
  untriaged candidates.
- Retired duplicate agent Make target aliases.
- Removed TUI `active_guards` guard-shape compatibility.
- Removed `EventStore.events` and `EventStore.path` compatibility aliases.

## Verification Bundle

Use the smallest relevant subset, but prefer this bundle after cleanup passes that
touch docs, scripts, config, or generated-artifact inputs:

```bash
git status --short --branch
uv run ruff check .
uv run python scripts/ci/check_legacy_patterns.py
uv run python scripts/ci/check_deprecation_registry.py
uv run python scripts/maintenance/docs_link_audit.py
uv run python scripts/maintenance/docs_reachability_check.py
uv run agent-regenerate --verify
git diff --check
```
