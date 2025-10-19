# Legacy Cleanup Inventory

Snapshot of active vs legacy surfaces to guide the deprecation effort. Paths
use repository-relative references.

## Code

- Active stack: `src/bot_v2/**` (CLI, orchestration, Coinbase adapters, risk)
- Legacy bundle: `var/legacy/legacy_bundle_latest.tar.gz` (contains the former
  `archived/experimental/**` tree and the `src/gpt_trader/**` PoC CLI)
- ✅ Rebranded metrics exporter outputs to the `coinbase_trader_*` prefix (update dashboards to match)
- ✅ Removed stale `ml_strategy` block from `config/system_config.yaml`; keep an eye
  out for reintroductions when syncing configs across environments

## Documentation

- Core guides with mixed messaging:
  - `README.md` (“Experimental” section referencing `archived/experimental/**`)
  - `docs/ARCHITECTURE.md` (describes archived slices as part of the system)
  - `docs/agents/Agents.md`, `docs/guides/agents.md`
  - `docs/DASHBOARD_GUIDE.md` (restoration steps for legacy monitoring)
- Historical bundle already segregated under `docs/archive/**`
- `docs/reference/system_capabilities.md` now stubs to `docs/archive/2024/system_capabilities.md`,
  the December 2024 perps-first snapshot (historical context only)

## Tests

- Active suite: `tests/unit/bot_v2/**` (1484 collected / 1483 selected / 1 deselected)
- ✅ Legacy PoC test modules removed from the workspace; refer to the legacy
  bundle if historical coverage is required.

## Bundling Legacy Modules

- Preferred: use the curated tarball checked in under
  `var/legacy/legacy_bundle_latest.tar.gz`.
- Need to regenerate the bundle? Check out a commit/tag that still contains
  `archived/experimental/**` and `src/gpt_trader/**`, then archive the paths
  manually. The historical helper (`scripts/maintenance/create_legacy_bundle.py`)
  has been retired to avoid promising tooling that no longer works on the trimmed
  tree.

## Immediate Targets

1. Maintain a versioned tag for the legacy bundle (e.g., `legacy/2025-10`) so the
   archived code remains easy to retrieve.
2. Continue collapsing legacy recovery instructions into `docs/archive/legacy_recovery.md`
   and prune remaining doc references to removed modules.
3. Audit lingering environment/database naming for legacy branding (e.g.,
   `gpt_trader` defaults) and plan follow-up remediation if necessary.
