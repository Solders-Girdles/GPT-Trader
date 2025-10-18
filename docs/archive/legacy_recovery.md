# Legacy Recovery Guide

Historical modules (archived experimental slices and the `gpt_trader` PoC) were
removed from the active workspace. Use this guide to restore them when needed.

## Sources

- Preferred: use the pre-generated bundle `var/legacy/legacy_bundle_latest.tar.gz`
  (copied before the legacy modules were removed)
- Recreate from history: check out a commit/tag that still contains the
  directories and run `make legacy-bundle` or
  `poetry run python scripts/maintenance/create_legacy_bundle.py --output legacy/legacy_bundle.tar.gz`
- The bundle includes:
  - `archived/experimental/**`
  - `src/gpt_trader/**`
  - Matching configs, scripts, monitoring helpers, and tests
  - `legacy_manifest.json` documenting captured paths

## Restore Steps

1. Extract the archive (or check out the `legacy/2025-10` tag when created).
2. Copy the required modules back into the repository:
   - Features → `src/bot_v2/features/`
   - Monitoring helpers → `src/bot_v2/monitoring/`
   - Scripts → `scripts/`
   - Configs → `config/`
   - Tests → `tests/unit/`
3. Install optional extras if the slice requires them (`poetry install -E ml -E research -E api`).
4. Run `poetry run pytest -q` to confirm coverage.
5. Remove the modules again after inspection so the mainline workspace stays legacy-free.
