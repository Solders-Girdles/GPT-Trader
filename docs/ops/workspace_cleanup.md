# Workspace Cleanup Runbook

The workspace accumulates runtime artifacts (logs, caches, coverage reports) that
are safe to discard but can quickly consume hundreds of megabytes. Use this
runbook to keep the repository lean and predictable.

## When to Run

- Before cutting a release branch or handing the workspace to another teammate.
- When disk usage from `var/logs` or `var/data` starts to grow beyond a few
  hundred megabytes.
- After large scale local testing that generates `.mypy_cache`, `htmlcov`, or
  other tooling by-products.

> **Tip:** Stop any running trading bots before applying the cleanup so active
> logs are not rotated while being written.

## Commands

- Dry-run with a detailed plan:
  ```bash
  make clean-dry-run
  ```
- Apply the cleanup:
  ```bash
  make clean
  ```
- Run the script directly for custom thresholds:
  ```bash
  poetry run python scripts/maintenance/cleanup_workspace.py \
      --apply \
      --log-threshold-mb 96 \
      --data-threshold-mb 96 \
      --retention-days 10
  ```

## What Gets Cleaned

- Tool caches: `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.benchmarks/`
- Coverage artifacts: `htmlcov/`, `.coverage`, `coverage.json`,
  `pip-audit-report.json`
- Runtime logs (`var/logs`):
  - Keep the two most recent numbered rotations (e.g. `.log.1`, `.log.2`)
  - Archive older rotations to `var/logs/archive/*.gz`
  - Remove archived `.gz` files older than the retention window (default 14 days)
  - Rotate any active log above the size threshold once it has been idle for at
    least one hour
- Event-store JSONL files under `var/data/**` larger than the threshold are
  rotated into `var/data/archive/*.gz`

All actions are recorded with JSON lines in `logs/cleanup_audit.log`.

## Recovery

- Archived files remain under `var/logs/archive/` or `var/data/archive/`. Move
  them back and decompress if you need to inspect old runs.
- If tooling caches are required again, re-run the relevant command
  (`poetry run mypy`, `poetry run pytest`, etc.) and they will be recreated.

## Extending the Cleanup

To add an additional directory or file pattern:

1. Update `scripts/maintenance/cleanup_workspace.py`.
2. Document the addition in this runbook so future operators know what will be
   touched.
