# Selective Test Runner

The selective test runner combines the dependency analysis tooling with the
CI workflow to run only the tests impacted by a change set.

## Local Usage

1. Regenerate the dependency artifacts after code changes:
   ```bash
   poetry run python scripts/analysis/dependency_map.py --tests "tests/**/*.py" --output dependency_report.json
   poetry run python scripts/analysis/test_categorizer.py --output test_categories.json
   ```
2. Dry-run the selection to preview which tests will execute:
   poetry run python scripts/testing/selective_runner.py --paths src/gpt_trader/orchestration/trading_bot/bot.py --dry-run

   # Or execute it
   poetry run python scripts/testing/selective_runner.py --paths src/gpt_trader/orchestration/trading_bot/bot.py
3. Execute the selected tests (the command honours `PYTEST_ADDOPTS`):
   ```bash
   PYTEST_ADDOPTS='-m "not slow and not performance" -q' \
     poetry run python scripts/testing/selective_runner.py --paths src/gpt_trader/orchestration/trading_bot/bot.py
   ```

Passing test paths (e.g. `tests/unit/...`) forces those files to run directly.
High-impact modules listed in CI are upgraded to a full-suite run automatically;
the same behaviour can be triggered locally with
`--auto-full-module <module-prefix>`.

## CI Integration

Pull request builds use the selective runner with these safeguards:

- Dependency and category artifacts are regenerated before selection.
- Changed files are collected with `git diff --name-only base HEAD -- '*.py'` and
  passed via `--paths`.
- `PYTEST_ADDOPTS` enforces the standard unit-test markers.
- The runner upgrades to the full suite if the selected set exceeds 70% of the
  total catalogued tests.
- Changes touching `gpt_trader.features.brokerages.core.interfaces` force a full
  run and log the trigger.
- Push builds (including merges to `main`) continue to run the full coverage
  suite.

## Verifying CI Behaviour

1. Push a PR that only touches a leaf helper (e.g. a strategy helper) and
   confirm the action log shows `poetry run python scripts/testing/selective_runner.py --paths ...` followed by a short pytest run.
2. Modify `gpt_trader/features/brokerages/core/interfaces.py` and ensure the log
   prints `Full run triggered by high-impact module` and the job executes the
   full suite.
3. Push to `main` (or run the workflow on a branch with `workflow_dispatch`) and
   verify the coverage step still executes the full pytest command with
   `--cov`.

These checks keep the selective runner trustworthy while preserving the safety
net of full-suite coverage on integration branches.
