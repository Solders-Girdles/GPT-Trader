# Experimental Features Archive

These modules were removed from the production codebase on 2025-10-09 and kept here for historical reference and research experiments.

**Status:** reference-only. The files retain their original imports (for example `bot_v2.features.backtest`) and will not run in place without being restored to `src/`.

## Archived Assets

- `features/` – experimental feature slices (ml_strategy, market_regime, backtest, adaptive_portfolio, paper_trade, strategies)
- `monitoring/` – legacy alerting, dashboard, and metrics helpers
- `scripts/` – retired operational scripts (for example `archived/experimental/scripts/monitoring/canary_monitor.py`)
- `config/` – configuration files associated with the experimental slices
- `tests/` – unit tests covering the archived code paths

## Restoring a Feature

1. Copy the relevant feature directory back to `src/bot_v2/features/`.
2. Copy configuration files into `config/`.
3. Copy the matching tests into `tests/unit/bot_v2/features/`.
4. Update imports if paths changed and run the test suite.
5. Address any integration or dependency gaps that arise.
