# Strategy Artifacts

---
status: current
last-updated: 2026-01-30
---

## Purpose

Strategy artifacts are the contract between research/backtesting and live trading. They
capture strategy parameters, validation evidence, and approval status in a versioned,
auditable file so live trading can only run approved research outputs.

When using the research backtesting adapter, order intent can be expressed via
`Decision.indicators` (`order_type`, `price`/`limit_price`, `stop_price`,
`tif`/`time_in_force`, `reduce_only`). This ensures research backtests align with
the canonical broker semantics referenced by artifacts.

## Storage

Artifacts are stored as JSON under:

- `runtime_data/strategy_artifacts/`

## Automatic emission (optimize)

`gpt-trader optimize run` automatically writes a Strategy Artifact from the best feasible
trial. The artifact includes backtest metrics plus an evidence pointer to the
optimization `results.json` file. Artifacts are saved unapproved by default, so they
must still be published and activated before live trading.

## Schema (summary)

Each artifact includes:
- `artifact_id`, `strategy_type`, `created_at`
- `symbols`, `interval`
- `strategy_parameters`, `mean_reversion_parameters`, `ensemble_parameters`, `regime_parameters`
- `risk_parameters`
- `metrics`, `validation`, `evidence_paths`
- `approved`, `approved_at`, `approved_by`, `notes`

See `src/gpt_trader/features/research/artifacts/models.py` for the full definition.

## Publish and Activate

Use the StrategyArtifactStore API to write, publish, and activate artifacts:

```python
from gpt_trader.features.research.artifacts import StrategyArtifact, StrategyArtifactStore

artifact = StrategyArtifact.create(
    strategy_type="baseline",
    symbols=["BTC-USD"],
    interval=60,
    strategy_parameters={"short_ma_period": 5, "long_ma_period": 20},
    metrics={"sharpe": 1.2},
    validation={"max_drawdown": 0.08},
    evidence_paths=["runtime_data/canary/reports/backtest_20260130.json"],
    source="research",
)

store = StrategyArtifactStore()
store.save(artifact)
store.publish(artifact.artifact_id, approved_by="owner")
store.set_active("canary", artifact.artifact_id)
```

CLI shortcuts:

```bash
gpt-trader optimize artifact-publish artifact-123 --approved-by owner
gpt-trader optimize artifact-activate artifact-123 --profile canary
```

## Using artifacts in live trading

Live config can reference artifacts in three ways:

1. `strategy_artifact_id`: load a specific artifact by id.
2. `strategy_artifact_path`: load a specific artifact file path.
3. `strategy_artifact_use_registry`: load the active artifact for the profile (from
   `runtime_data/strategy_artifacts/active.json`).

Artifacts are required to be approved for live trading unless you are in dry-run, mock,
or paper-fills mode, or you explicitly set `strategy_artifact_allow_unapproved=True`.

### Environment variables

- `STRATEGY_ARTIFACT_ID`
- `STRATEGY_ARTIFACT_PATH`
- `STRATEGY_ARTIFACT_USE_REGISTRY`
- `STRATEGY_ARTIFACT_ALLOW_UNAPPROVED`
