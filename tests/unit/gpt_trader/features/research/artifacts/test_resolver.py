from __future__ import annotations

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.features.research.artifacts import (
    StrategyArtifact,
    StrategyArtifactResolutionError,
    StrategyArtifactStore,
    apply_strategy_artifact_to_config,
)


def _write_artifact(store: StrategyArtifactStore, *, approved: bool) -> StrategyArtifact:
    artifact = StrategyArtifact.create(
        strategy_type="baseline",
        symbols=["BTC-USD"],
        interval=120,
        strategy_parameters={"short_ma_period": 8, "long_ma_period": 34},
        metrics={"sharpe": 1.3},
        validation={"max_drawdown": 0.08},
        source="unit_test",
    )
    artifact.approved = approved
    store.save(artifact)
    return artifact


def test_apply_strategy_artifact_updates_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STRATEGY_ARTIFACT_ROOT", str(tmp_path))
    store = StrategyArtifactStore()
    artifact = _write_artifact(store, approved=True)

    config = BotConfig()
    config.strategy_artifact_id = artifact.artifact_id

    applied = apply_strategy_artifact_to_config(config)

    assert applied is not None
    assert config.strategy_type == "baseline"
    assert config.symbols == ["BTC-USD"]
    assert config.interval == 120
    assert config.strategy.short_ma_period == 8
    assert config.strategy.long_ma_period == 34
    assert config.metadata.get("strategy_artifact_id") == artifact.artifact_id


def test_unapproved_artifact_blocks_live(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STRATEGY_ARTIFACT_ROOT", str(tmp_path))
    store = StrategyArtifactStore()
    artifact = _write_artifact(store, approved=False)

    config = BotConfig()
    config.strategy_artifact_id = artifact.artifact_id

    with pytest.raises(StrategyArtifactResolutionError):
        apply_strategy_artifact_to_config(config)


def test_unapproved_artifact_allowed_in_dry_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("STRATEGY_ARTIFACT_ROOT", str(tmp_path))
    store = StrategyArtifactStore()
    artifact = _write_artifact(store, approved=False)

    config = BotConfig(dry_run=True)
    config.strategy_artifact_id = artifact.artifact_id

    applied = apply_strategy_artifact_to_config(config)

    assert applied is not None
    assert config.symbols == ["BTC-USD"]
