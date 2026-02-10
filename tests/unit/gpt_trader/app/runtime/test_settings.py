"""Tests for the immutable runtime settings snapshot helpers."""

from __future__ import annotations

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.runtime.settings import (
    RuntimeSettingsSnapshot,
    create_runtime_settings_snapshot,
    ensure_runtime_settings_snapshot,
)


def test_runtime_settings_snapshot_is_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot data should be immutable and include captured env vars."""
    monkeypatch.setenv("GPT_TRADER_TEST_OVERRIDE", "snapshot-value")

    config = BotConfig()
    config.symbols = ["BTC-USD"]
    snapshot = create_runtime_settings_snapshot(config)

    assert snapshot.env_vars["GPT_TRADER_TEST_OVERRIDE"] == "snapshot-value"
    assert isinstance(snapshot.config_data["symbols"], tuple)
    assert snapshot.config_data.risk.max_leverage == config.risk.max_leverage
    serialized = snapshot.serialize()
    assert serialized == snapshot.serialize()

    with pytest.raises(AttributeError):
        snapshot.config_data.symbols = ["ETH-USD"]

    with pytest.raises(AttributeError):
        snapshot.config_data.risk = None

    with pytest.raises(AttributeError):
        snapshot.config_data["symbols"].append("ETH-USD")  # tuple doesn't support append


def test_ensure_runtime_settings_snapshot_handles_both_types() -> None:
    """Ensure helper returns snapshots for BotConfig and passes through existing snapshots."""
    config = BotConfig()
    snapshot = create_runtime_settings_snapshot(config)

    assert ensure_runtime_settings_snapshot(snapshot) is snapshot
    result = ensure_runtime_settings_snapshot(config)

    assert isinstance(result, RuntimeSettingsSnapshot)
    assert result is not snapshot
