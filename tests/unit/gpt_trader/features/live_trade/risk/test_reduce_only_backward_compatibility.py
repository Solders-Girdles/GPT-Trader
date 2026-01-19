"""Tests to verify backward compatibility of reduce-only state persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gpt_trader.features.live_trade.risk import manager as risk_manager_module
from gpt_trader.features.live_trade.risk.manager import LiveRiskManager


def test_reduce_only_in_memory_without_state_file() -> None:
    manager = LiveRiskManager(state_file=None)

    assert manager.is_reduce_only_mode() is False
    manager.set_reduce_only_mode(True, reason="test_enable")
    assert manager.is_reduce_only_mode() is True
    manager.set_reduce_only_mode(False, reason="test_disable")
    assert manager.is_reduce_only_mode() is False

    fresh_manager = LiveRiskManager(state_file=None)
    assert fresh_manager.is_reduce_only_mode() is False


def test_reduce_only_state_persists_across_restart(tmp_path: Path, monkeypatch) -> None:
    fixed_time = datetime(2024, 1, 2, tzinfo=UTC)
    monkeypatch.setattr(risk_manager_module, "utc_now", lambda: fixed_time)

    state_file = tmp_path / "risk_state.json"
    manager = LiveRiskManager(state_file=str(state_file))
    manager.set_reduce_only_mode(True, reason="persistence_test")

    reloaded = LiveRiskManager(state_file=str(state_file))
    assert reloaded.is_reduce_only_mode() is True
    assert getattr(reloaded, "_reduce_only_reason") == "persistence_test"
