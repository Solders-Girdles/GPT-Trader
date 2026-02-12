"""Regression coverage for risk state persistence recovery paths."""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal
from typing import Any

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
from gpt_trader.utilities.datetime_helpers import utc_now
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,  # naming: allow
)


def _cfm_balance(buffer_pct: float) -> dict[str, Any]:
    return {
        "liquidation_buffer_percentage": buffer_pct,
        "total_usd_balance": Decimal("10000"),
        "maintenance_margin": Decimal("500"),
    }


def test_load_state_handles_malformed_json(tmp_path: pathlib.Path) -> None:
    """Corrupted JSON should not crash initialization and should reset defaults."""
    state_file = tmp_path / "risk_state.json"
    state_file.write_text("{ malformed json }\n")

    manager = LiveRiskManager(config=MockConfig(), state_file=str(state_file))

    assert manager._state_load_error is not None
    assert manager._start_of_day_equity is None
    assert manager._reduce_only_mode is False
    assert manager._daily_pnl_triggered is False

    warnings = manager.check_cfm_liquidation_buffer(_cfm_balance(0.1))
    assert warnings
    assert manager.get_risk_warnings()


def test_load_state_handles_partial_payload(tmp_path: pathlib.Path) -> None:
    """Valid but partial state payload should apply available fields."""
    today = utc_now().strftime("%Y-%m-%d")
    state_payload = {
        "date": today,
        "daily_pnl_triggered": True,
        "reduce_only_mode": True,
        "reduce_only_reason": "partial_payload",
    }

    state_file = tmp_path / "risk_state.json"
    state_file.write_text(json.dumps(state_payload))

    manager = LiveRiskManager(state_file=str(state_file))

    assert manager._state_load_error is None
    assert manager._daily_pnl_triggered is True
    assert manager._reduce_only_mode is True
    assert manager._reduce_only_reason == "partial_payload"
    assert manager._start_of_day_equity is None
    assert manager._cfm_reduce_only_mode is False


def test_save_state_failure_records_error_and_allows_warnings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistence permission failures should be tracked and should not block warnings."""
    state_file = tmp_path / "risk_state.json"
    manager = LiveRiskManager(config=MockConfig(), state_file=str(state_file))

    def _raise_permission(self, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(pathlib.Path, "mkdir", _raise_permission)

    manager.set_reduce_only_mode(True, reason="save_failure")

    assert manager._reduce_only_mode is True
    assert manager._state_save_error is not None

    warnings = manager.check_cfm_liquidation_buffer(_cfm_balance(0.1))
    assert warnings
    assert manager.get_risk_warnings()
