from __future__ import annotations

from typing import Any

import pytest
from bot.live import audit as audit_mod


class FakeOrchestrator:
    def __init__(self):
        self.ops: list[dict[str, Any]] = []

        class Cfg:
            mode = type("M", (), {"value": "semi_automated"})

        self.config = Cfg()

        class Obs:
            def log_decision(
                self, decision_type: str, decision_data: dict[str, Any], metadata: dict[str, Any]
            ):
                pass

        self.observability = Obs()

    def _record_operation(self, op: str, data: dict[str, Any]) -> None:
        self.ops.append({"operation": op, "data": dict(data)})


def test_record_selection_change_schema():
    orch = FakeOrchestrator()
    old_sel = ["s1", "s2"]
    new_sel = ["s2", "s3"]
    audit_mod.record_selection_change(orch, old_sel, new_sel)

    assert orch.ops and orch.ops[-1]["operation"] == "selection_change"
    payload = orch.ops[-1]["data"]
    assert set(payload.keys()) >= {
        "timestamp",
        "old_selection",
        "new_selection",
        "added",
        "removed",
    }
    assert payload["added"] == ["s3"]
    assert payload["removed"] == ["s1"]


def test_record_rebalance_schema():
    orch = FakeOrchestrator()
    changes = {
        "AAPL": {"current_weight": 0.1, "target_weight": 0.2, "change": 0.1},
        "MSFT": {"current_weight": 0.2, "target_weight": 0.1, "change": -0.1},
    }
    audit_mod.record_rebalance(orch, changes)
    assert orch.ops and orch.ops[-1]["operation"] == "rebalance"
    payload = orch.ops[-1]["data"]
    assert set(payload.keys()) >= {"timestamp", "changes", "total_abs_change"}
    assert pytest.approx(payload["total_abs_change"], rel=1e-9) == 0.2


def test_record_trade_blocked_schema():
    orch = FakeOrchestrator()
    audit_mod.record_trade_blocked(
        orch, reason="drawdown_guard", details={"current_drawdown": 0.2, "limit": 0.1}
    )
    assert orch.ops and orch.ops[-1]["operation"] == "trade_blocked"
    payload = orch.ops[-1]["data"]
    assert set(payload.keys()) >= {"timestamp", "reason", "current_drawdown", "limit"}
