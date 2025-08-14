from __future__ import annotations

from datetime import datetime, timedelta

from bot.live.production_orchestrator import OrchestratorConfig, ProductionOrchestrator


class DummyBroker:
    pass


class DummyKB:
    pass


def test_get_audit_summary_returns_counts_and_latest(monkeypatch):
    cfg = OrchestratorConfig(background_enabled=False)
    orch = ProductionOrchestrator(cfg, DummyBroker(), DummyKB(), symbols=["AAPL"])

    # Inject a few operations
    now = datetime.now()
    orch.operation_history.extend(
        [
            {
                "operation": "selection_change",
                "timestamp": now - timedelta(hours=1),
                "data": {"a": 1},
            },
            {"operation": "rebalance", "timestamp": now - timedelta(hours=2), "data": {"b": 2}},
            {
                "operation": "trade_blocked",
                "timestamp": now - timedelta(minutes=5),
                "data": {"c": 3},
            },
        ]
    )

    summary = orch.get_audit_summary(window_hours=24)
    counts = summary.get("counts", {})
    assert counts.get("selection_change") == 1
    assert counts.get("rebalance") == 1
    assert counts.get("trade_blocked") == 1

    latest = summary.get("latest", {})
    assert latest.get("selection_change") is not None
    assert latest.get("rebalance") is not None
    assert latest.get("trade_blocked") is not None
