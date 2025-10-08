from __future__ import annotations

from datetime import datetime

from src.bot_v2.config.live_trade_config import RiskConfig
from src.bot_v2.features.live_trade.risk.state_management import StateManager


class StubEventStore:
    def __init__(self) -> None:
        self.metrics: list[dict[str, object]] = []

    def append_metric(self, *, bot_id: str, metrics: dict[str, object]) -> None:
        self.metrics.append({"bot_id": bot_id, **metrics})


def test_state_manager_sets_reduce_only_and_persists_event() -> None:
    config = RiskConfig()
    config.reduce_only_mode = False
    store = StubEventStore()
    manager = StateManager(
        config=config,
        event_store=store,
        now_provider=lambda: datetime(2025, 1, 1, 0, 0, 0),
    )

    manager.set_reduce_only_mode(True, "limit_breach")

    assert manager.is_reduce_only_mode() is True
    assert config.reduce_only_mode is True
    assert store.metrics[-1]["bot_id"] == "risk_engine"
    assert store.metrics[-1]["enabled"] is True
    assert store.metrics[-1]["reason"] == "limit_breach"
    assert store.metrics[-1]["timestamp"] == "2025-01-01T00:00:00"

    manager.set_reduce_only_mode(False, "clear")
    assert manager.is_reduce_only_mode() is False
