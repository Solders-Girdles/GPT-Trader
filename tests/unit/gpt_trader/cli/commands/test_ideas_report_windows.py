from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from gpt_trader.features.trade_ideas import CloseoutResolution, TimeHorizon
from gpt_trader.features.trade_ideas.report import build_trade_idea_track_record_report
from gpt_trader.features.trade_ideas.service import TradeIdeaService
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea


def _idea(decision_id: str, *, expires_at: datetime | None = None, **overrides: Any) -> Any:
    return build_trade_idea(
        decision_id=decision_id,
        time_horizon=TimeHorizon(
            expected_hold="3-10 days",
            expires_at=expires_at or datetime(2035, 6, 19, 16, 0, tzinfo=UTC),
        ),
        **overrides,
    )


def test_windowed_report_ignores_lifecycle_after_cutoff(tmp_path: Path) -> None:
    root = tmp_path / "ideas"
    current_time = [datetime(2026, 5, 30, 12, 0, tzinfo=UTC)]
    service = TradeIdeaService(root, now_factory=lambda: current_time[0])

    idea = _idea("trade-window-late-fill")
    service.propose(idea, actor_id="idea-generator-v1")
    current_time[0] = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    service.approve(idea.decision_id, actor_id="rj", reason="Risk verified")
    service.record_submission(idea.decision_id, actor_id="operator", venue="manual")
    service.record_fill(idea.decision_id, actor_id="operator", venue="manual")
    service.record_closeout_attribution(
        idea.decision_id,
        actor_id="rj",
        resolution=CloseoutResolution.THESIS_TARGET,
        realized_profit_loss_amount=Decimal("42.00"),
        realized_profit_loss_percent=Decimal("1.5"),
    )

    report = build_trade_idea_track_record_report(
        service,
        now=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        since=datetime(2026, 5, 1, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 31, 23, 59, 59, tzinfo=UTC),
    )

    assert report["source"] == {
        "audit_event_count": 1,
        "closeout_count": 0,
        "idea_count": 1,
    }
    assert report["workflow"]["event_counts"]["proposed"] == 1
    assert report["workflow"]["event_counts"]["approved"] == 0
    assert report["workflow"]["event_counts"]["filled"] == 0
    assert report["workflow"]["current_state_counts"]["proposed"] == 1
    assert report["workflow"]["current_state_counts"]["filled"] == 0
    assert report["workflow"]["ever_approved_count"] == 0
    assert report["workflow"]["ever_filled_count"] == 0
    assert report["closeouts"]["terminal_count"] == 0
    assert report["closeouts"]["with_closeout_count"] == 0
    assert report["closeouts"]["resolution_counts"]["thesis_target"] == 0
    assert report["closeouts"]["realized_profit_loss"]["total_amount"] == "0"
    may = report["proposal_volume"]["by_month"]["2026-05"]
    assert may["approved_count"] == 0
    assert may["terminal_count"] == 0
    assert may["with_closeout_count"] == 0
    assert may["realized_profit_loss_amount"] == "0"


def test_windowed_report_uses_cutoff_for_approval_readiness(tmp_path: Path) -> None:
    root = tmp_path / "ideas"
    service = TradeIdeaService(
        root,
        now_factory=lambda: datetime(2026, 5, 20, 12, 0, tzinfo=UTC),
    )
    service.propose(
        _idea(
            "trade-window-cutoff-readiness",
            expires_at=datetime(2026, 6, 5, 12, 0, tzinfo=UTC),
        ),
        actor_id="idea-generator-v1",
    )

    report = build_trade_idea_track_record_report(
        service,
        now=datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        since=datetime(2026, 5, 1, 0, 0, tzinfo=UTC),
        until=datetime(2026, 5, 31, 23, 59, 59, tzinfo=UTC),
    )

    assert report["quality"]["approval_ready_count"] == 1
    assert report["quality"]["approval_policy_violation_counts"] == {}
