from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    DEFAULT_RISK_BUDGET,
    ActorType,
    MaxLoss,
    PolicyViolationError,
    RiskBudget,
    TradeIdeaState,
)
from gpt_trader.features.trade_ideas.service import TradeIdeaService


@pytest.fixture
def service(tmp_path: Path) -> TradeIdeaService:
    return TradeIdeaService(
        tmp_path / "trade_ideas",
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )


def test_open_notional_budget_uses_existing_exposure_equity_before_candidate(
    service: TradeIdeaService,
) -> None:
    first = build_trade_idea(
        decision_id="trade-20260612-open-small-equity",
        max_loss=MaxLoss(amount=Decimal("100"), percent_of_account=Decimal("1")),
    )
    service.propose(first, actor_id="idea-generator-v1")
    service.approve(first.decision_id, actor_id="rj", reason="Risk verified")
    strict_budget = RiskBudget.from_dict(
        {
            **DEFAULT_RISK_BUDGET.to_dict(),
            "version": 2,
            "max_open_notional_pct": "100",
        }
    )
    service.update_budget(strict_budget, actor_type=ActorType.HUMAN, actor_id="rj")
    candidate = build_trade_idea(
        decision_id="trade-20260612-candidate-large-equity",
        max_loss=MaxLoss(amount=Decimal("1000"), percent_of_account=Decimal("1")),
    )
    service.propose(candidate, actor_id="idea-generator-v1")

    with pytest.raises(PolicyViolationError) as exc_info:
        service.approve(candidate.decision_id, actor_id="rj", reason="Risk verified")

    assert any(
        "projected open notional 121.5% exceeds limit 100%" in violation
        for violation in exc_info.value.violations
    )
    assert service.get(candidate.decision_id).state is TradeIdeaState.PROPOSED
