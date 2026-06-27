from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    Confidence,
    ConfidenceLabel,
    MaxLoss,
    TradeDirection,
    TradeIdeaListQuery,
    TradeIdeaListSortKey,
    TradeIdeaService,
)


@pytest.fixture
def service(tmp_path: Path) -> TradeIdeaService:
    return TradeIdeaService(
        tmp_path / "trade_ideas",
        now_factory=lambda: datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
    )


def test_list_view_result_filters_sorts_and_paginates(service: TradeIdeaService) -> None:
    low_loss = build_trade_idea(
        decision_id="trade-20260612-btc-low",
        instrument="BTC-USD",
        confidence=Confidence(label=ConfidenceLabel.MEDIUM, rationale="Constructive setup"),
        max_loss=MaxLoss(amount=Decimal("150"), percent_of_account=Decimal("1")),
    )
    high_loss = build_trade_idea(
        decision_id="trade-20260612-btc-high",
        instrument="BTC-USD",
        confidence=Confidence(label=ConfidenceLabel.HIGH, rationale="Strong confirmation"),
        max_loss=MaxLoss(amount=Decimal("400"), percent_of_account=Decimal("4")),
    )
    other = build_trade_idea(
        decision_id="trade-20260612-eth-short",
        instrument="ETH-USD",
        direction=TradeDirection.SHORT,
        confidence=Confidence(label=ConfidenceLabel.LOW, rationale="Weak confirmation"),
        max_loss=MaxLoss(amount=Decimal("100"), percent_of_account=Decimal("0.5")),
    )
    service.propose(low_loss, actor_id="idea-generator-v1")
    service.propose(high_loss, actor_id="idea-generator-v1")
    service.propose(other, actor_id="idea-generator-v1")

    result = service.list_view_result(
        TradeIdeaListQuery(
            instrument="btc-usd",
            direction=TradeDirection.LONG,
            min_confidence=ConfidenceLabel.MEDIUM,
            sort_by=TradeIdeaListSortKey.MAX_LOSS_PCT,
            descending=True,
            limit=1,
            offset=0,
        )
    )

    assert result.total_count == 2
    assert result.returned_count == 1
    assert result.has_more is True
    assert [view.idea.decision_id for view in result.views] == ["trade-20260612-btc-high"]
