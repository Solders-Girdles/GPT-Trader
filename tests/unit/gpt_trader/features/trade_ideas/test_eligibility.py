from __future__ import annotations

from decimal import Decimal

from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.features.trade_ideas import (
    EntryZone,
    MaxLoss,
    TimeHorizon,
    TradeIdea,
    evaluate_eligibility,
    is_eligible,
)


def test_fully_specified_idea_is_eligible(trade_idea: TradeIdea) -> None:
    assert evaluate_eligibility(trade_idea) == []
    assert is_eligible(trade_idea)


def test_missing_invalidation_is_rejected() -> None:
    idea = build_trade_idea(invalidation="   ")

    reasons = evaluate_eligibility(idea)

    assert any("invalidation" in reason for reason in reasons)
    assert not is_eligible(idea)


def test_missing_max_loss_is_rejected() -> None:
    idea = build_trade_idea(max_loss=MaxLoss())

    assert any("max_loss" in reason for reason in evaluate_eligibility(idea))


def test_partial_max_loss_is_accepted() -> None:
    only_percent = build_trade_idea(max_loss=MaxLoss(percent_of_account=Decimal("1.5")))
    only_amount = build_trade_idea(max_loss=MaxLoss(amount=Decimal("250")))

    assert is_eligible(only_percent)
    assert is_eligible(only_amount)


def test_missing_data_sources_is_rejected() -> None:
    idea = build_trade_idea(data_used=())

    assert any("data_used" in reason for reason in evaluate_eligibility(idea))


def test_missing_expiry_is_rejected() -> None:
    idea = build_trade_idea(time_horizon=TimeHorizon(expected_hold="3 days", expires_at=None))

    assert any("expiry" in reason for reason in evaluate_eligibility(idea))


def test_missing_entry_zone_is_rejected() -> None:
    idea = build_trade_idea(entry_zone=EntryZone())

    assert any("entry_zone" in reason for reason in evaluate_eligibility(idea))


def test_trigger_only_entry_zone_is_accepted() -> None:
    idea = build_trade_idea(entry_zone=EntryZone(trigger="4h close above 61000"))

    assert is_eligible(idea)


def test_multiple_gaps_report_every_reason() -> None:
    idea = build_trade_idea(
        invalidation="",
        target_exit="",
        max_loss=MaxLoss(),
        data_used=(),
        failure_mode="",
    )

    reasons = evaluate_eligibility(idea)

    assert len(reasons) == 5
