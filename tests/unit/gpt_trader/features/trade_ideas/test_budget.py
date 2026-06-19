from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from gpt_trader.features.trade_ideas import (
    DEFAULT_RISK_BUDGET,
    ActorType,
    BudgetIntegrityError,
    BudgetLogEntry,
    RiskBudget,
    RiskBudgetLog,
)


def build_entry(budget: RiskBudget, minute: int = 0) -> BudgetLogEntry:
    return BudgetLogEntry(
        timestamp=datetime(2026, 6, 12, 9, minute, tzinfo=UTC),
        actor_type=ActorType.HUMAN,
        actor_id="rj",
        budget=budget,
    )


@pytest.fixture
def budget_log(tmp_path: Path) -> RiskBudgetLog:
    return RiskBudgetLog(tmp_path / "risk_budget.jsonl")


def test_seeded_defaults_reflect_accepted_risk_philosophy() -> None:
    assert DEFAULT_RISK_BUDGET.version == 1
    assert DEFAULT_RISK_BUDGET.max_loss_per_idea_pct == Decimal("5")
    assert DEFAULT_RISK_BUDGET.max_daily_loss_pct == Decimal("10")
    assert DEFAULT_RISK_BUDGET.gain_retention_floor_pct == Decimal("50")
    assert DEFAULT_RISK_BUDGET.sizing_capped_by_budget is True
    assert DEFAULT_RISK_BUDGET.allow_futures_leverage is False


def test_budget_round_trip() -> None:
    restored = RiskBudget.from_dict(DEFAULT_RISK_BUDGET.to_dict())

    assert restored == DEFAULT_RISK_BUDGET
    assert isinstance(restored.max_loss_per_idea_pct, Decimal)


def test_empty_log_has_no_current_budget(budget_log: RiskBudgetLog) -> None:
    assert budget_log.current() is None
    assert budget_log.history() == []


def test_append_and_current(budget_log: RiskBudgetLog) -> None:
    budget_log.append(build_entry(DEFAULT_RISK_BUDGET))

    assert budget_log.current() == DEFAULT_RISK_BUDGET
    assert len(budget_log.history()) == 1


def test_versions_must_be_contiguous(budget_log: RiskBudgetLog) -> None:
    budget_log.append(build_entry(DEFAULT_RISK_BUDGET))
    skipped = RiskBudget.from_dict({**DEFAULT_RISK_BUDGET.to_dict(), "version": 3})

    with pytest.raises(BudgetIntegrityError):
        budget_log.append(build_entry(skipped, minute=1))


def test_renegotiated_budget_becomes_current(budget_log: RiskBudgetLog) -> None:
    budget_log.append(build_entry(DEFAULT_RISK_BUDGET))
    widened = RiskBudget.from_dict(
        {
            **DEFAULT_RISK_BUDGET.to_dict(),
            "version": 2,
            "max_loss_per_idea_pct": "8",
            "reason": "Earned after 90 days of accurate max-loss estimates",
        }
    )

    budget_log.append(build_entry(widened, minute=1))

    assert budget_log.current() == widened
    assert [entry.budget.version for entry in budget_log.history()] == [1, 2]
