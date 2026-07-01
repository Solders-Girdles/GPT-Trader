from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from gpt_trader.features.trade_ideas import (
    DEFAULT_RISK_BUDGET,
    ActorType,
    AuditAction,
    AuditEvent,
    BudgetLogEntry,
    RiskBudgetLog,
    TradeIdeaAuditLog,
    TradeIdeaState,
    new_event_id,
)
from gpt_trader.preflight.checks.trade_ideas import check_trade_ideas_readiness
from gpt_trader.preflight.core import PreflightCheck
from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea


def _seed_budget(ideas_root: Path) -> None:
    RiskBudgetLog(ideas_root / "risk_budget.jsonl").append(
        BudgetLogEntry(
            timestamp=datetime(2026, 6, 12, 9, 0, tzinfo=UTC),
            actor_type=ActorType.SYSTEM,
            actor_id="seed-defaults",
            budget=DEFAULT_RISK_BUDGET,
        )
    )


def _append_proposed_event(ideas_root: Path) -> None:
    idea = build_trade_idea()
    TradeIdeaAuditLog(ideas_root / "audit.jsonl").append(
        AuditEvent(
            event_id=new_event_id(),
            timestamp=datetime(2026, 6, 12, 10, 0, tzinfo=UTC),
            decision_id=idea.decision_id,
            actor_type=ActorType.AI,
            actor_id="idea-generator-v1",
            action=AuditAction.PROPOSED,
            before_state=None,
            after_state=TradeIdeaState.PROPOSED,
            reason="test proposal",
            record_hash=idea.record_hash(),
        )
    )


def test_trade_ideas_readiness_passes_with_seeded_budget_and_empty_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ideas_root = tmp_path / "trade_ideas"
    _seed_budget(ideas_root)
    monkeypatch.setenv("GPT_TRADER_IDEAS_ROOT", str(ideas_root))
    monkeypatch.setenv("MOCK_BROKER", "1")
    monkeypatch.setenv("DRY_RUN", "1")

    checker = PreflightCheck(profile="dev")

    assert check_trade_ideas_readiness(checker) is True
    assert any(str(ideas_root) in message for message in checker.successes)
    assert any("0 event(s)" in message for message in checker.successes)
    assert any("risk budget current" in message for message in checker.successes)
    assert not checker.errors


def test_trade_ideas_readiness_fails_on_corrupted_audit_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ideas_root = tmp_path / "trade_ideas"
    _seed_budget(ideas_root)
    (ideas_root / "audit.jsonl").write_text("{not json}\n", encoding="utf-8")
    monkeypatch.setenv("GPT_TRADER_IDEAS_ROOT", str(ideas_root))

    checker = PreflightCheck(profile="dev")

    assert check_trade_ideas_readiness(checker) is False
    assert any("audit integrity failed" in message for message in checker.errors)


def test_trade_ideas_readiness_fails_when_budget_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ideas_root = tmp_path / "trade_ideas"
    ideas_root.mkdir()
    monkeypatch.setenv("GPT_TRADER_IDEAS_ROOT", str(ideas_root))

    checker = PreflightCheck(profile="dev")

    assert check_trade_ideas_readiness(checker) is False
    assert any("risk budget not seeded" in message for message in checker.errors)


def test_trade_ideas_readiness_fails_on_malformed_budget_decimal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ideas_root = tmp_path / "trade_ideas"
    _seed_budget(ideas_root)
    budget_path = ideas_root / "risk_budget.jsonl"
    budget_payload = budget_path.read_text(encoding="utf-8").replace(
        '"max_loss_per_idea_pct":"5"',
        '"max_loss_per_idea_pct":"bad"',
    )
    budget_path.write_text(budget_payload, encoding="utf-8")
    monkeypatch.setenv("GPT_TRADER_IDEAS_ROOT", str(ideas_root))

    checker = PreflightCheck(profile="dev")

    assert check_trade_ideas_readiness(checker) is False
    assert any("risk budget unreadable" in message for message in checker.errors)


def test_trade_ideas_readiness_reports_pending_proposed_ideas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ideas_root = tmp_path / "trade_ideas"
    _seed_budget(ideas_root)
    _append_proposed_event(ideas_root)
    monkeypatch.setenv("GPT_TRADER_IDEAS_ROOT", str(ideas_root))

    checker = PreflightCheck(profile="dev")

    assert check_trade_ideas_readiness(checker) is True
    assert any("pending review: 1 proposed" in message for message in checker.warnings)
