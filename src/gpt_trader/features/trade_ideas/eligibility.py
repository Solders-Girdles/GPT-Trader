"""Strategy-eligibility gate for trade ideas.

Encodes the automatic rejection conditions from the accepted framework
(docs/PRE_MIGRATION_DECISION_FRAMEWORK.md, Decision 2): ideas without an
invalidation level, max-loss estimate, reproducible data source, defined
entry, exit rule, or expiry must never reach a reviewer as actionable.
"""

from __future__ import annotations

from gpt_trader.features.trade_ideas.models import TradeIdea


def evaluate_eligibility(idea: TradeIdea) -> list[str]:
    """Return human-readable rejection reasons; an empty list means eligible."""
    reasons: list[str] = []

    if not idea.thesis.strip():
        reasons.append("Missing thesis: no plain-language reason the trade exists")
    if not idea.instrument.strip():
        reasons.append("Missing instrument: no exact symbol or product identifier")
    if not idea.invalidation.strip():
        reasons.append("Missing invalidation: no level or condition that makes the thesis false")
    if not idea.target_exit.strip():
        reasons.append("Missing target_exit: no target, time stop, or exit condition")
    if idea.max_loss.amount is None and idea.max_loss.percent_of_account is None:
        reasons.append("Missing max_loss: no dollar or percent loss estimate")
    if not idea.data_used:
        reasons.append("Missing data_used: no reproducible data sources recorded")
    if idea.time_horizon.expires_at is None:
        reasons.append("Missing expiry: no review deadline or expiration time")
    entry = idea.entry_zone
    if entry.lower is None and entry.upper is None and not entry.trigger.strip():
        reasons.append("Missing entry_zone: no price range or conditional trigger")
    if not idea.failure_mode.strip():
        reasons.append("Missing failure_mode: most likely way the trade fails is not recorded")

    return reasons


def is_eligible(idea: TradeIdea) -> bool:
    """True when the idea survives every automatic rejection condition."""
    return not evaluate_eligibility(idea)
