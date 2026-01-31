"""
Voting mechanisms for ensemble decision aggregation.

Provides different strategies for combining multiple strategy votes:
- WeightedMajorityVoting: Sum weighted votes per action
- ConfidenceLeaderVoting: Follow highest-confidence strategy on conflict
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

from gpt_trader.features.intelligence.contracts import Action, Decision

from .models import StrategyVote

if TYPE_CHECKING:
    from gpt_trader.features.intelligence.regime.models import RegimeState


class VotingMechanism(ABC):
    """Base class for vote aggregation strategies."""

    @abstractmethod
    def aggregate(
        self,
        votes: list[StrategyVote],
        regime: RegimeState,
    ) -> Decision:
        """Aggregate strategy votes into final decision.

        Args:
            votes: List of strategy votes with weights
            regime: Current regime state

        Returns:
            Aggregated decision
        """
        ...

    @staticmethod
    def create(method: str) -> VotingMechanism:
        """Factory method to create voting mechanism by name.

        Args:
            method: Voting method name

        Returns:
            VotingMechanism instance

        Raises:
            ValueError: If method is unknown
        """
        mechanisms: dict[str, type[VotingMechanism]] = {
            "weighted_majority": WeightedMajorityVoting,
            "confidence_leader": ConfidenceLeaderVoting,
        }

        if method not in mechanisms:
            raise ValueError(
                f"Unknown voting method: {method!r}. Valid options: {list(mechanisms.keys())}"
            )

        return mechanisms[method]()


class WeightedMajorityVoting(VotingMechanism):
    """Aggregate by summing weighted votes per action.

    Final action = highest weighted sum.
    Confidence = proportion of total weight for winning action.

    This is a more democratic approach where all strategies
    contribute proportionally to their weights.
    """

    def aggregate(
        self,
        votes: list[StrategyVote],
        regime: RegimeState,
    ) -> Decision:
        """Aggregate votes using weighted majority."""
        if not votes:
            return Decision(
                action=Action.HOLD,
                reason="No strategy votes received",
                confidence=0.0,
                indicators={},
            )

        # Sum weighted confidence by action
        action_scores: dict[Action, float] = defaultdict(float)
        action_reasons: dict[Action, list[str]] = defaultdict(list)

        for vote in votes:
            action = vote.decision.action
            score = vote.weighted_confidence
            action_scores[action] += score
            action_reasons[action].append(f"{vote.strategy_name}({vote.decision.confidence:.2f})")

        # Select highest scoring action
        if not action_scores:
            return Decision(
                action=Action.HOLD,
                reason="No valid votes",
                confidence=0.0,
                indicators={},
            )

        best_action = max(action_scores, key=lambda a: action_scores[a])
        total_score = sum(action_scores.values())

        # Calculate confidence as proportion of total
        confidence = action_scores[best_action] / total_score if total_score > 0 else 0.0

        # Build reason string
        reason = f"Weighted majority: {', '.join(action_reasons[best_action])}"

        return Decision(
            action=best_action,
            reason=reason,
            confidence=confidence,
            indicators={"voting_method": "weighted_majority"},
        )


class ConfidenceLeaderVoting(VotingMechanism):
    """Follow the highest-confidence individual strategy on conflict.

    This is a more active approach that trusts the strategy
    with the strongest conviction, weighted by its assigned weight.

    On conflict (strategies disagree), follows the vote with
    highest weighted_confidence (weight * confidence).
    """

    def aggregate(
        self,
        votes: list[StrategyVote],
        regime: RegimeState,
    ) -> Decision:
        """Aggregate votes using confidence leader approach."""
        if not votes:
            return Decision(
                action=Action.HOLD,
                reason="No strategy votes received",
                confidence=0.0,
                indicators={},
            )

        # Check for agreement
        unique_actions = {v.decision.action for v in votes}

        if len(unique_actions) == 1:
            # All strategies agree - combine their confidence
            action = votes[0].decision.action
            total_weighted_conf = sum(v.weighted_confidence for v in votes)
            total_weight = sum(v.weight for v in votes)

            # Normalize confidence
            avg_confidence = total_weighted_conf / total_weight if total_weight > 0 else 0.0

            reasons = [f"{v.strategy_name}({v.decision.confidence:.2f})" for v in votes]
            reason = f"Unanimous: {', '.join(reasons)}"

            return Decision(
                action=action,
                reason=reason,
                confidence=min(0.95, avg_confidence * 1.2),  # Boost for agreement
                indicators={"voting_method": "confidence_leader", "unanimous": True},
            )

        # Disagreement - follow highest weighted confidence
        leader = max(votes, key=lambda v: v.weighted_confidence)

        # Calculate confidence penalty for disagreement
        # More disagreement = lower confidence
        agreement_ratio = sum(
            1 for v in votes if v.decision.action == leader.decision.action
        ) / len(votes)

        adjusted_confidence = leader.decision.confidence * (0.7 + 0.3 * agreement_ratio)

        # Build explanation
        other_actions = [
            f"{v.strategy_name}:{v.decision.action.value}"
            for v in votes
            if v.decision.action != leader.decision.action
        ]
        reason = (
            f"Leader: {leader.strategy_name}({leader.decision.confidence:.2f})"
            f", opposed by: {', '.join(other_actions)}"
        )

        return Decision(
            action=leader.decision.action,
            reason=reason,
            confidence=adjusted_confidence,
            indicators={
                "voting_method": "confidence_leader",
                "leader": leader.strategy_name,
                "agreement_ratio": agreement_ratio,
            },
        )


__all__ = [
    "ConfidenceLeaderVoting",
    "VotingMechanism",
    "WeightedMajorityVoting",
]
