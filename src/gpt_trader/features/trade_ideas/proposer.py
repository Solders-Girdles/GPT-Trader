"""Proposer protocol: snapshot in, complete trade-idea records out.

Every proposer — deterministic baseline or future LLM-backed — implements the
same contract, so all of them can be replayed over historical snapshots and
scored against each other on identical inputs. Proposers never see "the
present", only a :class:`MarketSnapshot`, and they never submit anything; their
output enters the workflow through ``TradeIdeaService.propose``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gpt_trader.features.trade_ideas.models import TradeIdea
from gpt_trader.features.trade_ideas.snapshot import MarketSnapshot


@runtime_checkable
class Proposer(Protocol):
    """Generates eligible trade-idea records from a point-in-time snapshot."""

    @property
    def proposer_id(self) -> str:
        """Stable actor identifier recorded on every proposed idea."""
        ...

    def propose(self, snapshot: MarketSnapshot) -> list[TradeIdea]:
        """Return zero or more complete, eligibility-passing records."""
        ...
