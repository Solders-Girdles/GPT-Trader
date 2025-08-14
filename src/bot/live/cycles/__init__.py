from __future__ import annotations

"""Cycle helpers for the production orchestrator.

These thin wrappers allow us to gradually move implementation details out of
`production_orchestrator.py` without disrupting public interfaces or tests.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only for typing
    from bot.live.production_orchestrator import ProductionOrchestrator


async def run_selection_cycle(orchestrator: ProductionOrchestrator) -> None:
    """Execute one selection cycle via the orchestrator implementation."""
    from .selection import execute_selection_cycle

    await execute_selection_cycle(orchestrator)


async def run_performance_cycle(orchestrator: ProductionOrchestrator) -> None:
    """Execute one performance monitoring cycle via the orchestrator implementation."""
    from .performance import execute_performance_cycle

    await execute_performance_cycle(orchestrator)


async def run_risk_cycle(orchestrator: ProductionOrchestrator) -> None:
    """Execute one risk monitoring cycle via the orchestrator implementation."""
    from .risk import execute_risk_cycle

    await execute_risk_cycle(orchestrator)
