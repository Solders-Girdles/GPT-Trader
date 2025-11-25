"""Spot-market specific filters for strategy decisions.

Note: The filter implementations have been removed as they were stubs.
This mixin now provides a pass-through implementation.
"""

from __future__ import annotations

from gpt_trader.features.live_trade.strategies.perps_baseline import Decision

from .models import SymbolProcessingContext


class SpotFiltersMixin:
    """Apply spot-specific filters to strategy decisions.

    Currently a no-op pass-through - filter implementations were removed.
    """

    async def _apply_spot_filters(
        self, context: SymbolProcessingContext, decision: Decision
    ) -> Decision:
        """Pass through decisions unchanged (filter stubs removed)."""
        return decision
