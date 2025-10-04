"""
Position Valuation Component.

Handles single-position valuation with mark prices, staleness detection,
and PnL tracker integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot_v2.features.brokerages.core.interfaces import Position
    from bot_v2.features.live_trade.pnl_tracker import PnLTracker
    from bot_v2.features.live_trade.portfolio_valuation import MarkDataSource

logger = logging.getLogger(__name__)


@dataclass
class PositionValuation:
    """Complete valuation result for a single position."""

    symbol: str
    side: str
    quantity: Decimal
    mark_price: Decimal
    notional_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    funding_paid: Decimal
    avg_entry_price: Decimal
    is_stale: bool

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "side": self.side,
            "quantity": self.quantity,
            "mark_price": self.mark_price,
            "notional_value": self.notional_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "funding_paid": self.funding_paid,
            "avg_entry_price": self.avg_entry_price,
            "is_stale": self.is_stale,
        }


class PositionValuer:
    """
    Single-position valuation with mark integration.

    Stateless helper that values positions using current marks,
    detects staleness, and integrates with PnL tracker.
    """

    @staticmethod
    def value_position(
        symbol: str,
        position: Position,
        mark_source: MarkDataSource,
        pnl_tracker: PnLTracker,
    ) -> PositionValuation | None:
        """
        Value a single position.

        Args:
            symbol: Position symbol
            position: Position data from account
            mark_source: Mark price source
            pnl_tracker: PnL tracker for unrealized PnL

        Returns:
            PositionValuation or None if position should be skipped
            (zero quantity or missing mark)
        """
        # Skip zero quantity positions
        if position.quantity == 0:
            return None

        # Get mark price
        mark_data = mark_source.get_mark(symbol)
        if not mark_data:
            logger.warning(f"No mark price for position {symbol}")
            return None

        mark_price, is_stale = mark_data

        # Calculate notional value
        notional_value = abs(position.quantity) * mark_price

        # Get position state from PnL tracker
        pnl_position = pnl_tracker.get_or_create_position(symbol)
        pnl_position.update_mark(mark_price)

        # Determine side
        side = "long" if position.quantity > 0 else "short"

        return PositionValuation(
            symbol=symbol,
            side=side,
            quantity=position.quantity,
            mark_price=mark_price,
            notional_value=notional_value,
            unrealized_pnl=pnl_position.unrealized_pnl,
            realized_pnl=pnl_position.realized_pnl,
            funding_paid=pnl_position.funding_paid,
            avg_entry_price=pnl_position.avg_entry_price,
            is_stale=is_stale,
        )

    @staticmethod
    def value_positions(
        positions: dict[str, Position],
        mark_source: MarkDataSource,
        pnl_tracker: PnLTracker,
    ) -> tuple[dict[str, dict], Decimal, set[str], set[str]]:
        """
        Value multiple positions.

        Args:
            positions: Map of symbol to Position
            mark_source: Mark price source
            pnl_tracker: PnL tracker

        Returns:
            Tuple of:
            - position_details: Dict of symbol to position valuation dict
            - total_positions_value: Sum of all notional values
            - stale_marks: Set of symbols with stale marks
            - missing_positions: Set of symbols missing marks
        """
        position_details = {}
        total_positions_value = Decimal("0")
        stale_marks = set()
        missing_positions = set()

        for symbol, position in positions.items():
            valuation = PositionValuer.value_position(symbol, position, mark_source, pnl_tracker)

            if valuation is None:
                # Check if it was missing mark (not just zero quantity)
                if position.quantity != 0:
                    missing_positions.add(symbol)
                continue

            # Track stale marks
            if valuation.is_stale:
                stale_marks.add(symbol)

            # Accumulate total value
            total_positions_value += valuation.notional_value

            # Store details
            position_details[symbol] = valuation.to_dict()

        return position_details, total_positions_value, stale_marks, missing_positions
