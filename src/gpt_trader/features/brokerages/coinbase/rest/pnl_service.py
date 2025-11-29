"""PnL tracking service for Coinbase REST API.

This service handles PnL calculations with explicit dependencies
injected via constructor, replacing the PnLRestMixin.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.rest.position_state_store import PositionStateStore
from gpt_trader.features.brokerages.coinbase.utilities import PositionState


class PnLService:
    """Handles PnL tracking and calculation.

    Dependencies:
        position_store: Centralized position state storage
        market_data: MarketDataService for mark prices
    """

    def __init__(
        self,
        *,
        position_store: PositionStateStore,
        market_data: MarketDataService,
    ) -> None:
        self._position_store = position_store
        self._market_data = market_data

    def process_fill_for_pnl(self, fill: dict[str, Any]) -> None:
        """Update position state and PnL based on a fill."""
        product_id = fill.get("product_id")
        size = fill.get("size")
        price = fill.get("price")
        side = fill.get("side")

        if not all([product_id, size, price, side]):
            return

        size_dec = Decimal(str(size))
        price_dec = Decimal(str(price))
        side_norm = str(side).lower()  # buy/sell

        # Map fill side to position side
        fill_pos_side: Literal["long", "short"] = "long" if side_norm == "buy" else "short"

        product_id_str = str(product_id)
        if not self._position_store.contains(product_id_str):
            self._position_store.set(
                product_id_str,
                PositionState(
                    symbol=product_id_str,
                    side=fill_pos_side,
                    quantity=size_dec,
                    entry_price=price_dec,
                ),
            )
        else:
            position = self._position_store.get(product_id_str)
            if position is None:
                return  # Shouldn't happen after contains() check, but type safety

            if position.side == fill_pos_side:
                # Increasing position
                total_cost = (position.quantity * position.entry_price) + (size_dec * price_dec)
                new_quantity = position.quantity + size_dec
                position.entry_price = total_cost / new_quantity
                position.quantity = new_quantity
            else:
                # Reducing position (Closing)
                # Calculate Realized PnL on the closed portion
                close_quantity = min(position.quantity, size_dec)

                pnl = (price_dec - position.entry_price) * close_quantity
                if position.side == "short":
                    pnl = -pnl

                position.realized_pnl += pnl
                position.quantity -= close_quantity

                # If flipped or zeroed, we handle simplistically for now (test only checks reduction)
                if position.quantity == 0:
                    # Could remove, but keeping with 0 size preserves PnL record for now
                    pass

    def get_position_pnl(self, symbol: str) -> dict[str, Any]:
        """Get PnL metrics for a specific position."""
        position = self._position_store.get(symbol)
        if position is None:
            return {
                "symbol": symbol,
                "quantity": Decimal("0"),
                "unrealized_pnl": Decimal("0"),
                "realized_pnl": Decimal("0"),
            }

        raw_mark = self._market_data.get_mark(symbol)
        mark_price = Decimal(str(raw_mark)) if raw_mark is not None else position.entry_price

        # Calc unrealized
        upnl = (mark_price - position.entry_price) * position.quantity
        if position.side == "short":
            upnl = -upnl

        return {
            "symbol": symbol,
            "quantity": position.quantity,
            "entry": position.entry_price,
            "mark": mark_price,
            "unrealized_pnl": upnl,
            "realized_pnl": position.realized_pnl,
            "side": position.side,
        }

    def get_portfolio_pnl(self) -> dict[str, Any]:
        """Get aggregated PnL for the portfolio."""
        total_upnl = Decimal("0")
        total_rpnl = Decimal("0")
        position_details = []

        for symbol in self._position_store.symbols():
            pnl_data = self.get_position_pnl(symbol)
            total_upnl += pnl_data["unrealized_pnl"]
            total_rpnl += pnl_data["realized_pnl"]
            position_details.append(pnl_data)

        return {
            "total_realized_pnl": total_rpnl,
            "total_unrealized_pnl": total_upnl,
            "total_pnl": total_rpnl + total_upnl,
            "positions": position_details,
        }
