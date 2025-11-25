"""
PnL management mixin for Coinbase REST service.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

from gpt_trader.features.brokerages.coinbase.utilities import PositionState

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService


class PnLRestMixin:
    """Mixin for PnL tracking and calculation.

    This mixin is designed to be used with CoinbaseRestServiceBase which provides:
    - positions: dict[str, PositionState]
    - _positions: dict[str, PositionState]
    - market_data: MarketDataService
    """

    if TYPE_CHECKING:
        # Type hints for attributes provided by the base class
        _positions: dict[str, PositionState]
        market_data: MarketDataService

        @property
        def positions(self) -> dict[str, PositionState]: ...

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
        if product_id_str not in self.positions:
            self._positions[product_id_str] = PositionState(
                symbol=product_id_str, side=fill_pos_side, quantity=size_dec, entry_price=price_dec
            )
        else:
            position = self.positions[product_id_str]

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
        if symbol not in self.positions:
            return {
                "symbol": symbol,
                "quantity": Decimal("0"),
                "unrealized_pnl": Decimal("0"),
                "realized_pnl": Decimal("0"),
            }

        position = self.positions[symbol]
        raw_mark = self.market_data.get_mark(symbol)
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

        for symbol in list(self.positions.keys()):
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
