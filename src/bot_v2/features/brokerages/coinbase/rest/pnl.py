"""PnL bookkeeping for the Coinbase REST service."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bot_v2.features.brokerages.coinbase.rest.base import logger
from bot_v2.features.brokerages.coinbase.utilities import PositionState

if TYPE_CHECKING:
    from bot_v2.features.brokerages.coinbase.rest.base import CoinbaseRestServiceBase


class PnLRestMixin:
    """Tracks realised/unrealised PnL and funding accruals."""

    def process_fill_for_pnl(self, fill: dict[str, Any]) -> None:
        base = cast("CoinbaseRestServiceBase", self)
        positions = base.positions
        symbol = fill.get("product_id")
        if not symbol:
            return
        fill_quantity = Decimal(str(fill.get("size", "0")))
        fill_price = Decimal(str(fill.get("price", "0")))
        fill_side = str(fill.get("side", "")).lower()
        if fill_quantity == 0 or fill_price == 0:
            return
        if symbol not in positions:
            position_side = "long" if fill_side == "buy" else "short"
            positions[symbol] = PositionState(
                symbol=symbol,
                side=position_side,
                quantity=fill_quantity,
                entry_price=fill_price,
            )
        else:
            position = positions[symbol]
            realized_delta = position.update_from_fill(fill_quantity, fill_price, fill_side)
            if realized_delta != 0:
                logger.info("Realized PnL for %s: %s", symbol, realized_delta)
        base.update_position_metrics(symbol)

    def get_position_pnl(self, symbol: str) -> dict[str, Any]:
        base = cast("CoinbaseRestServiceBase", self)
        positions = base.positions
        if symbol not in positions:
            return {
                "symbol": symbol,
                "quantity": Decimal("0"),
                "side": None,
                "entry": None,
                "mark": None,
                "unrealized_pnl": Decimal("0"),
                "realized_pnl": Decimal("0"),
                "funding_accrued": Decimal("0"),
            }
        position = positions[symbol]
        mark = base.market_data.get_mark(symbol) or Decimal("0")
        unrealized = position.get_unrealized_pnl(mark)
        events = base._event_store.tail(bot_id="coinbase_perps", limit=100, types=["metric"])
        funding_events = [
            e for e in events if e.get("type") == "funding" and e.get("symbol") == symbol
        ]
        total_funding = sum(Decimal(e.get("funding_amount", "0")) for e in funding_events)
        return {
            "symbol": symbol,
            "quantity": position.quantity,
            "side": position.side,
            "entry": position.entry_price,
            "mark": mark,
            "unrealized_pnl": unrealized,
            "realized_pnl": position.realized_pnl,
            "funding_accrued": total_funding,
        }

    def get_portfolio_pnl(self) -> dict[str, Any]:
        total_unrealized = Decimal("0")
        total_realized = Decimal("0")
        total_funding = Decimal("0")
        breakdown: dict[str, Any] = {}
        base = cast("CoinbaseRestServiceBase", self)
        for symbol in list(base.positions.keys()):
            pnl = self.get_position_pnl(symbol)
            breakdown[symbol] = pnl
            total_unrealized += pnl["unrealized_pnl"]
            total_realized += pnl["realized_pnl"]
            total_funding += pnl["funding_accrued"]
        return {
            "total_unrealized_pnl": total_unrealized,
            "total_realized_pnl": total_realized,
            "total_funding": total_funding,
            "total_pnl": total_unrealized + total_realized,
            "positions": breakdown,
        }


__all__ = ["PnLRestMixin"]
