from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class Order:
    symbol: str
    side: str  # "BUY" | "SELL"
    qty: int
    price: float
    ts: datetime
    reason: str = ""


@dataclass
class Fill:
    order: Order
    qty: int
    price: float
    ts: datetime
    cost: float = 0.0  # USD cost for this fill


@dataclass
class Position:
    symbol: str
    qty: int = 0
    avg_price: float = 0.0
    entry_ts: datetime | None = None
    realized_pnl: float = 0.0
    costs: float = 0.0  # accumulated transaction costs


@dataclass
class Trade:
    symbol: str
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    qty: int
    pnl: float
    rtn: float
    bars_held: int
    reason_exit: str = ""


class Ledger:
    """
    Minimal, long-only ledger.
    - Average price for adds.
    - First-in-first-out close (since we track one net position).
    - A Trade is realized when qty goes to 0.
    """

    def __init__(self) -> None:
        self.orders: list[Order] = []
        self.fills: list[Fill] = []
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []

    def get_pos(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def submit_and_fill(
        self,
        symbol: str,
        new_qty: int,
        price: float,
        ts: datetime,
        reason: str,
        cost_usd: float,
    ) -> None:
        """
        Adjust position to new_qty at given price (one synthetic MKT fill).
        Positive qty => long; zero => flat. (No shorts in Phase A.)
        """
        pos = self.get_pos(symbol)
        delta = new_qty - pos.qty
        if delta == 0:
            return

        side = "BUY" if delta > 0 else "SELL"
        ord_ = Order(symbol=symbol, side=side, qty=abs(delta), price=price, ts=ts, reason=reason)
        self.orders.append(ord_)
        fill = Fill(order=ord_, qty=abs(delta), price=price, ts=ts, cost=cost_usd)
        self.fills.append(fill)

        # Apply cost immediately to realized (conservative)
        pos.costs += cost_usd

        if delta > 0:
            # Add to position: average price
            new_notional = pos.avg_price * pos.qty + price * delta
            pos.qty += delta
            pos.avg_price = new_notional / max(pos.qty, 1)
            if pos.entry_ts is None:
                pos.entry_ts = ts
        else:
            # Reduce/close position
            sell_qty = min(abs(delta), pos.qty)
            realized = (price - pos.avg_price) * sell_qty
            pos.realized_pnl += realized
            pos.qty -= sell_qty
            if pos.qty == 0:
                # Close trade
                if pos.entry_ts is not None:
                    bars_held = max((ts.date() - pos.entry_ts.date()).days, 0)
                    total_pnl = pos.realized_pnl - pos.costs
                    rtn = 0.0 if pos.avg_price == 0 else total_pnl / (pos.avg_price * sell_qty)
                    self.trades.append(
                        Trade(
                            symbol=symbol,
                            entry_ts=pos.entry_ts,
                            entry_price=pos.avg_price,
                            exit_ts=ts,
                            exit_price=price,
                            qty=sell_qty,
                            pnl=total_pnl,
                            rtn=rtn,
                            bars_held=bars_held,
                            reason_exit=reason,
                        )
                    )
                # Reset container
                self.positions[symbol] = Position(symbol=symbol)

    def to_trades_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "entry_ts",
                    "entry_price",
                    "exit_ts",
                    "exit_price",
                    "qty",
                    "pnl",
                    "rtn",
                    "bars_held",
                    "reason_exit",
                ]
            )
        rows = [
            {
                "symbol": t.symbol,
                "entry_ts": t.entry_ts,
                "entry_price": t.entry_price,
                "exit_ts": t.exit_ts,
                "exit_price": t.exit_price,
                "qty": t.qty,
                "pnl": t.pnl,
                "rtn": t.rtn,
                "bars_held": t.bars_held,
                "reason_exit": t.reason_exit,
            }
            for t in self.trades
        ]
        return pd.DataFrame(rows)
