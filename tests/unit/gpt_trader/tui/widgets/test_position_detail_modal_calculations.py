from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.types import Position, Trade
from gpt_trader.tui.widgets.portfolio.position_detail_modal import PositionDetailModal


class TestPositionDetailModalCalculations:
    def test_calculate_total_fees(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("2.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        trades = [
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
                fee=Decimal("10.50"),
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
                fee=Decimal("15.25"),
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        result = modal._calculate_total_fees()

        assert result == Decimal("25.75")

    def test_calculate_realized_pnl_long_position(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("52000.00"),
            unrealized_pnl=Decimal("1000.00"),
            side="long",
        )

        trades = [
            Trade(
                trade_id="t1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("55000.00"),
                order_id="o2",
                time="2024-01-15T12:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        result = modal._calculate_realized_pnl()

        assert result == Decimal("2500.00")

    def test_calculate_realized_pnl_short_position(self) -> None:
        position = Position(
            symbol="ETH-USD",
            quantity=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
            mark_price=Decimal("2800.00"),
            unrealized_pnl=Decimal("400.00"),
            side="short",
        )

        trades = [
            Trade(
                trade_id="t1",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("3.0"),
                price=Decimal("3000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="ETH-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("2700.00"),
                order_id="o2",
                time="2024-01-15T12:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        result = modal._calculate_realized_pnl()

        assert result == Decimal("300.00")
