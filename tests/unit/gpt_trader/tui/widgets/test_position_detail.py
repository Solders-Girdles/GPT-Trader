"""Tests for PositionDetailModal initialization and recent fills functionality."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.types import Position, Trade
from gpt_trader.tui.widgets.portfolio.position_detail_modal import PositionDetailModal


class TestPositionDetailModalInit:
    def test_init_with_position(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("52000.00"),
            unrealized_pnl=Decimal("3000.00"),
            side="long",
        )

        modal = PositionDetailModal(position)

        assert modal.position == position
        assert modal.trades == []
        assert modal._linked_trades == []

    def test_init_with_trades(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.5"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("52000.00"),
            unrealized_pnl=Decimal("3000.00"),
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
                fee=Decimal("10.00"),
            ),
            Trade(
                trade_id="t2",
                symbol="ETH-USD",
                side="BUY",
                quantity=Decimal("5.0"),
                price=Decimal("2500.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
                fee=Decimal("5.00"),
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)

        assert modal.position == position
        assert len(modal.trades) == 2

    def test_filters_trades_by_symbol(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
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
            ),
            Trade(
                trade_id="t2",
                symbol="ETH-USD",
                side="BUY",
                quantity=Decimal("5.0"),
                price=Decimal("2500.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
            ),
            Trade(
                trade_id="t3",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("51000.00"),
                order_id="o3",
                time="2024-01-15T12:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        assert len(modal._linked_trades) == 2
        assert all(t.symbol == "BTC-USD" for t in modal._linked_trades)


class TestPositionDetailModalRecentFills:
    def test_get_recent_fills_ordering_and_limit(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
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
                quantity=Decimal("0.25"),
                price=Decimal("49000.00"),
                order_id="o1",
                time="2024-01-15T08:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.25"),
                price=Decimal("50000.00"),
                order_id="o2",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t3",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.25"),
                price=Decimal("51000.00"),
                order_id="o3",
                time="2024-01-15T12:00:00Z",
            ),
            Trade(
                trade_id="t4",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.25"),
                price=Decimal("52000.00"),
                order_id="o4",
                time="2024-01-15T14:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]
        modal._linked_trades.sort(key=lambda t: t.time, reverse=True)

        recent = modal._get_recent_fills()
        assert [t.trade_id for t in recent] == ["t4", "t3", "t2"]

        recent_2 = modal._get_recent_fills(limit=2)
        assert [t.trade_id for t in recent_2] == ["t4", "t3"]

    def test_format_recent_fill_row_contains_all_fields(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        trade = Trade(
            trade_id="t1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.5"),
            price=Decimal("49500.00"),
            order_id="o1",
            time="2024-01-15T14:30:45.123Z",
            fee=Decimal("12.50"),
        )

        modal = PositionDetailModal(position)
        result = modal._format_recent_fill_row(trade)

        plain_text = result.plain

        assert "14:30:45" in plain_text
        assert "BUY" in plain_text
        assert "OPEN" in plain_text
        assert "0.5" in plain_text
        assert "49,500" in plain_text
        assert "12.50" in plain_text
