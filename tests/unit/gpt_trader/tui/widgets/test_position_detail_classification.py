from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.types import Position, Trade
from gpt_trader.tui.widgets.portfolio.position_detail_modal import PositionDetailModal


class TestPositionDetailModalTradeClassification:
    def test_categorize_trades_long(self) -> None:
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
                quantity=Decimal("1.5"),
                price=Decimal("50000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("51000.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = trades

        opens, closes = modal._categorize_trades()

        assert len(opens) == 1
        assert len(closes) == 1
        assert opens[0].side == "BUY"
        assert closes[0].side == "SELL"

    def test_categorize_trades_short(self) -> None:
        position = Position(
            symbol="ETH-USD",
            quantity=Decimal("2.0"),
            entry_price=Decimal("3000.00"),
            mark_price=Decimal("3000.00"),
            unrealized_pnl=Decimal("0"),
            side="short",
        )

        trades = [
            Trade(
                trade_id="t1",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("2.0"),
                price=Decimal("3000.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
            ),
            Trade(
                trade_id="t2",
                symbol="ETH-USD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("2900.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
            ),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = trades

        opens, closes = modal._categorize_trades()

        assert len(opens) == 1
        assert len(closes) == 1
        assert opens[0].side == "SELL"
        assert closes[0].side == "BUY"

    def test_get_trade_type_long_position(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        buy_trade = Trade(
            trade_id="t1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            order_id="o1",
            time="2024-01-15T10:00:00Z",
        )

        sell_trade = Trade(
            trade_id="t2",
            symbol="BTC-USD",
            side="SELL",
            quantity=Decimal("1.0"),
            price=Decimal("51000.00"),
            order_id="o2",
            time="2024-01-15T11:00:00Z",
        )

        modal = PositionDetailModal(position)

        assert modal._get_trade_type(buy_trade) == "OPEN"
        assert modal._get_trade_type(sell_trade) == "CLOSE"

    def test_get_trade_type_short_position(self) -> None:
        position = Position(
            symbol="ETH-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("3000.00"),
            mark_price=Decimal("3000.00"),
            unrealized_pnl=Decimal("0"),
            side="short",
        )

        sell_trade = Trade(
            trade_id="t1",
            symbol="ETH-USD",
            side="SELL",
            quantity=Decimal("1.0"),
            price=Decimal("3000.00"),
            order_id="o1",
            time="2024-01-15T10:00:00Z",
        )

        buy_trade = Trade(
            trade_id="t2",
            symbol="ETH-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("2900.00"),
            order_id="o2",
            time="2024-01-15T11:00:00Z",
        )

        modal = PositionDetailModal(position)

        assert modal._get_trade_type(sell_trade) == "OPEN"
        assert modal._get_trade_type(buy_trade) == "CLOSE"

    def test_format_trade_time(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        modal = PositionDetailModal(position)

        assert modal._format_trade_time("2024-01-15T14:30:45.123Z") == "14:30:45"
        assert modal._format_trade_time("2024-01-15T09:05:00Z") == "09:05:00"
