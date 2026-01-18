from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.types import Order, Trade
from gpt_trader.tui.widgets.portfolio.order_detail_modal import OrderDetailModal


class TestOrderDetailModal:
    def test_init_with_order(self) -> None:
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            type="LIMIT",
            time_in_force="GTC",
            creation_time=1600000000.0,
            filled_quantity=Decimal("0.5"),
            avg_fill_price=Decimal("49900.00"),
        )

        modal = OrderDetailModal(order)

        assert modal.order == order
        assert modal.trades == []
        assert modal._linked_trades == []

    def test_init_with_trades(self) -> None:
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            creation_time=1600000000.0,
        )

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("50000.00"),
                order_id="ord_123",
                time="2024-01-01T12:00:00Z",
                fee=Decimal("25.00"),
            ),
            Trade(
                trade_id="trd_2",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("50100.00"),
                order_id="ord_456",
                time="2024-01-01T12:01:00Z",
                fee=Decimal("25.05"),
            ),
        ]

        modal = OrderDetailModal(order, trades=trades)

        assert modal.order == order
        assert len(modal.trades) == 2

    def test_calculate_fill_pct(self) -> None:
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("2.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            creation_time=1600000000.0,
            filled_quantity=Decimal("0.5"),
        )

        modal = OrderDetailModal(order)
        result = modal._calculate_fill_pct()

        assert result == "25%"

    def test_calculate_total_fees(self) -> None:
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            creation_time=1600000000.0,
        )

        trades = [
            Trade(
                trade_id="trd_1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("50000.00"),
                order_id="ord_123",
                time="2024-01-01T12:00:00Z",
                fee=Decimal("25.00"),
            ),
            Trade(
                trade_id="trd_2",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("50100.00"),
                order_id="ord_123",
                time="2024-01-01T12:01:00Z",
                fee=Decimal("25.05"),
            ),
        ]

        modal = OrderDetailModal(order, trades=trades)
        modal._linked_trades = [t for t in trades if t.order_id == order.order_id]

        result = modal._calculate_total_fees()

        assert result == Decimal("50.05")

    def test_format_trade_time(self) -> None:
        order = Order(
            order_id="ord_123",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            status="OPEN",
            creation_time=1600000000.0,
        )

        modal = OrderDetailModal(order)

        assert modal._format_trade_time("2024-01-15T14:30:45.123Z") == "14:30:45"
        assert modal._format_trade_time("2024-01-15T09:05:00Z") == "09:05:00"
