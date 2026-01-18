from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.types import Position, Trade
from gpt_trader.tui.widgets.portfolio.position_detail_modal import PositionDetailModal


class TestPositionDetailModalFillQuality:
    def test_build_fill_quality_line_no_trades(self) -> None:
        position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000.00"),
            mark_price=Decimal("50000.00"),
            unrealized_pnl=Decimal("0"),
            side="long",
        )

        modal = PositionDetailModal(position)
        result = modal._build_fill_quality_line([], "Open fills")
        assert result == "Open fills: --"

    def test_build_fill_quality_line_formats_values(self) -> None:
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
                quantity=Decimal("0.5"),
                price=Decimal("100.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
                fee=Decimal("0.50"),
            ),
            Trade(
                trade_id="t2",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.5"),
                price=Decimal("110.00"),
                order_id="o2",
                time="2024-01-15T11:00:00Z",
                fee=Decimal("0.55"),
            ),
        ]

        modal = PositionDetailModal(position)
        result = modal._build_fill_quality_line(trades, "Open fills")

        assert "Open fills (n=2)" in result
        assert "avg" in result
        assert "105" in result
        assert "range" in result
        assert "9.52%" in result
        assert "bps" in result

    def test_build_fill_quality_line_single_trade_zero_range(self) -> None:
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
                price=Decimal("100.00"),
                order_id="o1",
                time="2024-01-15T10:00:00Z",
                fee=Decimal("0.10"),
            ),
        ]

        modal = PositionDetailModal(position)
        result = modal._build_fill_quality_line(trades, "Close fills")

        assert "Close fills (n=1)" in result
        assert "range 0.00%" in result
