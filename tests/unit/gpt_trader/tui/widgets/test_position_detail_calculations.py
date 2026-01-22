"""Tests for PositionDetailModal calculation methods and fill quality."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.tui.types import Position, Trade
from gpt_trader.tui.widgets.portfolio.position_detail_modal import PositionDetailModal


def _make_position(
    *,
    symbol: str = "BTC-USD",
    quantity: str = "1.0",
    entry_price: str = "50000.00",
    mark_price: str = "50000.00",
    unrealized_pnl: str = "0",
    side: str = "long",
) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(quantity),
        entry_price=Decimal(entry_price),
        mark_price=Decimal(mark_price),
        unrealized_pnl=Decimal(unrealized_pnl),
        side=side,
    )


def _make_trade(
    *,
    trade_id: str = "t1",
    symbol: str = "BTC-USD",
    side: str = "BUY",
    quantity: str = "1.0",
    price: str = "50000.00",
    order_id: str = "o1",
    time: str = "2024-01-15T10:00:00Z",
    fee: str | None = None,
) -> Trade:
    payload = {
        "trade_id": trade_id,
        "symbol": symbol,
        "side": side,
        "quantity": Decimal(quantity),
        "price": Decimal(price),
        "order_id": order_id,
        "time": time,
    }
    if fee is not None:
        payload["fee"] = Decimal(fee)
    return Trade(**payload)


class TestPositionDetailModalCalculations:
    def test_calculate_total_fees(self) -> None:
        position = _make_position(quantity="2.0")
        trades = [
            _make_trade(trade_id="t1", fee="10.50"),
            _make_trade(trade_id="t2", fee="15.25"),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        result = modal._calculate_total_fees()

        assert result == Decimal("25.75")

    @pytest.mark.parametrize(
        ("position", "trades", "expected"),
        [
            (
                _make_position(
                    symbol="BTC-USD",
                    quantity="0.5",
                    mark_price="52000.00",
                    unrealized_pnl="1000.00",
                    side="long",
                ),
                [
                    _make_trade(trade_id="t1", side="BUY", quantity="1.0"),
                    _make_trade(trade_id="t2", side="SELL", quantity="0.5", price="55000.00"),
                ],
                Decimal("2500.00"),
            ),
            (
                _make_position(
                    symbol="ETH-USD",
                    quantity="2.0",
                    entry_price="3000.00",
                    mark_price="2800.00",
                    unrealized_pnl="400.00",
                    side="short",
                ),
                [
                    _make_trade(
                        trade_id="t1",
                        symbol="ETH-USD",
                        side="SELL",
                        quantity="3.0",
                        price="3000.00",
                    ),
                    _make_trade(
                        trade_id="t2",
                        symbol="ETH-USD",
                        side="BUY",
                        quantity="1.0",
                        price="2700.00",
                    ),
                ],
                Decimal("300.00"),
            ),
        ],
    )
    def test_calculate_realized_pnl(
        self, position: Position, trades: list[Trade], expected: Decimal
    ) -> None:
        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        result = modal._calculate_realized_pnl()

        assert result == expected


class TestPositionDetailModalFillQuality:
    def test_build_fill_quality_line_no_trades(self) -> None:
        position = _make_position()

        modal = PositionDetailModal(position)
        result = modal._build_fill_quality_line([], "Open fills")
        assert result == "Open fills: --"

    def test_build_fill_quality_line_formats_values(self) -> None:
        position = _make_position()
        trades = [
            _make_trade(trade_id="t1", quantity="0.5", price="100.00", fee="0.50"),
            _make_trade(trade_id="t2", quantity="0.5", price="110.00", fee="0.55"),
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
        position = _make_position()
        trades = [_make_trade(trade_id="t1", quantity="1.0", price="100.00", fee="0.10")]

        modal = PositionDetailModal(position)
        result = modal._build_fill_quality_line(trades, "Close fills")

        assert "Close fills (n=1)" in result
        assert "range 0.00%" in result
