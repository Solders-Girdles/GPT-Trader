"""Tests for PositionDetailModal functionality and trade helpers."""

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


class TestPositionDetailModalInit:
    def test_init_with_position(self) -> None:
        position = _make_position(quantity="1.5", mark_price="52000.00", unrealized_pnl="3000.00")

        modal = PositionDetailModal(position)

        assert modal.position == position
        assert modal.trades == []
        assert modal._linked_trades == []

    def test_init_with_trades(self) -> None:
        position = _make_position(quantity="1.5", mark_price="52000.00", unrealized_pnl="3000.00")
        trades = [
            _make_trade(trade_id="t1", quantity="1.0", fee="10.00"),
            _make_trade(trade_id="t2", symbol="ETH-USD", price="2500.00", fee="5.00"),
        ]

        modal = PositionDetailModal(position, trades=trades)

        assert modal.position == position
        assert len(modal.trades) == 2

    def test_filters_trades_by_symbol(self) -> None:
        position = _make_position()
        trades = [
            _make_trade(trade_id="t1"),
            _make_trade(trade_id="t2", symbol="ETH-USD", price="2500.00"),
            _make_trade(trade_id="t3", side="SELL", quantity="0.5", price="51000.00"),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = [t for t in trades if t.symbol == position.symbol]

        assert len(modal._linked_trades) == 2
        assert all(t.symbol == "BTC-USD" for t in modal._linked_trades)


class TestPositionDetailModalRecentFills:
    def test_get_recent_fills_ordering_and_limit(self) -> None:
        position = _make_position()
        trades = [
            _make_trade(
                trade_id="t1", quantity="0.25", price="49000.00", time="2024-01-15T08:00:00Z"
            ),
            _make_trade(
                trade_id="t2", quantity="0.25", price="50000.00", time="2024-01-15T10:00:00Z"
            ),
            _make_trade(
                trade_id="t3", quantity="0.25", price="51000.00", time="2024-01-15T12:00:00Z"
            ),
            _make_trade(
                trade_id="t4", quantity="0.25", price="52000.00", time="2024-01-15T14:00:00Z"
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
        position = _make_position()
        trade = _make_trade(
            trade_id="t1",
            quantity="0.5",
            price="49500.00",
            time="2024-01-15T14:30:45.123Z",
            fee="12.50",
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


class TestPositionDetailModalTradeClassification:
    @pytest.mark.parametrize(
        ("position_side", "open_side", "close_side"),
        [("long", "BUY", "SELL"), ("short", "SELL", "BUY")],
    )
    def test_categorize_trades(self, position_side: str, open_side: str, close_side: str) -> None:
        position = _make_position(symbol="BTC-USD", side=position_side)
        trades = [
            _make_trade(trade_id="t1", side=open_side, quantity="1.5"),
            _make_trade(trade_id="t2", side=close_side, quantity="0.5", price="51000.00"),
        ]

        modal = PositionDetailModal(position, trades=trades)
        modal._linked_trades = trades

        opens, closes = modal._categorize_trades()

        assert len(opens) == 1
        assert len(closes) == 1
        assert opens[0].side == open_side
        assert closes[0].side == close_side

    @pytest.mark.parametrize(
        ("position_side", "trade_side", "expected"),
        [
            ("long", "BUY", "OPEN"),
            ("long", "SELL", "CLOSE"),
            ("short", "SELL", "OPEN"),
            ("short", "BUY", "CLOSE"),
        ],
    )
    def test_get_trade_type(self, position_side: str, trade_side: str, expected: str) -> None:
        position = _make_position(side=position_side)
        trade = _make_trade(side=trade_side)

        modal = PositionDetailModal(position)

        assert modal._get_trade_type(trade) == expected

    @pytest.mark.parametrize(
        ("timestamp", "expected"),
        [
            ("2024-01-15T14:30:45.123Z", "14:30:45"),
            ("2024-01-15T09:05:00Z", "09:05:00"),
        ],
    )
    def test_format_trade_time(self, timestamp: str, expected: str) -> None:
        modal = PositionDetailModal(_make_position())

        assert modal._format_trade_time(timestamp) == expected
