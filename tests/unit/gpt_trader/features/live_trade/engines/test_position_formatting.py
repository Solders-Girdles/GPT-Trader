"""Unit tests for the pure position-formatting helpers.

Exercises the logic extracted from TradingEngine into
``engines/position_formatting.py`` directly, using a duck-typed position stub
so the tests stay independent of the concrete ``Position`` constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.engines.position_formatting import (
    build_position_state,
    positions_to_risk_format,
    positions_to_status_format,
    resolve_close_order,
)


@dataclass
class _Pos:
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    side: str
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")


def _pos(**kw: object) -> _Pos:
    base = dict(
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        mark_price=Decimal("110"),
        side="long",
    )
    base.update(kw)
    return _Pos(**base)  # type: ignore[arg-type]


class TestBuildPositionState:
    def test_present_symbol(self) -> None:
        positions = {"BTC-USD": _pos(quantity=Decimal("2"), entry_price=Decimal("50"))}
        assert build_position_state("BTC-USD", positions) == {
            "quantity": Decimal("2"),
            "entry_price": Decimal("50"),
            "side": "long",
        }

    def test_missing_symbol_returns_none(self) -> None:
        assert build_position_state("ETH-USD", {"BTC-USD": _pos()}) is None


class TestResolveCloseOrder:
    def test_long_closes_with_sell(self) -> None:
        assert resolve_close_order({"quantity": Decimal("0.75"), "side": "long"}) == (
            OrderSide.SELL,
            Decimal("0.75"),
        )

    def test_short_closes_with_buy(self) -> None:
        assert resolve_close_order({"quantity": Decimal("0.75"), "side": "short"}) == (
            OrderSide.BUY,
            Decimal("0.75"),
        )

    def test_infers_sell_from_positive_quantity_when_side_missing(self) -> None:
        assert resolve_close_order({"quantity": Decimal("0.75")}) == (
            OrderSide.SELL,
            Decimal("0.75"),
        )

    def test_infers_buy_from_negative_quantity_when_side_missing(self) -> None:
        assert resolve_close_order({"quantity": Decimal("-0.75")}) == (
            OrderSide.BUY,
            Decimal("0.75"),
        )

    def test_zero_quantity_returns_none(self) -> None:
        assert resolve_close_order({"quantity": Decimal("0")}) is None

    def test_invalid_quantity_returns_none(self) -> None:
        assert resolve_close_order({"quantity": "not-a-number"}) is None


class TestPositionsToFormats:
    def test_risk_format(self) -> None:
        positions = {"BTC-USD": _pos(quantity=Decimal("2"), mark_price=Decimal("99"))}
        assert positions_to_risk_format(positions) == {
            "BTC-USD": {"quantity": Decimal("2"), "mark": Decimal("99")}
        }

    def test_status_format_stringifies_all_fields(self) -> None:
        positions = {
            "BTC-USD": _pos(
                quantity=Decimal("2"),
                mark_price=Decimal("99"),
                entry_price=Decimal("100"),
                unrealized_pnl=Decimal("-2"),
                realized_pnl=Decimal("5"),
                side="short",
            )
        }
        assert positions_to_status_format(positions) == {
            "BTC-USD": {
                "quantity": "2",
                "mark_price": "99",
                "entry_price": "100",
                "unrealized_pnl": "-2",
                "realized_pnl": "5",
                "side": "short",
            }
        }
