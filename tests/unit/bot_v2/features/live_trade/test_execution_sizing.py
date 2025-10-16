from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType
from bot_v2.features.live_trade.execution.sizing import PositionSizer
from bot_v2.features.live_trade.risk.position_sizing import PositionSizingAdvice


class FakeRiskManager:
    def __init__(self, *, enabled: bool = True) -> None:
        self.config = SimpleNamespace(
            enable_dynamic_position_sizing=enabled,
            position_sizing_method="intelligent",
            position_sizing_multiplier=1.2,
            max_leverage=2,
        )
        self.positions = {"BTC-USD": SimpleNamespace(quantity=Decimal("0.25"))}
        self.start_of_day_equity = Decimal("10000")
        self.advice = PositionSizingAdvice(
            symbol="BTC-USD",
            side="buy",
            target_notional=Decimal("6000"),
            target_quantity=Decimal("0.3"),
            used_dynamic=True,
        )
        self.context_seen = None

    def size_position(self, context):
        self.context_seen = context
        return self.advice


class FakeBroker:
    def __init__(self, *, balances=None, quote=None):
        self._balances = balances or []
        self._quote = quote

    def list_balances(self):
        return self._balances

    def get_quote(self, symbol: str):
        return self._quote


def test_maybe_apply_position_sizing_returns_advice():
    risk_manager = FakeRiskManager()
    broker = FakeBroker()
    sizer = PositionSizer(broker=broker, risk_manager=risk_manager)

    advice = sizer.maybe_apply_position_sizing(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        order_quantity=Decimal("1"),
        limit_price=None,
        product=None,
        quote=SimpleNamespace(bid=Decimal("20000"), ask=Decimal("20010"), last=Decimal("20005")),
        leverage=2,
    )

    assert advice is risk_manager.advice
    assert sizer.last_advice is advice
    assert risk_manager.context_seen is not None
    assert risk_manager.context_seen.current_price == Decimal("20010")


def test_maybe_apply_position_sizing_disabled_returns_none():
    risk_manager = FakeRiskManager(enabled=False)
    sizer = PositionSizer(broker=None, risk_manager=risk_manager)

    advice = sizer.maybe_apply_position_sizing(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        order_quantity=Decimal("1"),
        limit_price=None,
        product=None,
        quote=None,
        leverage=None,
    )

    assert advice is None


def test_determine_reference_price_fallbacks(monkeypatch):
    broker_quote = SimpleNamespace(bid=Decimal("100"), ask=Decimal("101"), last=Decimal("100.5"))
    broker = FakeBroker(quote=broker_quote)
    risk_manager = FakeRiskManager()
    sizer = PositionSizer(broker=broker, risk_manager=risk_manager)

    # Limit price takes precedence
    price = sizer.determine_reference_price(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("99"),
        quote=None,
        product=None,
    )
    assert price == Decimal("99")

    # Market order uses quote ask
    price = sizer.determine_reference_price(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        limit_price=None,
        quote=broker_quote,
        product=None,
    )
    assert price == Decimal("101")

    # Fallback to broker.get_quote
    price = sizer.determine_reference_price(
        symbol="BTC-USD",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        limit_price=None,
        quote=None,
        product=None,
    )
    assert price == Decimal("100")

    # Product mark fallback
    product = SimpleNamespace(mark_price=Decimal("123"))
    sizer_no_quote = PositionSizer(broker=FakeBroker(quote=None), risk_manager=risk_manager)
    price = sizer_no_quote.determine_reference_price(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        limit_price=None,
        quote=None,
        product=product,
    )
    assert price == Decimal("123")


def test_estimate_equity_prefers_risk_manager_then_balances():
    risk_manager = FakeRiskManager()
    sizer = PositionSizer(broker=None, risk_manager=risk_manager)
    assert sizer.estimate_equity() == Decimal("10000")

    risk_manager.start_of_day_equity = None
    broker = FakeBroker(
        balances=[SimpleNamespace(total=Decimal("3000")), SimpleNamespace(total=Decimal("2500"))]
    )
    sizer = PositionSizer(broker=broker, risk_manager=risk_manager)
    assert sizer.estimate_equity() == Decimal("5500")


def test_extract_position_quantity_handles_missing():
    risk_manager = FakeRiskManager()
    sizer = PositionSizer(broker=None, risk_manager=risk_manager)
    assert sizer._extract_position_quantity("BTC-USD") == Decimal("0.25")
    assert sizer._extract_position_quantity("ETH-USD") == Decimal("0")
