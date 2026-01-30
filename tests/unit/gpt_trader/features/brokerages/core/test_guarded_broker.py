from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.brokerages.core.guarded_broker import (
    GuardedBroker,
    OrderGuardBypassError,
    bypass_order_guard,
    guarded_order_context,
)


class StubBroker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object, Decimal | None, dict[str, object]]] = []

    def place_order(
        self,
        symbol: str,
        side: object = None,
        order_type: object = None,
        quantity: Decimal | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append((symbol, side, order_type, quantity, dict(kwargs)))
        return {"order_id": "order-1"}

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_product(self, symbol: str) -> object | None:
        return None

    def get_quote(self, symbol: str) -> object | None:
        return None

    def get_ticker(self, product_id: str) -> dict[str, object]:
        return {}

    def list_positions(self) -> list[object]:
        return []

    def list_balances(self) -> list[object]:
        return []

    def get_candles(self, symbol: str, **kwargs: object) -> list[object]:
        return []


def test_guarded_broker_blocks_direct_order() -> None:
    broker = StubBroker()
    guarded = GuardedBroker(broker, strict=True)

    with pytest.raises(OrderGuardBypassError):
        guarded.place_order("BTC-USD", side="BUY", order_type="MARKET")

    assert broker.calls == []


def test_guarded_broker_allows_guarded_context() -> None:
    broker = StubBroker()
    guarded = GuardedBroker(broker, strict=True)

    with guarded_order_context("unit_test"):
        guarded.place_order("BTC-USD", side="BUY", order_type="MARKET")

    assert len(broker.calls) == 1


def test_guarded_broker_allows_bypass_context() -> None:
    broker = StubBroker()
    guarded = GuardedBroker(broker, strict=True)

    with bypass_order_guard("emergency"):
        guarded.place_order("BTC-USD", side="SELL", order_type="MARKET")

    assert len(broker.calls) == 1
