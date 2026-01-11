"""Edge coverage for PositionStateStore behavior."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.brokerages.coinbase.rest.position_state_store import (
    PositionStateStore,
)
from gpt_trader.features.brokerages.coinbase.utilities import PositionState


def _make_state(symbol: str, side: str) -> PositionState:
    return PositionState(
        symbol=symbol,
        side=side,
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
    )


def test_set_get_contains_and_len() -> None:
    store = PositionStateStore()
    btc = _make_state("BTC-USD", "long")
    eth = _make_state("ETH-USD", "short")

    store.set("BTC-USD", btc)
    store.set("ETH-USD", eth)

    assert store.get("BTC-USD") is btc
    assert store.contains("BTC-USD") is True
    assert "ETH-USD" in store
    assert len(store) == 2


def test_all_returns_defensive_copy() -> None:
    store = PositionStateStore()
    store.set("BTC-USD", _make_state("BTC-USD", "long"))

    positions = store.all()
    positions.pop("BTC-USD")

    assert "BTC-USD" in store.all()


def test_symbols_returns_snapshot() -> None:
    store = PositionStateStore()
    store.set("BTC-USD", _make_state("BTC-USD", "long"))

    snapshot = list(store.symbols())
    store.set("ETH-USD", _make_state("ETH-USD", "short"))

    assert snapshot == ["BTC-USD"]


def test_remove_missing_and_clear() -> None:
    store = PositionStateStore()
    store.set("BTC-USD", _make_state("BTC-USD", "long"))

    store.remove("ETH-USD")
    assert len(store) == 1

    store.clear()
    assert len(store) == 0
