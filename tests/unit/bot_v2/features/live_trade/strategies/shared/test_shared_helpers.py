from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import MarketType
from bot_v2.features.live_trade.strategies.decisions import Action
from bot_v2.features.live_trade.strategies.shared.decisions import (
    create_close_decision,
    create_entry_decision,
)
from bot_v2.features.live_trade.strategies.shared.mark_window import update_mark_window
from bot_v2.features.live_trade.strategies.shared.signals import calculate_ma_snapshot
from bot_v2.features.live_trade.strategies.shared.trailing_stop import update_trailing_stop


def test_create_entry_decision_applies_defaults_and_leverage():
    product = type("Product", (), {"market_type": MarketType.PERPETUAL})()
    position_adds: dict[str, int] = {"ETH-USD": 2}
    trailing_stops = {"ETH-USD": (Decimal("100"), Decimal("90"))}

    decision = create_entry_decision(
        symbol="BTC-USD",
        action=Action.BUY,
        equity=Decimal("10000"),
        product=product,
        position_fraction=0.0,
        target_leverage=3,
        max_trade_usd=Decimal("400"),
        position_adds=position_adds,
        trailing_stops=trailing_stops,
        reason="entry",
    )

    assert decision.action is Action.BUY
    # Default fraction 5%, capped to 400 then leverage multiplier 3 -> 1200
    assert decision.target_notional == Decimal("1200")
    assert decision.leverage == 3
    assert position_adds["BTC-USD"] == 0
    assert "BTC-USD" not in trailing_stops


def test_create_close_decision_resets_tracking_state():
    position_adds = {"BTC-USD": 2}
    trailing_stops = {"BTC-USD": (Decimal("120"), Decimal("110"))}

    decision = create_close_decision(
        symbol="BTC-USD",
        position_state={"quantity": "1.50"},
        position_adds=position_adds,
        trailing_stops=trailing_stops,
        reason="exit",
    )

    assert decision.action is Action.CLOSE
    assert decision.reduce_only is True
    assert decision.quantity == Decimal("1.50")
    assert "BTC-USD" not in position_adds
    assert "BTC-USD" not in trailing_stops


def test_update_mark_window_truncates_to_buffer():
    store: dict[str, list[Decimal]] = {}
    window = update_mark_window(
        store,
        symbol="BTC-USD",
        current_mark=Decimal("101"),
        short_period=3,
        long_period=5,
        recent_marks=[Decimal("100") + Decimal(i) for i in range(10)],
        buffer=2,
    )

    assert len(window) == max(3, 5) + 2
    assert store["BTC-USD"] == window
    assert window[-1] == Decimal("101")


def test_calculate_ma_snapshot_detects_cross():
    marks = [
        Decimal("110"),
        Decimal("109"),
        Decimal("108"),
        Decimal("107"),
        Decimal("106"),
        Decimal("105"),
        Decimal("104"),
        Decimal("105"),
        Decimal("106"),
        Decimal("107"),
    ]

    snapshot = calculate_ma_snapshot(
        marks,
        short_period=3,
        long_period=5,
        epsilon_bps=Decimal("5"),
        confirm_bars=0,
    )

    assert snapshot.short_ma > snapshot.long_ma
    assert snapshot.bullish_cross is True
    assert snapshot.bearish_cross is False

    empty_snapshot = calculate_ma_snapshot([], short_period=3, long_period=5)
    assert empty_snapshot.short_ma == Decimal("0")
    assert empty_snapshot.bullish_cross is False


@pytest.mark.parametrize(
    "side,initial_price,trailing_pct,price_sequence,expected_triggers",
    [
        # Long position: normal trailing behavior
        (
            "long",
            Decimal("100"),
            Decimal("0.1"),
            [Decimal("100"), Decimal("120"), Decimal("105")],
            [False, False, True],
        ),
        # Long position: price doesn't move enough to trigger
        (
            "long",
            Decimal("100"),
            Decimal("0.1"),
            [Decimal("100"), Decimal("110"), Decimal("100")],
            [False, False, False],
        ),
        # Short position: normal trailing behavior
        (
            "short",
            Decimal("200"),
            Decimal("0.05"),
            [Decimal("200"), Decimal("180"), Decimal("195")],
            [False, False, True],
        ),
        # Short position: price rises, stop should tighten
        (
            "short",
            Decimal("200"),
            Decimal("0.05"),
            [Decimal("200"), Decimal("190"), Decimal("185"), Decimal("190")],
            [False, False, False, True],
        ),
        # Edge case: zero trailing percentage
        (
            "long",
            Decimal("100"),
            Decimal("0"),
            [Decimal("100"), Decimal("120"), Decimal("99")],
            [False, False, True],
        ),
        # Edge case: very small trailing percentage
        (
            "long",
            Decimal("100"),
            Decimal("0.001"),
            [Decimal("100"), Decimal("101"), Decimal("100")],
            [False, False, False],
        ),
    ],
)
def test_update_trailing_stop_various_scenarios(
    side, initial_price, trailing_pct, price_sequence, expected_triggers
):
    stops: dict[str, tuple[Decimal, Decimal]] = {}
    symbol = "TEST-USD"

    for i, price in enumerate(price_sequence):
        triggered = update_trailing_stop(
            stops,
            symbol=symbol,
            side=side,
            current_price=price,
            trailing_pct=trailing_pct,
        )
        assert (
            triggered == expected_triggers[i]
        ), f"Step {i}: expected {expected_triggers[i]}, got {triggered}"

        if i == 0:  # Check initial stop placement
            peak, stop_price = stops[symbol]
            if side.lower() == "long":
                assert stop_price == initial_price * (Decimal("1") - trailing_pct)
            else:
                assert stop_price == initial_price * (Decimal("1") + trailing_pct)


def test_update_trailing_stop_case_insensitive_side():
    stops: dict[str, tuple[Decimal, Decimal]] = {}

    # Test LONG vs long
    update_trailing_stop(
        stops,
        symbol="BTC-USD",
        side="LONG",
        current_price=Decimal("100"),
        trailing_pct=Decimal("0.1"),
    )
    update_trailing_stop(
        stops,
        symbol="BTC-USD",
        side="long",
        current_price=Decimal("120"),
        trailing_pct=Decimal("0.1"),
    )

    peak, stop_price = stops["BTC-USD"]
    assert peak == Decimal("120")
    assert stop_price == Decimal("108")

    # Test SHORT vs short
    update_trailing_stop(
        stops,
        symbol="ETH-USD",
        side="SHORT",
        current_price=Decimal("200"),
        trailing_pct=Decimal("0.05"),
    )
    update_trailing_stop(
        stops,
        symbol="ETH-USD",
        side="short",
        current_price=Decimal("180"),
        trailing_pct=Decimal("0.05"),
    )

    peak, stop_price = stops["ETH-USD"]
    assert peak == Decimal("180")
    assert stop_price == Decimal("189")


def test_update_trailing_stop_invalid_side():
    stops: dict[str, tuple[Decimal, Decimal]] = {}

    # Invalid side should not update stops and return False
    triggered = update_trailing_stop(
        stops,
        symbol="BTC-USD",
        side="invalid",
        current_price=Decimal("100"),
        trailing_pct=Decimal("0.1"),
    )
    assert triggered is False
    assert "BTC-USD" not in stops
