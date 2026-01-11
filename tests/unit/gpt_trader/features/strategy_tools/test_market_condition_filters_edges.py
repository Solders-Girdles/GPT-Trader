from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.strategy_tools.filters import MarketConditionFilters


@pytest.mark.parametrize(
    ("filters_kwargs", "snapshot", "expected_reason"),
    [
        (
            {"max_spread_bps": Decimal("10")},
            {"spread_bps": Decimal("12")},
            "Spread too wide",
        ),
        (
            {"min_depth_l1": Decimal("100")},
            {"depth_l1": Decimal("50")},
            "L1 depth insufficient",
        ),
        (
            {"min_depth_l10": Decimal("200")},
            {"depth_l10": Decimal("150")},
            "L10 depth insufficient",
        ),
        (
            {"min_volume_1m": Decimal("20")},
            {"vol_1m": Decimal("10")},
            "1m volume too low",
        ),
        (
            {"min_volume_5m": Decimal("40")},
            {"vol_5m": Decimal("30")},
            "5m volume too low",
        ),
    ],
)
def test_long_entry_gates_reject(filters_kwargs, snapshot, expected_reason) -> None:
    filters = MarketConditionFilters(**filters_kwargs)

    ok, reason = filters.should_allow_long_entry(snapshot)

    assert ok is False
    assert expected_reason in reason


def test_rsi_confirmation_rejects_long_overbought() -> None:
    filters = MarketConditionFilters(require_rsi_confirmation=True)

    ok, reason = filters.should_allow_long_entry({}, rsi=Decimal("80"))

    assert ok is False
    assert "RSI too high for long entry" in reason


def test_rsi_confirmation_rejects_short_oversold() -> None:
    filters = MarketConditionFilters(require_rsi_confirmation=True)

    ok, reason = filters.should_allow_short_entry({}, rsi=Decimal("20"))

    assert ok is False
    assert "RSI too low for short entry" in reason


def test_short_entry_surfaces_long_gate_reason() -> None:
    filters = MarketConditionFilters(max_spread_bps=Decimal("5"))
    snapshot = {"spread_bps": Decimal("10")}

    long_ok, long_reason = filters.should_allow_long_entry(snapshot)
    short_ok, short_reason = filters.should_allow_short_entry(snapshot, rsi=Decimal("20"))

    assert long_ok is False
    assert short_ok is False
    assert short_reason == long_reason
