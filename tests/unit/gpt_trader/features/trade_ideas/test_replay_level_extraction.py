from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from tests.unit.gpt_trader.features.trade_ideas.conftest import build_trade_idea

from gpt_trader.core import Candle
from gpt_trader.features.trade_ideas import (
    EntryZone,
    ScoringLevels,
    TimeHorizon,
    TradeDirection,
    TradeIdea,
    extract_numeric_scoring_levels,
)

AS_OF = datetime(2026, 6, 12, 12, 0, tzinfo=UTC)


def candle(
    offset_hours: int,
    *,
    open_: str = "101",
    high: str = "102",
    low: str = "100",
    close: str = "101",
) -> Candle:
    return Candle(
        ts=AS_OF + timedelta(hours=offset_hours),
        open=Decimal(open_),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


def scoreable_idea(**overrides: object) -> TradeIdea:
    fields = {
        "entry_zone": EntryZone(lower=Decimal("100"), upper=Decimal("102")),
        "invalidation": "Close below 95",
        "target_exit": "Take profit at 113 or exit at expiry",
        "time_horizon": TimeHorizon(
            expected_hold="1-4 hours",
            expires_at=AS_OF + timedelta(hours=4),
        ),
    }
    fields.update(overrides)
    return build_trade_idea(**fields)


def test_extract_numeric_scoring_levels_prefers_parenthesized_baseline_stop() -> None:
    idea = scoreable_idea(
        invalidation="Close below the 50-bar average (97.50)",
        target_exit="Take profit near 113.00 (2R) or exit at expiry",
    )

    assert extract_numeric_scoring_levels(idea) == ScoringLevels(
        entry_lower=Decimal("100"),
        entry_upper=Decimal("102"),
        stop=Decimal("97.50"),
        target=Decimal("113.00"),
    )


def test_extract_numeric_scoring_levels_ignores_indicator_period_before_stop() -> None:
    idea = scoreable_idea(
        invalidation="Close below 95 if RSI (14) rolls over",
        target_exit="Take profit near 113.00 (2R) or exit at expiry",
    )

    assert extract_numeric_scoring_levels(idea).stop == Decimal("95")


def test_extract_numeric_scoring_levels_uses_direct_stop_when_indicator_is_closer() -> None:
    idea = scoreable_idea(
        entry_zone=EntryZone(lower=Decimal("15"), upper=Decimal("17")),
        invalidation="Close below 10 if RSI (14) rolls over",
        target_exit="Take profit near 25.00 (2R) or exit at expiry",
    )

    assert extract_numeric_scoring_levels(idea).stop == Decimal("10")


def test_extract_numeric_scoring_levels_ignores_indicator_threshold_before_stop() -> None:
    idea = scoreable_idea(
        invalidation="Invalid if RSI below 30 or close below 95",
        target_exit="Take profit near 113.00 (2R) or exit at expiry",
    )

    assert extract_numeric_scoring_levels(idea).stop == Decimal("95")


def test_extract_numeric_scoring_levels_ignores_reward_multiple_before_short_target() -> None:
    idea = scoreable_idea(
        direction=TradeDirection.SHORT,
        invalidation="Close above 106",
        target_exit="2R target at 95 or exit at expiry",
    )

    assert extract_numeric_scoring_levels(idea).target == Decimal("95")


def test_extract_numeric_scoring_levels_ignores_percentage_before_long_target() -> None:
    idea = scoreable_idea(
        entry_zone=EntryZone(lower=Decimal("19"), upper=Decimal("21")),
        invalidation="Close below 18",
        target_exit="Take 25% at 30 or exit at expiry",
    )

    assert extract_numeric_scoring_levels(idea).target == Decimal("30")


def test_extract_numeric_scoring_levels_keeps_target_before_resistance_word() -> None:
    idea = scoreable_idea(
        direction=TradeDirection.SHORT,
        invalidation="Close above 106",
        target_exit="Take profit near 95 resistance or exit at expiry",
    )

    assert extract_numeric_scoring_levels(idea).target == Decimal("95")


def test_extract_numeric_scoring_levels_ignores_percentage_quantity_targets() -> None:
    idea = scoreable_idea(
        entry_zone=EntryZone(lower=Decimal("15"), upper=Decimal("17")),
        invalidation="Close below 12",
        target_exit="Take 25% at 20, then exit the rest at expiry",
    )

    assert extract_numeric_scoring_levels(idea).target == Decimal("20")
