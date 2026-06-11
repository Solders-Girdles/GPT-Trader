from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest

from gpt_trader.features.trade_ideas import (
    AutonomyMode,
    Confidence,
    ConfidenceLabel,
    EntryZone,
    MaxLoss,
    ProductType,
    SizingRecommendation,
    TimeHorizon,
    TradeDirection,
    TradeIdea,
)


def build_trade_idea(**overrides: Any) -> TradeIdea:
    """Build a fully-populated, eligible trade idea; override fields per test."""
    fields: dict[str, Any] = {
        "decision_id": "trade-20260612-001",
        "autonomy_mode": AutonomyMode.HUMAN_APPROVED_EXECUTION,
        "thesis": "BTC reclaiming the 50-day average with rising spot volume",
        "instrument": "BTC-USD",
        "product_type": ProductType.SPOT,
        "direction": TradeDirection.LONG,
        "entry_zone": EntryZone(lower=Decimal("60000"), upper=Decimal("61500")),
        "invalidation": "Daily close below 58000",
        "target_exit": "Take profit at 67000 or exit after 10 trading days",
        "max_loss": MaxLoss(
            amount=Decimal("250"),
            percent_of_account=Decimal("1.5"),
            assumptions=("Fill at zone midpoint", "No slippage beyond 10 bps"),
        ),
        "sizing_recommendation": SizingRecommendation(
            quantity=Decimal("0.1"),
            notional=Decimal("6075"),
            rationale="Half-Kelly on backtested edge",
        ),
        "time_horizon": TimeHorizon(
            expected_hold="3-10 days",
            expires_at=datetime(2026, 6, 19, 16, 0, tzinfo=UTC),
        ),
        "data_used": ("coinbase:candles:BTC-USD:1d:2026-06-11",),
        "confidence": Confidence(
            label=ConfidenceLabel.MEDIUM,
            rationale="Volume confirmation present, macro calendar risk this week",
        ),
        "failure_mode": "False breakout into a macro-driven selloff",
        "do_not_trade_if": ("FOMC announcement within 24 hours",),
    }
    fields.update(overrides)
    return TradeIdea(**fields)


@pytest.fixture
def trade_idea() -> TradeIdea:
    return build_trade_idea()
