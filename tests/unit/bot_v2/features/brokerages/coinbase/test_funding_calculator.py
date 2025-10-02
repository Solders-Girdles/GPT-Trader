from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.coinbase.utilities import FundingCalculator


def test_accrue_if_due_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    calc = FundingCalculator()
    caplog.set_level("WARNING")

    delta1 = calc.accrue_if_due(
        symbol="BTC-USD-PERP",
        position_size=Decimal("2"),
        position_side="long",
        mark_price=Decimal("100"),
        funding_rate=None,
        next_funding_time=None,
        now=datetime(2025, 1, 15, 12, 0, 0),
    )
    delta2 = calc.accrue_if_due(
        symbol="BTC-USD-PERP",
        position_size=Decimal("2"),
        position_side="long",
        mark_price=Decimal("100"),
        funding_rate=None,
        next_funding_time=None,
        now=datetime(2025, 1, 15, 12, 5, 0),
    )

    assert delta1 == Decimal("0") == delta2
    warnings = [record for record in caplog.records if "No funding data" in record.message]
    assert len(warnings) == 1


def test_accrue_if_due_first_observation_waits() -> None:
    calc = FundingCalculator()
    now = datetime(2025, 1, 15, 12, 0, 0)
    initial_funding_time = now - timedelta(minutes=5)

    # First observation past funding time should record timestamp but not accrue
    delta = calc.accrue_if_due(
        symbol="ETH-USD-PERP",
        position_size=Decimal("1"),
        position_side="long",
        mark_price=Decimal("2000"),
        funding_rate=Decimal("0.0001"),
        next_funding_time=initial_funding_time,
        now=now,
    )
    assert delta == Decimal("0")

    # Funding due on next interval
    next_time = now + timedelta(hours=8)
    delta = calc.accrue_if_due(
        symbol="ETH-USD-PERP",
        position_size=Decimal("1"),
        position_side="long",
        mark_price=Decimal("2100"),
        funding_rate=Decimal("0.0001"),
        next_funding_time=next_time,
        now=next_time + timedelta(minutes=1),
    )
    # Longs pay positive funding -> negative delta
    assert delta == Decimal("-0.21")


def test_accrue_if_due_short_receives_positive() -> None:
    calc = FundingCalculator()
    first_funding_time = datetime(2025, 1, 15, 0, 0, 0)

    # Prime state with first observation
    calc.accrue_if_due(
        symbol="SOL-USD-PERP",
        position_size=Decimal("3"),
        position_side="short",
        mark_price=Decimal("100"),
        funding_rate=Decimal("0.0002"),
        next_funding_time=first_funding_time,
        now=first_funding_time + timedelta(minutes=1),
    )

    next_time = first_funding_time + timedelta(hours=8)
    delta = calc.accrue_if_due(
        symbol="SOL-USD-PERP",
        position_size=Decimal("3"),
        position_side="short",
        mark_price=Decimal("110"),
        funding_rate=Decimal("0.0002"),
        next_funding_time=next_time,
        now=next_time + timedelta(minutes=1),
    )

    # Shorts receive when funding positive
    assert delta == Decimal("0.066")
