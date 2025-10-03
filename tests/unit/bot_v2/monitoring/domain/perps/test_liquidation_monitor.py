from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.monitoring.domain.perps.liquidation import (
    LiquidationMonitor,
    LiquidationRisk,
    MarginInfo,
)


def make_margin(
    *,
    side: str,
    entry: float,
    current: float,
    leverage: float = 5.0,
    maintenance: float = 0.05,
    size: float = 1.0,
) -> MarginInfo:
    return MarginInfo(
        symbol="BTC-USD-PERP",
        position_size=Decimal(str(size)),
        position_side=side,
        entry_price=Decimal(str(entry)),
        current_price=Decimal(str(current)),
        leverage=Decimal(str(leverage)),
        maintenance_margin_rate=Decimal(str(maintenance)),
    )


def test_calculate_liquidation_price_long_short() -> None:
    monitor = LiquidationMonitor()
    long_margin = make_margin(side="long", entry=100, current=90, leverage=4)
    short_margin = make_margin(side="short", entry=100, current=110, leverage=3)

    long_liq = monitor.calculate_liquidation_price(long_margin)
    short_liq = monitor.calculate_liquidation_price(short_margin)

    assert long_liq and long_liq < long_margin.entry_price
    assert short_liq and short_liq > short_margin.entry_price


def test_calculate_liquidation_price_handles_errors(caplog: pytest.LogCaptureFixture) -> None:
    monitor = LiquidationMonitor()
    bad_margin = make_margin(side="long", entry=100, current=90, leverage=0)
    caplog.set_level("ERROR")
    liq = monitor.calculate_liquidation_price(bad_margin)
    assert liq is None
    assert any(
        "Failed to calculate liquidation price" in record.message for record in caplog.records
    )


def test_assess_liquidation_risk_states() -> None:
    monitor = LiquidationMonitor(warning_buffer_pct=20.0, critical_buffer_pct=10.0)

    no_position = monitor.assess_liquidation_risk(
        make_margin(side="long", entry=100, current=100, size=0)
    )
    assert no_position.risk_level == "safe"

    liquidated = monitor.assess_liquidation_risk(
        make_margin(side="long", entry=100, current=10, leverage=1.5)
    )
    assert liquidated.risk_level == "liquidated"
    assert liquidated.should_reduce_only is True

    critical = monitor.assess_liquidation_risk(
        make_margin(side="short", entry=100, current=112, leverage=5)
    )
    assert critical.risk_level == "critical"
    assert critical.should_reject_entry is True

    warning = monitor.assess_liquidation_risk(
        make_margin(side="long", entry=100, current=85, leverage=3)
    )
    assert warning.risk_level == "warning"
    assert warning.reason.startswith("Liquidation warning")

    safe = monitor.assess_liquidation_risk(
        make_margin(side="long", entry=100, current=150, leverage=3)
    )
    assert safe.risk_level == "safe"


def test_should_block_new_position_existing_and_portfolio() -> None:
    monitor = LiquidationMonitor(enable_entry_rejection=True)
    risky_margin = make_margin(side="long", entry=100, current=85, leverage=3)
    positions = {"BTC-USD-PERP": risky_margin}

    should_block, reason = monitor.should_block_new_position("BTC-USD-PERP", positions)
    assert should_block is True
    assert "Existing position" in reason

    other_positions = {
        "ETH-USD-PERP": make_margin(side="short", entry=200, current=224, leverage=4, size=2.0),
    }
    should_block, reason = monitor.should_block_new_position("SOL-USD-PERP", other_positions)
    assert should_block is True
    assert "Portfolio liquidation risk" in reason


def test_should_not_block_when_safe() -> None:
    monitor = LiquidationMonitor(enable_entry_rejection=True)
    safe_positions = {
        "BTC-USD-PERP": make_margin(side="long", entry=100, current=150, leverage=3),
    }
    should_block, reason = monitor.should_block_new_position("SOL-USD-PERP", safe_positions)
    assert should_block is False
    assert reason == "No liquidation risk blocking new entries"
