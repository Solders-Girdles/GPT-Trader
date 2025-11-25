from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.risk import LiveRiskManager
from gpt_trader.orchestration.configuration import RiskConfig


def test_liquidation_buffer_with_real_liq_price_triggers_reduce_only():
    config = RiskConfig(min_liquidation_buffer_pct=0.15)
    rm = LiveRiskManager(config=config)

    equity = Decimal("20000")
    pos = {
        "quantity": Decimal("1"),
        "mark": Decimal("100"),
        "liquidation_price": Decimal("90"),  # 10% from mark
    }

    triggered = rm.check_liquidation_buffer("BTC-PERP", pos, equity)
    assert triggered is True
    assert rm.positions["BTC-PERP"]["reduce_only"] is True


def test_liquidation_buffer_with_real_liq_price_ok_when_above_min():
    config = RiskConfig(min_liquidation_buffer_pct=0.10)
    rm = LiveRiskManager(config=config)

    equity = Decimal("20000")
    pos = {
        "quantity": Decimal("2"),
        "mark": Decimal("100"),
        "liquidation_price": Decimal("85"),  # 15%
    }

    triggered = rm.check_liquidation_buffer("BTC-PERP", pos, equity)
    assert triggered is False
