from __future__ import annotations

from decimal import Decimal

from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.config.live_trade_config import RiskConfig


def test_slippage_guard_disabled_no_error():
    cfg = RiskConfig(slippage_guard_bps=0)
    rm = LiveRiskManager(config=cfg)

    # Huge slippage should be ignored when disabled
    rm.validate_slippage_guard(
        symbol="BTC-PERP",
        side="buy",
        qty=Decimal("1"),
        expected_price=Decimal("55000"),  # 10% above
        mark_or_quote=Decimal("50000"),
    )


def test_slippage_just_under_threshold_allowed():
    cfg = RiskConfig(slippage_guard_bps=50)
    rm = LiveRiskManager(config=cfg)

    # 49 bps above mark → allowed
    rm.validate_slippage_guard(
        symbol="BTC-PERP",
        side="buy",
        qty=Decimal("1"),
        expected_price=Decimal("50245"),
        mark_or_quote=Decimal("50000"),
    )

