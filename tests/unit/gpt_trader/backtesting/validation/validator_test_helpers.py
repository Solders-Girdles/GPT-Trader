"""Shared helpers for golden-path validator tests."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from gpt_trader.backtesting.validation.decision_logger import StrategyDecision


def create_decision(
    action: str = "BUY",
    target_quantity: Decimal = Decimal("1.0"),
    target_price: Decimal | None = Decimal("50000"),
    risk_checks_passed: bool = True,
    order_type: str = "MARKET",
    cycle_id: str = "cycle-001",
    symbol: str = "BTC-USD",
) -> StrategyDecision:
    """Create a strategy decision for testing."""
    return StrategyDecision(
        decision_id="test-dec-001",
        cycle_id=cycle_id,
        symbol=symbol,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        equity=Decimal("100000"),
        position_quantity=Decimal("0"),
        position_side=None,
        mark_price=Decimal("50000"),
        recent_marks=[Decimal("50000")],
        action=action,
        target_quantity=target_quantity,
        target_price=target_price,
        order_type=order_type,
        risk_checks_passed=risk_checks_passed,
        reason="Test decision",
    )
