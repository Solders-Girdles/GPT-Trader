"""Tests for StrategyDecision.create()."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.backtesting.validation.decision_logger import StrategyDecision
from gpt_trader.utilities.datetime_helpers import utc_now


class TestStrategyDecisionCreate:
    """Tests for StrategyDecision.create() class method."""

    def test_create_generates_decision_id(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )
        assert decision.decision_id is not None
        assert len(decision.decision_id) == 12

    def test_create_sets_timestamp(self) -> None:
        before = utc_now()
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )
        after = utc_now()
        assert before <= decision.timestamp <= after

    def test_create_preserves_parameters(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="ETH-USD",
            equity=Decimal("50000"),
            position_quantity=Decimal("1.5"),
            position_side="long",
            mark_price=Decimal("3000"),
            recent_marks=[Decimal("2990"), Decimal("3000"), Decimal("3010")],
        )
        assert decision.cycle_id == "cycle-001"
        assert decision.symbol == "ETH-USD"
        assert decision.equity == Decimal("50000")
        assert decision.position_quantity == Decimal("1.5")
        assert decision.position_side == "long"
        assert decision.mark_price == Decimal("3000")
        assert len(decision.recent_marks) == 3
