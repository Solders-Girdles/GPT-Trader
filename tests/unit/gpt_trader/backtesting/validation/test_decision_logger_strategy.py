"""Tests for StrategyDecision.create() and serialization."""

from __future__ import annotations

from datetime import datetime
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


class TestStrategyDecisionSerialization:
    """Tests for StrategyDecision serialization."""

    def test_to_dict_converts_decimals(self) -> None:
        decision = StrategyDecision(
            decision_id="dec-001",
            cycle_id="cycle-001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("1.5"),
            position_side="long",
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("49900"), Decimal("50000")],
        )

        data = decision.to_dict()

        assert data["equity"] == "100000"
        assert data["mark_price"] == "50000"
        assert data["recent_marks"] == ["49900", "50000"]

    def test_to_dict_converts_datetime(self) -> None:
        ts = datetime(2024, 1, 1, 12, 30, 45)
        decision = StrategyDecision(
            decision_id="dec-001",
            cycle_id="cycle-001",
            timestamp=ts,
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        data = decision.to_dict()

        assert data["timestamp"] == "2024-01-01T12:30:45"

    def test_from_dict_restores_decimals(self) -> None:
        data = {
            "decision_id": "dec-001",
            "cycle_id": "cycle-001",
            "timestamp": "2024-01-01T12:00:00",
            "symbol": "BTC-USD",
            "equity": "100000",
            "position_quantity": "1.5",
            "position_side": "long",
            "mark_price": "50000",
            "recent_marks": ["49900", "50000"],
            "action": "BUY",
            "target_quantity": "1.0",
            "target_price": "50000",
            "order_type": "MARKET",
            "reason": "",
            "risk_checks_passed": True,
            "risk_check_failures": [],
            "strategy_name": "",
            "strategy_params": {},
        }

        decision = StrategyDecision.from_dict(data)

        assert decision.equity == Decimal("100000")
        assert decision.position_quantity == Decimal("1.5")
        assert decision.mark_price == Decimal("50000")
        assert decision.recent_marks == [Decimal("49900"), Decimal("50000")]

    def test_from_dict_restores_timestamp(self) -> None:
        data = {
            "decision_id": "dec-001",
            "cycle_id": "cycle-001",
            "timestamp": "2024-01-01T12:30:45",
            "symbol": "BTC-USD",
            "equity": "100000",
            "position_quantity": "0",
            "position_side": None,
            "mark_price": "50000",
            "recent_marks": ["50000"],
            "action": "HOLD",
            "target_quantity": "0",
            "target_price": None,
            "order_type": "MARKET",
            "reason": "",
            "risk_checks_passed": True,
            "risk_check_failures": [],
            "strategy_name": "",
            "strategy_params": {},
        }

        decision = StrategyDecision.from_dict(data)

        assert decision.timestamp == datetime(2024, 1, 1, 12, 30, 45)

    def test_roundtrip_serialization(self) -> None:
        original = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("1.5"),
            position_side="long",
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("49900"), Decimal("50000")],
        )
        original.with_action("BUY", Decimal("2.0"), Decimal("50100"))
        original.with_market_data(Decimal("50095"), Decimal("50105"))

        data = original.to_dict()
        restored = StrategyDecision.from_dict(data)

        assert restored.decision_id == original.decision_id
        assert restored.symbol == original.symbol
        assert restored.equity == original.equity
        assert restored.action == original.action
        assert restored.target_quantity == original.target_quantity
