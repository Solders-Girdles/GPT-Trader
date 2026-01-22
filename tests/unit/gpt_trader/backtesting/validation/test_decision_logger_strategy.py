"""Tests for StrategyDecision creation, builders, and serialization."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.backtesting.validation.decision_logger import StrategyDecision
from gpt_trader.utilities.datetime_helpers import utc_now


def make_decision(**overrides: object) -> StrategyDecision:
    params = {
        "cycle_id": "cycle-001",
        "symbol": "BTC-USD",
        "equity": Decimal("100000"),
        "position_quantity": Decimal("0"),
        "position_side": None,
        "mark_price": Decimal("50000"),
        "recent_marks": [Decimal("50000")],
    }
    params.update(overrides)
    return StrategyDecision.create(**params)


def make_decision_dict(**overrides: object) -> dict[str, object]:
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
    data.update(overrides)
    return data


def test_create_sets_id_and_timestamp() -> None:
    before = utc_now()
    decision = make_decision()
    assert decision.decision_id is not None
    assert len(decision.decision_id) == 12
    assert before <= decision.timestamp <= utc_now()


def test_create_preserves_parameters() -> None:
    decision = make_decision(
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


@pytest.mark.parametrize(
    ("method", "kwargs", "expected"),
    [
        (
            "with_market_data",
            {"bid": Decimal("49990"), "ask": Decimal("50010"), "volume": Decimal("1000")},
            {"bid": Decimal("49990"), "ask": Decimal("50010"), "volume": Decimal("1000")},
        ),
        (
            "with_market_data",
            {"bid": Decimal("49990")},
            {"bid": Decimal("49990"), "ask": None},
        ),
        (
            "with_strategy",
            {"name": "momentum_strategy", "params": {"lookback": 20, "threshold": 0.02}},
            {
                "strategy_name": "momentum_strategy",
                "strategy_params": {"lookback": 20, "threshold": 0.02},
            },
        ),
        (
            "with_strategy",
            {"name": "simple_strategy"},
            {"strategy_name": "simple_strategy", "strategy_params": {}},
        ),
        (
            "with_action",
            {
                "action": "BUY",
                "quantity": Decimal("1.0"),
                "price": Decimal("50000"),
                "order_type": "LIMIT",
                "reason": "Signal triggered",
            },
            {
                "action": "BUY",
                "target_quantity": Decimal("1.0"),
                "target_price": Decimal("50000"),
                "order_type": "LIMIT",
                "reason": "Signal triggered",
            },
        ),
        (
            "with_action",
            {"action": "HOLD"},
            {
                "action": "HOLD",
                "target_quantity": Decimal("0"),
                "target_price": None,
                "order_type": "MARKET",
            },
        ),
        (
            "with_risk_result",
            {"passed": True},
            {"risk_checks_passed": True, "risk_check_failures": []},
        ),
        (
            "with_risk_result",
            {"passed": False, "failures": ["too large", "daily loss"]},
            {"risk_checks_passed": False, "risk_check_failures": ["too large", "daily loss"]},
        ),
        (
            "with_execution",
            {
                "order_id": "order-12345",
                "fill_price": Decimal("50005"),
                "fill_quantity": Decimal("0.98"),
                "slippage_bps": Decimal("1.0"),
            },
            {
                "order_id": "order-12345",
                "fill_price": Decimal("50005"),
                "fill_quantity": Decimal("0.98"),
                "slippage_bps": Decimal("1.0"),
            },
        ),
    ],
)
def test_builder_methods(method, kwargs, expected) -> None:
    decision = make_decision()
    result = getattr(decision, method)(**kwargs)
    assert result is decision
    for attr, value in expected.items():
        assert getattr(decision, attr) == value


def test_to_dict_converts_decimals() -> None:
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


def test_to_dict_converts_datetime() -> None:
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


def test_from_dict_restores_decimals() -> None:
    decision = StrategyDecision.from_dict(make_decision_dict())
    assert decision.equity == Decimal("100000")
    assert decision.position_quantity == Decimal("1.5")
    assert decision.mark_price == Decimal("50000")
    assert decision.recent_marks == [Decimal("49900"), Decimal("50000")]


def test_from_dict_restores_timestamp() -> None:
    decision = StrategyDecision.from_dict(
        make_decision_dict(
            timestamp="2024-01-01T12:30:45",
            position_quantity="0",
            position_side=None,
            recent_marks=["50000"],
            action="HOLD",
            target_quantity="0",
            target_price=None,
        )
    )
    assert decision.timestamp == datetime(2024, 1, 1, 12, 30, 45)


def test_roundtrip_serialization() -> None:
    original = make_decision(
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
