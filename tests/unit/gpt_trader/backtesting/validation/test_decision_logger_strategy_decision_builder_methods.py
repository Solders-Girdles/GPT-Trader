"""Tests for StrategyDecision builder methods."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.backtesting.validation.decision_logger import StrategyDecision


class TestStrategyDecisionBuilderMethods:
    """Tests for StrategyDecision builder methods."""

    def test_with_market_data(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_market_data(
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            volume=Decimal("1000"),
        )

        assert result is decision
        assert decision.bid == Decimal("49990")
        assert decision.ask == Decimal("50010")
        assert decision.volume == Decimal("1000")

    def test_with_market_data_partial(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        decision.with_market_data(bid=Decimal("49990"))

        assert decision.bid == Decimal("49990")
        assert decision.ask is None

    def test_with_strategy(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_strategy(
            name="momentum_strategy",
            params={"lookback": 20, "threshold": 0.02},
        )

        assert result is decision
        assert decision.strategy_name == "momentum_strategy"
        assert decision.strategy_params == {"lookback": 20, "threshold": 0.02}

    def test_with_strategy_no_params(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        decision.with_strategy(name="simple_strategy")

        assert decision.strategy_name == "simple_strategy"
        assert decision.strategy_params == {}

    def test_with_action(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_action(
            action="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            order_type="LIMIT",
            reason="Signal triggered",
        )

        assert result is decision
        assert decision.action == "BUY"
        assert decision.target_quantity == Decimal("1.0")
        assert decision.target_price == Decimal("50000")
        assert decision.order_type == "LIMIT"
        assert decision.reason == "Signal triggered"

    def test_with_action_defaults(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        decision.with_action(action="HOLD")

        assert decision.action == "HOLD"
        assert decision.target_quantity == Decimal("0")
        assert decision.target_price is None
        assert decision.order_type == "MARKET"

    def test_with_risk_result_passed(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_risk_result(passed=True)

        assert result is decision
        assert decision.risk_checks_passed is True
        assert decision.risk_check_failures == []

    def test_with_risk_result_failed(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        decision.with_risk_result(
            passed=False,
            failures=["Position too large", "Daily loss limit exceeded"],
        )

        assert decision.risk_checks_passed is False
        assert len(decision.risk_check_failures) == 2
        assert "Position too large" in decision.risk_check_failures

    def test_with_execution(self) -> None:
        decision = StrategyDecision.create(
            cycle_id="cycle-001",
            symbol="BTC-USD",
            equity=Decimal("100000"),
            position_quantity=Decimal("0"),
            position_side=None,
            mark_price=Decimal("50000"),
            recent_marks=[Decimal("50000")],
        )

        result = decision.with_execution(
            order_id="order-12345",
            fill_price=Decimal("50005"),
            fill_quantity=Decimal("0.98"),
            slippage_bps=Decimal("1.0"),
        )

        assert result is decision
        assert decision.order_id == "order-12345"
        assert decision.fill_price == Decimal("50005")
        assert decision.fill_quantity == Decimal("0.98")
        assert decision.slippage_bps == Decimal("1.0")
