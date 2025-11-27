"""Tests for golden-path validator module."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from gpt_trader.backtesting.validation.decision_logger import StrategyDecision
from gpt_trader.backtesting.validation.validator import (
    GoldenPathValidator,
    ValidationResult,
)


def _create_decision(
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


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_matching_result(self) -> None:
        live = _create_decision()
        result = ValidationResult(
            matches=True,
            live_decision=live,
            sim_decision=live,
            confidence=Decimal("100"),
        )
        assert result.matches is True
        assert result.confidence == Decimal("100")
        assert result.divergence is None

    def test_non_matching_result(self) -> None:
        live = _create_decision(action="BUY")
        sim = _create_decision(action="SELL")
        result = ValidationResult(
            matches=False,
            live_decision=live,
            sim_decision=sim,
            confidence=Decimal("0"),
        )
        assert result.matches is False
        assert result.live_decision.action == "BUY"
        assert result.sim_decision.action == "SELL"

    def test_result_with_details(self) -> None:
        live = _create_decision()
        result = ValidationResult(
            matches=True,
            live_decision=live,
            details={"check1": "passed", "check2": "within tolerance"},
        )
        assert "check1" in result.details
        assert result.details["check1"] == "passed"


class TestGoldenPathValidatorInit:
    """Tests for GoldenPathValidator initialization."""

    def test_default_tolerances(self) -> None:
        validator = GoldenPathValidator()
        assert validator.quantity_tolerance == Decimal("0.01")
        assert validator.price_tolerance == Decimal("0.001")
        assert validator.strict_action_match is True

    def test_custom_tolerances(self) -> None:
        validator = GoldenPathValidator(
            quantity_tolerance_pct=Decimal("0.05"),
            price_tolerance_pct=Decimal("0.01"),
            strict_action_match=False,
        )
        assert validator.quantity_tolerance == Decimal("0.05")
        assert validator.price_tolerance == Decimal("0.01")
        assert validator.strict_action_match is False

    def test_initial_state(self) -> None:
        validator = GoldenPathValidator()
        assert validator.total_comparisons == 0
        assert validator.matching_comparisons == 0
        assert validator.divergence_count == 0


class TestValidateDecision:
    """Tests for validate_decision method."""

    def test_exact_match(self) -> None:
        validator = GoldenPathValidator()
        live = _create_decision()
        sim = _create_decision()

        result = validator.validate_decision(live, sim)

        assert result.matches is True
        assert result.confidence == Decimal("100")
        assert validator.total_comparisons == 1
        assert validator.matching_comparisons == 1

    def test_action_mismatch_strict(self) -> None:
        validator = GoldenPathValidator(strict_action_match=True)
        live = _create_decision(action="BUY")
        sim = _create_decision(action="SELL")

        result = validator.validate_decision(live, sim)

        assert result.matches is False
        assert result.confidence == Decimal("0")
        assert result.divergence is not None
        assert "Action mismatch" in result.divergence.reason

    def test_action_mismatch_non_strict(self) -> None:
        validator = GoldenPathValidator(strict_action_match=False)
        live = _create_decision(action="BUY")
        sim = _create_decision(action="SELL")

        result = validator.validate_decision(live, sim)

        # Non-strict mode records warning but doesn't fail
        assert "action_warning" in result.details
        assert result.confidence < Decimal("100")

    def test_quantity_within_tolerance(self) -> None:
        validator = GoldenPathValidator(quantity_tolerance_pct=Decimal("0.05"))
        live = _create_decision(target_quantity=Decimal("1.0"))
        sim = _create_decision(target_quantity=Decimal("1.02"))  # 2% diff

        result = validator.validate_decision(live, sim)

        assert result.matches is True
        assert "quantity_check" in result.details

    def test_quantity_outside_tolerance(self) -> None:
        validator = GoldenPathValidator(quantity_tolerance_pct=Decimal("0.01"))
        live = _create_decision(target_quantity=Decimal("1.0"))
        sim = _create_decision(target_quantity=Decimal("1.05"))  # 5% diff

        result = validator.validate_decision(live, sim)

        assert result.matches is False
        assert result.divergence is not None
        assert "Quantity mismatch" in result.divergence.reason

    def test_price_within_tolerance(self) -> None:
        validator = GoldenPathValidator(price_tolerance_pct=Decimal("0.01"))
        live = _create_decision(target_price=Decimal("50000"))
        sim = _create_decision(target_price=Decimal("50025"))  # 0.05% diff

        result = validator.validate_decision(live, sim)

        assert result.matches is True

    def test_price_outside_tolerance(self) -> None:
        validator = GoldenPathValidator(price_tolerance_pct=Decimal("0.001"))
        live = _create_decision(target_price=Decimal("50000"))
        sim = _create_decision(target_price=Decimal("50100"))  # 0.2% diff

        result = validator.validate_decision(live, sim)

        assert result.matches is False
        assert "Price mismatch" in result.divergence.reason

    def test_risk_check_mismatch(self) -> None:
        validator = GoldenPathValidator()
        live = _create_decision(risk_checks_passed=True)
        sim = _create_decision(risk_checks_passed=False)

        result = validator.validate_decision(live, sim)

        assert result.matches is False
        assert "Risk check mismatch" in result.divergence.reason

    def test_order_type_mismatch_warning(self) -> None:
        validator = GoldenPathValidator()
        live = _create_decision(order_type="MARKET")
        sim = _create_decision(order_type="LIMIT")

        result = validator.validate_decision(live, sim)

        # Order type mismatch is a warning, not a failure
        assert "order_type_warning" in result.details
        assert result.confidence < Decimal("100")

    def test_hold_actions_skip_quantity_check(self) -> None:
        validator = GoldenPathValidator()
        live = _create_decision(action="HOLD", target_quantity=Decimal("0"))
        sim = _create_decision(action="HOLD", target_quantity=Decimal("0"))

        result = validator.validate_decision(live, sim)

        assert result.matches is True


class TestEstimateImpact:
    """Tests for _estimate_impact method."""

    def test_hold_vs_trade_full_impact(self) -> None:
        validator = GoldenPathValidator()
        live = _create_decision(action="HOLD")
        sim = _create_decision(action="BUY")

        result = validator.validate_decision(live, sim)

        # One traded, one held = 100% impact
        assert result.divergence is not None
        assert result.divergence.impact_pct == Decimal("100")

    def test_both_hold_zero_impact(self) -> None:
        validator = GoldenPathValidator()
        # For both HOLD with mismatch on something else
        live = _create_decision(action="HOLD", risk_checks_passed=True)
        sim = _create_decision(action="HOLD", risk_checks_passed=False)

        result = validator.validate_decision(live, sim)

        # Both holding = 0 impact for action itself
        assert result.divergence is not None
        # Impact calculated for risk check mismatch


class TestGetMatchRate:
    """Tests for get_match_rate method."""

    def test_no_comparisons_returns_100(self) -> None:
        validator = GoldenPathValidator()
        assert validator.get_match_rate() == Decimal("100")

    def test_all_matches(self) -> None:
        validator = GoldenPathValidator()
        for _ in range(5):
            live = _create_decision()
            sim = _create_decision()
            validator.validate_decision(live, sim)

        assert validator.get_match_rate() == Decimal("100")

    def test_partial_matches(self) -> None:
        validator = GoldenPathValidator()

        # 3 matches
        for _ in range(3):
            live = _create_decision()
            sim = _create_decision()
            validator.validate_decision(live, sim)

        # 2 mismatches
        for _ in range(2):
            live = _create_decision(action="BUY")
            sim = _create_decision(action="SELL")
            validator.validate_decision(live, sim)

        # 3/5 = 60%
        assert validator.get_match_rate() == Decimal("60")


class TestGetDivergences:
    """Tests for get_divergences method."""

    def test_empty_divergences(self) -> None:
        validator = GoldenPathValidator()
        divergences = validator.get_divergences()
        assert divergences == []

    def test_filter_by_symbol(self) -> None:
        validator = GoldenPathValidator()

        # BTC divergence
        live = _create_decision(action="BUY", symbol="BTC-USD")
        sim = _create_decision(action="SELL", symbol="BTC-USD")
        validator.validate_decision(live, sim)

        # ETH divergence
        live = _create_decision(action="BUY", symbol="ETH-USD")
        sim = _create_decision(action="SELL", symbol="ETH-USD")
        validator.validate_decision(live, sim)

        btc_divergences = validator.get_divergences(symbol="BTC-USD")
        assert len(btc_divergences) == 1
        assert btc_divergences[0].symbol == "BTC-USD"

    def test_filter_by_min_impact(self) -> None:
        validator = GoldenPathValidator()

        # High impact (HOLD vs BUY = 100%)
        live = _create_decision(action="HOLD")
        sim = _create_decision(action="BUY")
        validator.validate_decision(live, sim)

        # Lower impact (quantity mismatch)
        live = _create_decision(target_quantity=Decimal("1.0"))
        sim = _create_decision(target_quantity=Decimal("1.05"))
        validator.validate_decision(live, sim)

        high_impact = validator.get_divergences(min_impact=Decimal("50"))
        assert len(high_impact) == 1


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_state(self) -> None:
        validator = GoldenPathValidator()

        # Make some comparisons
        for _ in range(3):
            live = _create_decision(action="BUY")
            sim = _create_decision(action="SELL")
            validator.validate_decision(live, sim)

        assert validator.total_comparisons == 3
        assert validator.divergence_count == 3

        validator.reset()

        assert validator.total_comparisons == 0
        assert validator.matching_comparisons == 0
        assert validator.divergence_count == 0


class TestGenerateReport:
    """Tests for generate_report method."""

    def test_generate_report_structure(self) -> None:
        validator = GoldenPathValidator()

        # Add some validations
        for i in range(5):
            live = _create_decision(cycle_id="cycle-001")
            sim = _create_decision(cycle_id="cycle-001")
            validator.validate_decision(live, sim)

        report = validator.generate_report("cycle-001")

        assert report.cycle_id == "cycle-001"
        assert report.total_decisions == 5
        assert report.matching_decisions == 5
        assert report.divergences == []

    def test_report_includes_divergences(self) -> None:
        validator = GoldenPathValidator()

        live = _create_decision(action="BUY", cycle_id="cycle-002")
        sim = _create_decision(action="SELL", cycle_id="cycle-002")
        validator.validate_decision(live, sim)

        report = validator.generate_report("cycle-002")

        assert len(report.divergences) == 1
        assert report.divergences[0].cycle_id == "cycle-002"


class TestProperties:
    """Tests for validator properties."""

    def test_total_comparisons_property(self) -> None:
        validator = GoldenPathValidator()
        assert validator.total_comparisons == 0

        validator.validate_decision(_create_decision(), _create_decision())
        assert validator.total_comparisons == 1

    def test_matching_comparisons_property(self) -> None:
        validator = GoldenPathValidator()
        validator.validate_decision(_create_decision(), _create_decision())
        assert validator.matching_comparisons == 1

    def test_divergence_count_property(self) -> None:
        validator = GoldenPathValidator()
        live = _create_decision(action="BUY")
        sim = _create_decision(action="SELL")
        validator.validate_decision(live, sim)
        assert validator.divergence_count == 1
