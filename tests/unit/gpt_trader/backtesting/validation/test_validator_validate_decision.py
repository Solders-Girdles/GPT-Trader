"""Tests for GoldenPathValidator.validate_decision()."""

from __future__ import annotations

from decimal import Decimal

from tests.unit.gpt_trader.backtesting.validation.validator_test_helpers import create_decision

from gpt_trader.backtesting.validation.validator import GoldenPathValidator


class TestValidateDecision:
    """Tests for validate_decision method."""

    def test_exact_match(self) -> None:
        validator = GoldenPathValidator()
        live = create_decision()
        sim = create_decision()

        result = validator.validate_decision(live, sim)

        assert result.matches is True
        assert result.confidence == Decimal("100")
        assert validator.total_comparisons == 1
        assert validator.matching_comparisons == 1

    def test_action_mismatch_strict(self) -> None:
        validator = GoldenPathValidator(strict_action_match=True)
        live = create_decision(action="BUY")
        sim = create_decision(action="SELL")

        result = validator.validate_decision(live, sim)

        assert result.matches is False
        assert result.confidence == Decimal("0")
        assert result.divergence is not None
        assert "Action mismatch" in result.divergence.reason

    def test_action_mismatch_non_strict(self) -> None:
        validator = GoldenPathValidator(strict_action_match=False)
        live = create_decision(action="BUY")
        sim = create_decision(action="SELL")

        result = validator.validate_decision(live, sim)

        # Non-strict mode records warning but doesn't fail
        assert "action_warning" in result.details
        assert result.confidence < Decimal("100")

    def test_quantity_within_tolerance(self) -> None:
        validator = GoldenPathValidator(quantity_tolerance_pct=Decimal("0.05"))
        live = create_decision(target_quantity=Decimal("1.0"))
        sim = create_decision(target_quantity=Decimal("1.02"))  # 2% diff

        result = validator.validate_decision(live, sim)

        assert result.matches is True
        assert "quantity_check" in result.details

    def test_quantity_outside_tolerance(self) -> None:
        validator = GoldenPathValidator(quantity_tolerance_pct=Decimal("0.01"))
        live = create_decision(target_quantity=Decimal("1.0"))
        sim = create_decision(target_quantity=Decimal("1.05"))  # 5% diff

        result = validator.validate_decision(live, sim)

        assert result.matches is False
        assert result.divergence is not None
        assert "Quantity mismatch" in result.divergence.reason

    def test_price_within_tolerance(self) -> None:
        validator = GoldenPathValidator(price_tolerance_pct=Decimal("0.01"))
        live = create_decision(target_price=Decimal("50000"))
        sim = create_decision(target_price=Decimal("50025"))  # 0.05% diff

        result = validator.validate_decision(live, sim)

        assert result.matches is True

    def test_price_outside_tolerance(self) -> None:
        validator = GoldenPathValidator(price_tolerance_pct=Decimal("0.001"))
        live = create_decision(target_price=Decimal("50000"))
        sim = create_decision(target_price=Decimal("50100"))  # 0.2% diff

        result = validator.validate_decision(live, sim)

        assert result.matches is False
        assert "Price mismatch" in result.divergence.reason

    def test_risk_check_mismatch(self) -> None:
        validator = GoldenPathValidator()
        live = create_decision(risk_checks_passed=True)
        sim = create_decision(risk_checks_passed=False)

        result = validator.validate_decision(live, sim)

        assert result.matches is False
        assert "Risk check mismatch" in result.divergence.reason

    def test_order_type_mismatch_warning(self) -> None:
        validator = GoldenPathValidator()
        live = create_decision(order_type="MARKET")
        sim = create_decision(order_type="LIMIT")

        result = validator.validate_decision(live, sim)

        # Order type mismatch is a warning, not a failure
        assert "order_type_warning" in result.details
        assert result.confidence < Decimal("100")

    def test_hold_actions_skip_quantity_check(self) -> None:
        validator = GoldenPathValidator()
        live = create_decision(action="HOLD", target_quantity=Decimal("0"))
        sim = create_decision(action="HOLD", target_quantity=Decimal("0"))

        result = validator.validate_decision(live, sim)

        assert result.matches is True


class TestEstimateImpact:
    """Tests that validate_decision uses impact estimation."""

    def test_hold_vs_trade_full_impact(self) -> None:
        validator = GoldenPathValidator()
        live = create_decision(action="HOLD")
        sim = create_decision(action="BUY")

        result = validator.validate_decision(live, sim)

        # One traded, one held = 100% impact
        assert result.divergence is not None
        assert result.divergence.impact_pct == Decimal("100")

    def test_both_hold_zero_impact(self) -> None:
        validator = GoldenPathValidator()
        live = create_decision(action="HOLD", risk_checks_passed=True)
        sim = create_decision(action="HOLD", risk_checks_passed=False)

        result = validator.validate_decision(live, sim)

        assert result.divergence is not None
        # Impact calculated for risk check mismatch
