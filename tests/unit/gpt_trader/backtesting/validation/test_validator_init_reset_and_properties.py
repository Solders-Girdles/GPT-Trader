"""Initialization/reset/property tests for GoldenPathValidator."""

from __future__ import annotations

from decimal import Decimal

from tests.unit.gpt_trader.backtesting.validation.validator_test_helpers import create_decision

from gpt_trader.backtesting.validation.validator import GoldenPathValidator


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


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_state(self) -> None:
        validator = GoldenPathValidator()

        for _ in range(3):
            live = create_decision(action="BUY")
            sim = create_decision(action="SELL")
            validator.validate_decision(live, sim)

        assert validator.total_comparisons == 3
        assert validator.divergence_count == 3

        validator.reset()

        assert validator.total_comparisons == 0
        assert validator.matching_comparisons == 0
        assert validator.divergence_count == 0


class TestProperties:
    """Tests for validator properties."""

    def test_total_comparisons_property(self) -> None:
        validator = GoldenPathValidator()
        assert validator.total_comparisons == 0

        validator.validate_decision(create_decision(), create_decision())
        assert validator.total_comparisons == 1

    def test_matching_comparisons_property(self) -> None:
        validator = GoldenPathValidator()
        validator.validate_decision(create_decision(), create_decision())
        assert validator.matching_comparisons == 1

    def test_divergence_count_property(self) -> None:
        validator = GoldenPathValidator()
        live = create_decision(action="BUY")
        sim = create_decision(action="SELL")
        validator.validate_decision(live, sim)
        assert validator.divergence_count == 1
