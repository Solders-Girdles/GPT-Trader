"""Tests for golden-path validator ValidationResult."""

from __future__ import annotations

from decimal import Decimal

from tests.unit.gpt_trader.backtesting.validation.validator_test_helpers import create_decision

from gpt_trader.backtesting.validation.validator import ValidationResult


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_matching_result(self) -> None:
        live = create_decision()
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
        live = create_decision(action="BUY")
        sim = create_decision(action="SELL")
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
        live = create_decision()
        result = ValidationResult(
            matches=True,
            live_decision=live,
            details={"check1": "passed", "check2": "within tolerance"},
        )
        assert "check1" in result.details
        assert result.details["check1"] == "passed"
