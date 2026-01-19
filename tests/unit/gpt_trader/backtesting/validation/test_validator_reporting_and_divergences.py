"""Match-rate and reporting tests for GoldenPathValidator."""

from __future__ import annotations

from decimal import Decimal

from tests.unit.gpt_trader.backtesting.validation.validator_test_helpers import create_decision

from gpt_trader.backtesting.validation.validator import GoldenPathValidator


class TestGetMatchRate:
    """Tests for get_match_rate method."""

    def test_no_comparisons_returns_100(self) -> None:
        validator = GoldenPathValidator()
        assert validator.get_match_rate() == Decimal("100")

    def test_all_matches(self) -> None:
        validator = GoldenPathValidator()
        for _ in range(5):
            validator.validate_decision(create_decision(), create_decision())

        assert validator.get_match_rate() == Decimal("100")

    def test_partial_matches(self) -> None:
        validator = GoldenPathValidator()

        for _ in range(3):
            validator.validate_decision(create_decision(), create_decision())

        for _ in range(2):
            live = create_decision(action="BUY")
            sim = create_decision(action="SELL")
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

        validator.validate_decision(
            create_decision(action="BUY", symbol="BTC-USD"),
            create_decision(action="SELL", symbol="BTC-USD"),
        )

        validator.validate_decision(
            create_decision(action="BUY", symbol="ETH-USD"),
            create_decision(action="SELL", symbol="ETH-USD"),
        )

        btc_divergences = validator.get_divergences(symbol="BTC-USD")
        assert len(btc_divergences) == 1
        assert btc_divergences[0].symbol == "BTC-USD"

    def test_filter_by_min_impact(self) -> None:
        validator = GoldenPathValidator()

        validator.validate_decision(create_decision(action="HOLD"), create_decision(action="BUY"))

        validator.validate_decision(
            create_decision(target_quantity=Decimal("1.0")),
            create_decision(target_quantity=Decimal("1.05")),
        )

        high_impact = validator.get_divergences(min_impact=Decimal("50"))
        assert len(high_impact) == 1


class TestGenerateReport:
    """Tests for generate_report method."""

    def test_generate_report_structure(self) -> None:
        validator = GoldenPathValidator()

        for _ in range(5):
            live = create_decision(cycle_id="cycle-001")
            sim = create_decision(cycle_id="cycle-001")
            validator.validate_decision(live, sim)

        report = validator.generate_report("cycle-001")

        assert report.cycle_id == "cycle-001"
        assert report.total_decisions == 5
        assert report.matching_decisions == 5
        assert report.divergences == []

    def test_report_includes_divergences(self) -> None:
        validator = GoldenPathValidator()

        live = create_decision(action="BUY", cycle_id="cycle-002")
        sim = create_decision(action="SELL", cycle_id="cycle-002")
        validator.validate_decision(live, sim)

        report = validator.generate_report("cycle-002")

        assert len(report.divergences) == 1
        assert report.divergences[0].cycle_id == "cycle-002"
