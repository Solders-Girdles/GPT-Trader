"""Tests for backtesting types module."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from gpt_trader.backtesting.types import (
    SimulationConfig,
    ValidationReport,
)


class TestSimulationConfigPostInit:
    """Tests for SimulationConfig __post_init__."""

    def test_sets_default_slippage_when_none(self) -> None:
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            granularity="ONE_MINUTE",
            initial_equity_usd=Decimal("10000"),
            slippage_bps=None,
        )
        assert config.slippage_bps == {}

    def test_preserves_custom_slippage(self) -> None:
        custom_slippage = {"BTC-USD": Decimal("2"), "ETH-USD": Decimal("3")}
        config = SimulationConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            granularity="ONE_MINUTE",
            initial_equity_usd=Decimal("10000"),
            slippage_bps=custom_slippage,
        )
        assert config.slippage_bps == custom_slippage


class TestValidationReportMatchRate:
    """Tests for ValidationReport.match_rate property."""

    def test_returns_100_when_no_decisions(self) -> None:
        report = ValidationReport(
            cycle_id="test-cycle",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            total_decisions=0,
            matching_decisions=0,
            divergences=[],
        )
        assert report.match_rate == Decimal("100")

    def test_calculates_match_rate_with_decisions(self) -> None:
        report = ValidationReport(
            cycle_id="test-cycle",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            total_decisions=10,
            matching_decisions=8,
            divergences=[],
        )
        assert report.match_rate == Decimal("80")

    def test_calculates_perfect_match_rate(self) -> None:
        report = ValidationReport(
            cycle_id="test-cycle",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            total_decisions=100,
            matching_decisions=100,
            divergences=[],
        )
        assert report.match_rate == Decimal("100")
