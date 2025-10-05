"""Tests for EquityCalculator - portfolio equity calculation.

This module tests the EquityCalculator's ability to:
- Extract cash balances from broker balance lists
- Calculate total equity including position values
- Handle edge cases (zero positions, missing data, errors)
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance
from bot_v2.features.live_trade.equity import EquityCalculator


@pytest.fixture
def calculator() -> EquityCalculator:
    """Create EquityCalculator instance."""
    return EquityCalculator()


@pytest.fixture
def sample_balances_usd() -> list[Balance]:
    """Create sample balances with USD."""
    return [
        Mock(spec=Balance, asset="BTC", total=Decimal("1.5")),
        Mock(spec=Balance, asset="USD", total=Decimal("10000")),
        Mock(spec=Balance, asset="ETH", total=Decimal("10")),
    ]


@pytest.fixture
def sample_balances_usdc() -> list[Balance]:
    """Create sample balances with USDC."""
    return [
        Mock(spec=Balance, asset="BTC", total=Decimal("0.5")),
        Mock(spec=Balance, asset="USDC", total=Decimal("5000")),
    ]


class TestExtractCashBalance:
    """Test cash balance extraction from broker balances."""

    def test_extracts_usd_balance(
        self, calculator: EquityCalculator, sample_balances_usd: list[Balance]
    ) -> None:
        """Extracts USD balance when present."""
        cash = calculator.extract_cash_balance(sample_balances_usd)

        assert cash == Decimal("10000")

    def test_extracts_usdc_balance(
        self, calculator: EquityCalculator, sample_balances_usdc: list[Balance]
    ) -> None:
        """Extracts USDC balance when present."""
        cash = calculator.extract_cash_balance(sample_balances_usdc)

        assert cash == Decimal("5000")

    def test_returns_zero_when_no_cash_balance(self, calculator: EquityCalculator) -> None:
        """Returns zero when no USD/USDC balance found."""
        balances = [
            Mock(spec=Balance, asset="BTC", total=Decimal("1.0")),
            Mock(spec=Balance, asset="ETH", total=Decimal("10")),
        ]

        cash = calculator.extract_cash_balance(balances)

        assert cash == Decimal("0")

    def test_returns_zero_for_empty_balance_list(self, calculator: EquityCalculator) -> None:
        """Returns zero for empty balance list."""
        cash = calculator.extract_cash_balance([])

        assert cash == Decimal("0")

    def test_handles_case_insensitive_asset_names(self, calculator: EquityCalculator) -> None:
        """Handles lowercase/mixed-case asset names."""
        balances = [
            Mock(spec=Balance, asset="usd", total=Decimal("1000")),
        ]

        cash = calculator.extract_cash_balance(balances)

        assert cash == Decimal("1000")

    def test_prefers_first_matching_cash_asset(self, calculator: EquityCalculator) -> None:
        """Returns first matching cash asset when multiple present."""
        balances = [
            Mock(spec=Balance, asset="USD", total=Decimal("1000")),
            Mock(spec=Balance, asset="USDC", total=Decimal("2000")),
        ]

        cash = calculator.extract_cash_balance(balances)

        # Should return first match (USD)
        assert cash == Decimal("1000")


class TestCalculateEquity:
    """Test total equity calculation."""

    def test_calculates_cash_only_equity(
        self, calculator: EquityCalculator, sample_balances_usd: list[Balance]
    ) -> None:
        """Calculates equity for cash-only portfolio."""
        equity = calculator.calculate(
            balances=sample_balances_usd,
            position_quantity=Decimal("0"),
            current_mark=None,
        )

        assert equity == Decimal("10000")

    def test_calculates_equity_with_long_position(
        self, calculator: EquityCalculator, sample_balances_usd: list[Balance]
    ) -> None:
        """Calculates equity including long position value."""
        equity = calculator.calculate(
            balances=sample_balances_usd,
            position_quantity=Decimal("0.5"),  # 0.5 BTC
            current_mark=Decimal("50000"),  # BTC at $50k
        )

        # $10k cash + 0.5 * $50k = $35k
        assert equity == Decimal("35000")

    def test_calculates_equity_with_short_position(
        self, calculator: EquityCalculator, sample_balances_usd: list[Balance]
    ) -> None:
        """Calculates equity using absolute value for short positions."""
        equity = calculator.calculate(
            balances=sample_balances_usd,
            position_quantity=Decimal("-0.5"),  # Short 0.5 BTC
            current_mark=Decimal("50000"),
        )

        # $10k cash + abs(-0.5) * $50k = $35k
        # Note: Short positions still contribute to equity via abs()
        assert equity == Decimal("35000")

    def test_ignores_position_when_mark_is_none(
        self, calculator: EquityCalculator, sample_balances_usd: list[Balance]
    ) -> None:
        """Returns cash-only equity when current_mark is None."""
        equity = calculator.calculate(
            balances=sample_balances_usd,
            position_quantity=Decimal("1.0"),
            current_mark=None,
        )

        # Should only return cash since mark is None
        assert equity == Decimal("10000")

    def test_ignores_position_when_quantity_is_zero(
        self, calculator: EquityCalculator, sample_balances_usd: list[Balance]
    ) -> None:
        """Returns cash-only equity when position_quantity is zero."""
        equity = calculator.calculate(
            balances=sample_balances_usd,
            position_quantity=Decimal("0"),
            current_mark=Decimal("50000"),
        )

        assert equity == Decimal("10000")

    def test_handles_calculation_error_gracefully(
        self, calculator: EquityCalculator, sample_balances_usd: list[Balance], caplog
    ) -> None:
        """Handles calculation errors and logs them."""
        # Create a mock that raises exception when multiplied
        bad_mark = Mock()
        bad_mark.__mul__ = Mock(side_effect=ValueError("Invalid multiplication"))

        with caplog.at_level("DEBUG"):
            equity = calculator.calculate(
                balances=sample_balances_usd,
                position_quantity=Decimal("1.0"),
                current_mark=bad_mark,
                symbol="BTC-USD",
            )

        # Should return cash-only equity on error
        assert equity == Decimal("10000")
        assert "Failed to adjust equity for BTC-USD position" in caplog.text

    def test_logs_symbol_in_error_message(
        self, calculator: EquityCalculator, sample_balances_usd: list[Balance], caplog
    ) -> None:
        """Includes symbol in error log when provided."""
        bad_mark = Mock()
        bad_mark.__mul__ = Mock(side_effect=ValueError("Test error"))

        with caplog.at_level("DEBUG"):
            calculator.calculate(
                balances=sample_balances_usd,
                position_quantity=Decimal("1.0"),
                current_mark=bad_mark,
                symbol="ETH-USD",
            )

        assert "ETH-USD" in caplog.text

    def test_omits_symbol_from_error_when_not_provided(
        self, calculator: EquityCalculator, sample_balances_usd: list[Balance], caplog
    ) -> None:
        """Omits symbol from error log when not provided."""
        bad_mark = Mock()
        bad_mark.__mul__ = Mock(side_effect=ValueError("Test error"))

        with caplog.at_level("DEBUG"):
            calculator.calculate(
                balances=sample_balances_usd,
                position_quantity=Decimal("1.0"),
                current_mark=bad_mark,
                symbol=None,
            )

        # Should still log error but without symbol
        assert "Failed to adjust equity" in caplog.text

    def test_handles_multiple_positions_scenario(self, calculator: EquityCalculator) -> None:
        """Calculates equity correctly for realistic scenario."""
        balances = [
            Mock(spec=Balance, asset="USDC", total=Decimal("25000")),
            Mock(spec=Balance, asset="BTC", total=Decimal("0")),
        ]

        # Portfolio: $25k cash + 1.5 BTC position at $40k
        equity = calculator.calculate(
            balances=balances,
            position_quantity=Decimal("1.5"),
            current_mark=Decimal("40000"),
        )

        # $25k + 1.5 * $40k = $85k
        assert equity == Decimal("85000")


class TestCashAssetsConstant:
    """Test CASH_ASSETS constant."""

    def test_cash_assets_includes_usd_and_usdc(self, calculator: EquityCalculator) -> None:
        """CASH_ASSETS constant includes USD and USDC."""
        assert "USD" in calculator.CASH_ASSETS
        assert "USDC" in calculator.CASH_ASSETS

    def test_cash_assets_is_set(self, calculator: EquityCalculator) -> None:
        """CASH_ASSETS is a set for efficient lookup."""
        assert isinstance(calculator.CASH_ASSETS, set)
