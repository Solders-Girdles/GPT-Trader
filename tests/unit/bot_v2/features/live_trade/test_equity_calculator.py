"""
Unit tests for EquityCalculator.

Tests cash balance aggregation and total equity calculation
with multi-currency support and edge cases.
"""

from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import Balance
from bot_v2.features.live_trade.equity_calculator import EquityCalculator


class TestCashBalanceCalculation:
    """Tests for cash balance aggregation."""

    def test_single_currency_usd(self):
        """Test cash balance with single USD currency."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("10000"), total=Decimal("10000"), hold=Decimal("0")
            )
        }

        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("10000")

    def test_single_currency_usdc(self):
        """Test cash balance with single USDC currency."""
        balances = {
            "USDC": Balance(
                asset="USDC", available=Decimal("5000"), total=Decimal("5000"), hold=Decimal("0")
            )
        }

        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("5000")

    def test_multiple_stablecoins(self):
        """Test cash balance aggregates multiple stablecoins."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("5000"), total=Decimal("5000"), hold=Decimal("0")
            ),
            "USDC": Balance(
                asset="USDC", available=Decimal("3000"), total=Decimal("3000"), hold=Decimal("0")
            ),
            "USDT": Balance(
                asset="USDT", available=Decimal("2000"), total=Decimal("2000"), hold=Decimal("0")
            ),
        }

        cash = EquityCalculator.calculate_cash_balance(balances)

        # Should sum all three: 5000 + 3000 + 2000 = 10000
        assert cash == Decimal("10000")

    def test_empty_balances(self):
        """Test cash balance with empty balances dict."""
        balances = {}

        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("0")

    def test_missing_currencies(self):
        """Test cash balance when some currencies missing."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("5000"), total=Decimal("5000"), hold=Decimal("0")
            ),
            # USDC and USDT missing
        }

        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("5000")

    def test_zero_balance_currencies(self):
        """Test cash balance with zero balance currencies."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("0"), total=Decimal("0"), hold=Decimal("0")
            ),
            "USDC": Balance(
                asset="USDC", available=Decimal("1000"), total=Decimal("1000"), hold=Decimal("0")
            ),
        }

        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("1000")

    def test_custom_currencies(self):
        """Test cash balance with custom currency list."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("5000"), total=Decimal("5000"), hold=Decimal("0")
            ),
            "EUR": Balance(
                asset="EUR", available=Decimal("3000"), total=Decimal("3000"), hold=Decimal("0")
            ),
        }

        # Only include USD
        cash = EquityCalculator.calculate_cash_balance(balances, currencies=["USD"])

        assert cash == Decimal("5000")

    def test_non_stablecoin_ignored(self):
        """Test non-stablecoin currencies are ignored by default."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("5000"), total=Decimal("5000"), hold=Decimal("0")
            ),
            "BTC": Balance(
                asset="BTC", available=Decimal("1.0"), total=Decimal("1.0"), hold=Decimal("0")
            ),
        }

        # BTC should be ignored (not in default stablecoin list)
        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("5000")


class TestTotalEquityCalculation:
    """Tests for total equity calculation."""

    def test_equity_cash_only(self):
        """Test equity with cash only, no PnL."""
        equity = EquityCalculator.calculate_total_equity(cash_balance=Decimal("10000"))

        assert equity == Decimal("10000")

    def test_equity_with_unrealized_pnl_profit(self):
        """Test equity includes unrealized profit."""
        equity = EquityCalculator.calculate_total_equity(
            cash_balance=Decimal("10000"),
            unrealized_pnl=Decimal("2000"),
        )

        assert equity == Decimal("12000")

    def test_equity_with_unrealized_pnl_loss(self):
        """Test equity includes unrealized loss."""
        equity = EquityCalculator.calculate_total_equity(
            cash_balance=Decimal("10000"),
            unrealized_pnl=Decimal("-3000"),
        )

        assert equity == Decimal("7000")

    def test_equity_with_realized_pnl(self):
        """Test equity includes realized PnL."""
        equity = EquityCalculator.calculate_total_equity(
            cash_balance=Decimal("10000"),
            realized_pnl=Decimal("1500"),
        )

        assert equity == Decimal("11500")

    def test_equity_with_funding_pnl(self):
        """Test equity includes funding fees."""
        equity = EquityCalculator.calculate_total_equity(
            cash_balance=Decimal("10000"),
            funding_pnl=Decimal("-250"),
        )

        assert equity == Decimal("9750")

    def test_equity_with_all_components(self):
        """Test equity with all PnL components."""
        equity = EquityCalculator.calculate_total_equity(
            cash_balance=Decimal("10000"),
            unrealized_pnl=Decimal("2000"),
            realized_pnl=Decimal("1500"),
            funding_pnl=Decimal("-250"),
        )

        # 10000 + 2000 + 1500 - 250 = 13250
        assert equity == Decimal("13250")

    def test_equity_from_pnl_dict(self):
        """Test equity calculation from PnL tracker dict."""
        total_pnl = {
            "total": Decimal("3000"),
            "unrealized": Decimal("2000"),
            "realized": Decimal("1000"),
            "funding": Decimal("0"),
        }

        equity = EquityCalculator.calculate_equity_from_pnl_dict(
            cash_balance=Decimal("10000"),
            total_pnl=total_pnl,
        )

        assert equity == Decimal("13000")

    def test_equity_negative_result(self):
        """Test equity can be negative with large losses."""
        equity = EquityCalculator.calculate_total_equity(
            cash_balance=Decimal("5000"),
            unrealized_pnl=Decimal("-8000"),
        )

        assert equity == Decimal("-3000")


class TestEquityBreakdown:
    """Tests for equity breakdown."""

    def test_equity_breakdown_complete(self):
        """Test equity breakdown with all components."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("10000"), total=Decimal("10000"), hold=Decimal("0")
            ),
        }

        breakdown = EquityCalculator.get_equity_breakdown(
            balances=balances,
            unrealized_pnl=Decimal("2000"),
            realized_pnl=Decimal("1500"),
            funding_pnl=Decimal("-250"),
        )

        assert breakdown["cash"] == Decimal("10000")
        assert breakdown["unrealized_pnl"] == Decimal("2000")
        assert breakdown["realized_pnl"] == Decimal("1500")
        assert breakdown["funding_pnl"] == Decimal("-250")
        assert breakdown["total"] == Decimal("13250")

    def test_equity_breakdown_multiple_currencies(self):
        """Test equity breakdown with multiple currencies."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("5000"), total=Decimal("5000"), hold=Decimal("0")
            ),
            "USDC": Balance(
                asset="USDC", available=Decimal("3000"), total=Decimal("3000"), hold=Decimal("0")
            ),
        }

        breakdown = EquityCalculator.get_equity_breakdown(
            balances=balances,
            unrealized_pnl=Decimal("1000"),
            realized_pnl=Decimal("500"),
            funding_pnl=Decimal("-100"),
        )

        assert breakdown["cash"] == Decimal("8000")  # 5000 + 3000
        assert breakdown["total"] == Decimal("9400")  # 8000 + 1000 + 500 - 100

    def test_equity_breakdown_zero_pnl(self):
        """Test equity breakdown with zero PnL components."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("10000"), total=Decimal("10000"), hold=Decimal("0")
            ),
        }

        breakdown = EquityCalculator.get_equity_breakdown(
            balances=balances,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            funding_pnl=Decimal("0"),
        )

        assert breakdown["cash"] == Decimal("10000")
        assert breakdown["total"] == Decimal("10000")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_balance(self):
        """Test with very large balance."""
        balances = {
            "USD": Balance(
                asset="USD",
                available=Decimal("999999999.99"),
                total=Decimal("999999999.99"),
                hold=Decimal("0"),
            ),
        }

        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("999999999.99")

    def test_very_small_balance(self):
        """Test with very small balance."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("0.01"), total=Decimal("0.01"), hold=Decimal("0")
            ),
        }

        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("0.01")

    def test_precision_maintained(self):
        """Test decimal precision is maintained."""
        balances = {
            "USD": Balance(
                asset="USD",
                available=Decimal("1.123456789"),
                total=Decimal("1.123456789"),
                hold=Decimal("0"),
            ),
        }

        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("1.123456789")

    def test_multiple_currency_precision(self):
        """Test precision maintained with multiple currencies."""
        balances = {
            "USD": Balance(
                asset="USD", available=Decimal("1.11"), total=Decimal("1.11"), hold=Decimal("0")
            ),
            "USDC": Balance(
                asset="USDC", available=Decimal("2.22"), total=Decimal("2.22"), hold=Decimal("0")
            ),
            "USDT": Balance(
                asset="USDT", available=Decimal("3.33"), total=Decimal("3.33"), hold=Decimal("0")
            ),
        }

        cash = EquityCalculator.calculate_cash_balance(balances)

        assert cash == Decimal("6.66")
