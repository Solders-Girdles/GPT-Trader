"""
Equity Calculator Component.

Handles cash balance aggregation and total equity calculation
with multi-currency support.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot_v2.features.brokerages.core.interfaces import Balance

logger = logging.getLogger(__name__)


class EquityCalculator:
    """
    Cash and equity calculator with multi-currency support.

    Stateless helper for aggregating cash balances across stablecoins
    and calculating total equity including unrealized PnL.
    """

    # Supported stablecoin currencies (treated as USD equivalent)
    STABLECOIN_CURRENCIES = ["USD", "USDC", "USDT"]

    @staticmethod
    def calculate_cash_balance(
        balances: dict[str, Balance],
        currencies: list[str] | None = None,
    ) -> Decimal:
        """
        Calculate total cash balance across currencies.

        Args:
            balances: Map of currency to Balance
            currencies: List of currencies to include (default: USD/USDC/USDT)

        Returns:
            Total cash balance (sum of available balances)
        """
        if currencies is None:
            currencies = EquityCalculator.STABLECOIN_CURRENCIES

        cash_balance = Decimal("0")

        for currency in currencies:
            if currency in balances:
                cash_balance += balances[currency].available

        return cash_balance

    @staticmethod
    def calculate_total_equity(
        cash_balance: Decimal,
        unrealized_pnl: Decimal = Decimal("0"),
        realized_pnl: Decimal = Decimal("0"),
        funding_pnl: Decimal = Decimal("0"),
    ) -> Decimal:
        """
        Calculate total equity (cash + all PnL components).

        Args:
            cash_balance: Available cash
            unrealized_pnl: Unrealized PnL from open positions
            realized_pnl: Realized PnL from closed trades
            funding_pnl: Funding fees paid/received

        Returns:
            Total equity (cash + unrealized + realized + funding)
        """
        return cash_balance + unrealized_pnl + realized_pnl + funding_pnl

    @staticmethod
    def calculate_equity_from_pnl_dict(
        cash_balance: Decimal,
        total_pnl: dict[str, Decimal],
    ) -> Decimal:
        """
        Calculate total equity from PnL tracker dict.

        Args:
            cash_balance: Available cash
            total_pnl: PnL dict with 'total', 'unrealized', 'realized', 'funding' keys

        Returns:
            Total equity (cash + total PnL)
        """
        return cash_balance + total_pnl["total"]

    @staticmethod
    def get_equity_breakdown(
        balances: dict[str, Balance],
        unrealized_pnl: Decimal,
        realized_pnl: Decimal,
        funding_pnl: Decimal,
    ) -> dict[str, Decimal]:
        """
        Get detailed equity breakdown.

        Args:
            balances: Map of currency to Balance
            unrealized_pnl: Unrealized PnL
            realized_pnl: Realized PnL
            funding_pnl: Funding PnL

        Returns:
            Dict with cash, unrealized, realized, funding, total
        """
        cash_balance = EquityCalculator.calculate_cash_balance(balances)
        total_equity = EquityCalculator.calculate_total_equity(
            cash_balance, unrealized_pnl, realized_pnl, funding_pnl
        )

        return {
            "cash": cash_balance,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "funding_pnl": funding_pnl,
            "total": total_equity,
        }
