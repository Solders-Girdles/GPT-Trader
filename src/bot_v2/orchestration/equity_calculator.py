"""Equity calculation for portfolio valuation including open positions.

This module provides utilities for calculating total portfolio equity by combining
cash balances with the market value of open positions.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bot_v2.features.brokerages.core.interfaces import Balance

logger = logging.getLogger(__name__)


class EquityCalculator:
    """Calculates portfolio equity including open positions.

    This calculator is responsible for determining the total value of a trading
    portfolio by combining:
    1. Cash balances (USD/USDC)
    2. Value of open positions at current market prices

    Example:
        >>> calculator = EquityCalculator()
        >>> balances = [Balance(asset="USDC", total=Decimal("10000"))]
        >>> equity = calculator.calculate(
        ...     balances=balances,
        ...     position_quantity=Decimal("0.5"),  # 0.5 BTC
        ...     current_mark=Decimal("50000"),     # BTC at $50k
        ... )
        >>> equity
        Decimal('35000')  # $10k cash + 0.5 * $50k = $35k
    """

    # Assets considered as cash for equity calculations
    CASH_ASSETS = {"USD", "USDC"}

    def calculate(
        self,
        balances: Sequence[Balance],
        position_quantity: Decimal,
        current_mark: Decimal | None = None,
        symbol: str | None = None,
    ) -> Decimal:
        """Calculate total equity (cash + position value).

        Args:
            balances: Broker balance snapshot
            position_quantity: Current position size (positive or negative, can be 0)
            current_mark: Current market price for position valuation
            symbol: Symbol for logging (optional)

        Returns:
            Total equity in USD/USDC

        Note:
            - Uses absolute value of position_quantity for valuation
            - Returns cash-only equity if current_mark is None
            - Handles errors gracefully with debug logging

        Example:
            >>> balances = [Balance(asset="USD", total=Decimal("5000"))]
            >>> calculator.calculate(balances, Decimal("2"), Decimal("100"))
            Decimal('5200')  # $5000 + 2 * $100
        """
        # Start with cash balance
        equity = self.extract_cash_balance(balances)

        # Add position value if applicable
        if position_quantity and current_mark is not None:
            try:
                position_value = abs(position_quantity) * current_mark
                equity += position_value
            except Exception as exc:
                log_symbol = f" for {symbol}" if symbol else ""
                logger.debug(
                    "Failed to adjust equity%s position: %s",
                    log_symbol,
                    exc,
                    exc_info=True,
                )

        return equity

    def extract_cash_balance(self, balances: Sequence[Balance]) -> Decimal:
        """Extract USD/USDC balance from broker balances.

        Searches for the first balance with an asset in CASH_ASSETS (USD, USDC)
        and returns its total value.

        Args:
            balances: Sequence of Balance objects from broker

        Returns:
            Total cash balance in USD/USDC, or Decimal("0") if none found

        Example:
            >>> balances = [
            ...     Balance(asset="BTC", total=Decimal("1.5")),
            ...     Balance(asset="USDC", total=Decimal("10000")),
            ...     Balance(asset="ETH", total=Decimal("10")),
            ... ]
            >>> calculator.extract_cash_balance(balances)
            Decimal('10000')
        """
        usd_balance = next(
            (b for b in balances if getattr(b, "asset", "").upper() in self.CASH_ASSETS),
            None,
        )
        return usd_balance.total if usd_balance else Decimal("0")
