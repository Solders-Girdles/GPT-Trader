"""Portfolio interface for backtesting and live trading."""

from decimal import Decimal
from typing import Protocol

from bot_v2.backtesting.types import PortfolioType
from bot_v2.features.brokerages.core.interfaces import Balance


class IPortfolio(Protocol):
    """
    Portfolio management interface.

    This interface abstracts balance tracking and portfolio transfers,
    allowing the same logic to work with both live exchange accounts
    and simulated portfolios.
    """

    async def balances(self) -> list[Balance]:
        """
        Get current asset balances across all portfolios.

        Returns:
            List of balance objects with available/hold amounts
        """
        ...

    async def balance(self, currency: str) -> Balance | None:
        """
        Get balance for a specific currency.

        Args:
            currency: Currency code (e.g., "USD", "USDC", "BTC")

        Returns:
            Balance object or None if not found
        """
        ...

    async def equity(self) -> Decimal:
        """
        Calculate total portfolio equity in USD.

        This includes:
        - Cash balances (converted to USD)
        - Position values at current mark prices
        - Unrealized PnL

        Returns:
            Total equity in USD
        """
        ...

    async def margin_info(self) -> dict[str, Decimal]:
        """
        Get margin information.

        Returns:
            Dictionary with keys:
            - total_equity: Total account equity
            - margin_used: Margin currently used by positions
            - margin_available: Available margin for new positions
            - margin_ratio: Used / total (0-1)
            - leverage: Effective leverage (position_value / equity)
        """
        ...

    async def transfer(
        self,
        amount: Decimal,
        currency: str,
        from_portfolio: PortfolioType,
        to_portfolio: PortfolioType,
    ) -> bool:
        """
        Transfer funds between portfolios.

        Note: Coinbase supports transfers between spot, futures, and perps portfolios.
        This is useful for moving collateral or isolating risk.

        Args:
            amount: Amount to transfer
            currency: Currency to transfer
            from_portfolio: Source portfolio type
            to_portfolio: Destination portfolio type

        Returns:
            True if transfer successful, False otherwise

        Raises:
            InsufficientFundsError: If source portfolio has insufficient funds
        """
        ...

    async def collateral_available(self) -> Decimal:
        """
        Get available collateral for new positions (USD).

        This is the amount that can be used for new margin positions,
        accounting for existing positions and margin requirements.

        Returns:
            Available collateral in USD
        """
        ...
