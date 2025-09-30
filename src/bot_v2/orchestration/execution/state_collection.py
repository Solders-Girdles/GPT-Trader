"""
State collection utilities for live trading execution.

This module handles collecting and transforming account state including
balances, positions, equity calculations, and collateral asset resolution.
"""

from __future__ import annotations

import logging
import os
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Balance, IBrokerage, Product
from bot_v2.utilities.quantities import quantity_from

logger = logging.getLogger(__name__)


class StateCollector:
    """Collects and transforms account state for execution and risk management."""

    def __init__(self, broker: IBrokerage) -> None:
        """
        Initialize state collector.

        Args:
            broker: Brokerage adapter
        """
        self.broker = broker
        self.collateral_assets = self._resolve_collateral_assets()

    def _resolve_collateral_assets(self) -> set[str]:
        """Resolve collateral assets from environment or use defaults."""
        env_value = os.getenv("PERPS_COLLATERAL_ASSETS") or ""
        default_assets = {"USD", "USDC"}
        parsed = {token.strip().upper() for token in env_value.split(",") if token.strip()}
        return parsed or set(default_assets)

    def calculate_equity_from_balances(
        self,
        balances: list[Balance],
    ) -> tuple[Decimal, list[Balance], Decimal]:
        """
        Calculate total equity from balance list.

        Args:
            balances: List of balance objects

        Returns:
            Tuple of (total_available, collateral_balances, total_balance)
        """
        total_available = Decimal("0")
        total_balance = Decimal("0")
        collateral_balances: list[Balance] = []

        for bal in balances:
            asset = (bal.asset or "").upper()
            if asset in self.collateral_assets:
                collateral_balances.append(bal)
                total_available += bal.available
                total_balance += bal.total

        if collateral_balances:
            return total_available, collateral_balances, total_balance

        usd_balance = next((bal for bal in balances if (bal.asset or "").upper() == "USD"), None)
        if usd_balance:
            return usd_balance.available, [usd_balance], usd_balance.total

        return Decimal("0"), [], Decimal("0")

    def collect_account_state(
        self,
    ) -> tuple[list[Balance], Decimal, list[Balance], Decimal, list[Any]]:
        """
        Collect complete account state from broker.

        Returns:
            Tuple of (balances, equity, collateral_balances, total_balance, positions)
        """
        balances = self.broker.list_balances()
        equity, collateral_balances, total_balance = self.calculate_equity_from_balances(balances)
        positions = self.broker.list_positions()

        return balances, equity, collateral_balances, total_balance, positions

    def build_positions_dict(self, positions: list[Any]) -> dict[str, dict[str, Any]]:
        """
        Build simplified position dictionary for validation.

        Args:
            positions: List of position objects

        Returns:
            Dictionary mapping symbol to position details
        """
        positions_dict: dict[str, dict[str, Any]] = {}
        for pos in positions:
            qty = quantity_from(pos)
            if qty is None or qty == Decimal("0"):
                continue
            try:
                positions_dict[pos.symbol] = {
                    "quantity": qty,
                    "side": getattr(pos, "side", "long").lower(),
                    "entry_price": Decimal(str(getattr(pos, "entry_price", "0"))),
                    "mark_price": Decimal(str(getattr(pos, "mark_price", "0"))),
                }
            except Exception as exc:
                logger.warning(f"Failed to parse position {pos.symbol}: {exc}")
                continue
        return positions_dict

    def resolve_effective_price(
        self,
        symbol: str,
        side: str,
        price: Decimal | None,
        product: Product,
    ) -> Decimal:
        """
        Resolve effective price for order validation.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            price: User-provided price (None for market orders)
            product: Product specifications

        Returns:
            Effective price to use for validation
        """
        if price is not None and price > Decimal("0"):
            return price

        # For market orders, use mark price or bid/ask
        if hasattr(self.broker, "get_mark_price"):
            try:
                mark = self.broker.get_mark_price(symbol)  # type: ignore[attr-defined]
                if mark and mark > Decimal("0"):
                    return Decimal(str(mark))
            except Exception:
                pass

        # Fallback to mid-price
        if hasattr(product, "bid_price") and hasattr(product, "ask_price"):
            if product.bid_price and product.ask_price:
                bid = Decimal(str(product.bid_price))
                ask = Decimal(str(product.ask_price))
                if bid > Decimal("0") and ask > Decimal("0"):
                    return (bid + ask) / Decimal("2")

        # Last resort: use last price or quote_increment
        if hasattr(product, "price") and product.price:
            return Decimal(str(product.price))

        # If all else fails, use a default based on quote_increment
        if hasattr(product, "quote_increment") and product.quote_increment:
            quote_inc = Decimal(str(product.quote_increment))
        else:
            quote_inc = Decimal("0.01")
        return quote_inc * Decimal("100")

    def require_product(self, symbol: str, product: Product | None) -> Product:
        """
        Ensure product specification is available.

        Args:
            symbol: Trading symbol
            product: Product object or None

        Returns:
            Valid product object

        Raises:
            ValidationError: If product cannot be resolved
        """
        if product is not None:
            return product

        # Try to fetch from broker
        fetched = self.broker.get_product(symbol)
        if fetched is None:
            # Import here to avoid circular dependency
            from bot_v2.features.live_trade.risk import ValidationError

            raise ValidationError(f"Product not found: {symbol}")
        return fetched
