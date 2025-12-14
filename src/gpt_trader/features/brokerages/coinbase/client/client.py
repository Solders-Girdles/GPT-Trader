"""
Unified Coinbase Client.
Combines all mixins into a single simple client class.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from gpt_trader.core import Balance, Position
from gpt_trader.features.brokerages.coinbase.client.accounts import AccountClientMixin
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.market import MarketDataClientMixin
from gpt_trader.features.brokerages.coinbase.client.orders import OrderClientMixin
from gpt_trader.features.brokerages.coinbase.client.portfolio import PortfolioClientMixin
from gpt_trader.features.brokerages.coinbase.client.websocket_mixin import (
    WebSocketClientMixin,
)
from gpt_trader.features.brokerages.coinbase.errors import (
    AuthError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__)


class CoinbaseClient(
    CoinbaseClientBase,
    MarketDataClientMixin,
    OrderClientMixin,
    AccountClientMixin,
    PortfolioClientMixin,
    WebSocketClientMixin,
):
    """
    The unified Coinbase Client.
    Inherits base HTTP machinery and specific endpoint mixins.
    Includes WebSocket streaming for real-time market data.
    """

    def list_positions(self) -> list[Position]:
        """Fetch positions and convert to domain objects."""
        # Call the mixin method (which returns dict)
        response = super().list_positions_raw()
        positions_data = response.get("positions", [])

        results = []
        for p in positions_data:
            size = Decimal(str(p.get("net_size", 0)))
            if size == 0:
                continue

            results.append(
                Position(
                    symbol=p.get("product_id", ""),
                    quantity=abs(size),
                    entry_price=Decimal(str(p.get("entry_price", 0))),
                    mark_price=Decimal(
                        str(p.get("mark_price", 0))
                    ),  # Note: API might not return mark
                    unrealized_pnl=Decimal(str(p.get("unrealized_pnl", 0))),
                    realized_pnl=Decimal("0"),  # Not always available in simple view
                    side="long" if size > 0 else "short",
                    leverage=int(p.get("leverage", 1)),
                )
            )
        return results

    def _warn_public_market_fallback(self, operation: str, error: Exception) -> None:
        if getattr(self, "_public_market_fallback_warned", False):
            return
        self._public_market_fallback_warned = True
        logger.warning(
            "Market data request unauthorized (%s). Falling back to public market endpoints. "
            "If you expected authenticated access, verify Coinbase API credentials/permissions. "
            "Error: %s",
            operation,
            error,
        )

    def get_ticker(self, product_id: str) -> dict[str, Any]:
        """Fetch ticker and normalize response to standard format."""
        try:
            # Prefer public market endpoints in Advanced Trade mode to keep market data
            # flowing even when account credentials are misconfigured.
            if self.api_mode == "advanced":
                try:
                    response = super().get_market_product_ticker(product_id)
                except (NotFoundError, InvalidRequestError):
                    # Some products may not be available via public endpoints.
                    response = super().get_ticker(product_id)
            else:
                response = super().get_ticker(product_id)
        except (AuthError, PermissionDeniedError) as e:
            if self.api_mode != "advanced":
                raise
            self._warn_public_market_fallback("ticker", e)
            response = super().get_market_product_ticker(product_id)

        # Normalize price
        price = response.get("price")
        if not price:
            trades = response.get("trades", [])
            if trades:
                price = trades[0].get("price")

        # Normalize bid/ask
        bid = response.get("bid") or response.get("best_bid")
        ask = response.get("ask") or response.get("best_ask")

        # Return normalized dict
        return {
            "product_id": product_id,
            "price": price or "0",
            "bid": bid or "0",
            "ask": ask or "0",
            "time": response.get("time") or (response.get("trades", [{}])[0].get("time")),
            # Include original data just in case
            **response,
        }

    def get_candles(
        self,
        product_id: str,
        granularity: str,
        limit: int = 300,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """Fetch historical candles, falling back to public endpoints when unauthenticated."""
        try:
            if self.api_mode == "advanced":
                try:
                    return super().get_market_product_candles(
                        product_id,
                        granularity,
                        limit,
                        start=start,
                        end=end,
                    )
                except (NotFoundError, InvalidRequestError):
                    return super().get_candles(
                        product_id,
                        granularity,
                        limit,
                        start=start,
                        end=end,
                    )

            return super().get_candles(
                product_id,
                granularity,
                limit,
                start=start,
                end=end,
            )
        except (AuthError, PermissionDeniedError) as e:
            if self.api_mode != "advanced":
                raise
            self._warn_public_market_fallback("candles", e)
            return super().get_market_product_candles(
                product_id,
                granularity,
                limit,
                start=start,
                end=end,
            )

    def list_balances(self) -> list[Balance]:
        """Fetch accounts and convert to domain objects."""
        response = self.get_accounts()
        accounts_data = response.get("accounts", [])

        # Log raw API response characteristics
        logger.debug(
            f"CoinbaseClient.list_balances: Raw API returned {len(accounts_data)} accounts"
        )
        if not accounts_data and response:
            logger.warning(
                f"API returned response but no accounts. Response keys: {list(response.keys())}"
            )

        results = []
        for a in accounts_data:
            currency = a.get("currency", "")
            available = Decimal(str(a.get("available_balance", {}).get("value", 0)))
            hold = Decimal(str(a.get("hold", {}).get("value", 0)))

            results.append(
                Balance(
                    asset=currency,
                    total=available + hold,
                    available=available,
                    hold=hold,
                )
            )
        return results


__all__ = ["CoinbaseClient"]
