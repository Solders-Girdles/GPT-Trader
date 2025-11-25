"""Type helpers for Coinbase REST service mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
    from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
    from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
    from gpt_trader.features.brokerages.coinbase.utilities import PositionState
    from gpt_trader.persistence.event_store import EventStore


class CoinbaseRestServiceProtocol(Protocol):
    """Protocol describing attributes available to REST mixins.

    This defines the interface that the mixins expect from the base class.
    """

    client: CoinbaseClient
    endpoints: CoinbaseEndpoints
    _event_store: EventStore
    market_data: MarketDataService
    _positions: dict[str, PositionState]

    @property
    def positions(self) -> dict[str, PositionState]:
        """Get positions dictionary."""
        ...

    def _build_order_payload(
        self,
        symbol: str,
        side: Any,
        order_type: Any,
        quantity: Any,
        price: Any | None = None,
        stop_price: Any | None = None,
        tif: Any = None,
        client_id: str | None = None,
        reduce_only: bool = False,
        leverage: int | None = None,
        post_only: bool = False,
        include_client_id: bool = True,
    ) -> dict[str, Any]:
        """Build order payload."""
        ...


__all__ = ["CoinbaseRestServiceProtocol"]
