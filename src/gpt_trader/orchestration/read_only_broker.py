"""
Read-only broker wrapper for observation mode.

Allows market data fetching but blocks all order execution.
"""

from __future__ import annotations

from typing import Any


class ReadOnlyBroker:
    """
    Wrapper that allows market data fetching but blocks order execution.

    Perfect for strategy observation with zero execution risk. Delegates
    all read operations to the underlying broker while rejecting all
    write operations.
    """

    def __init__(self, underlying_broker: Any) -> None:
        """
        Initialize read-only broker wrapper.

        Args:
            underlying_broker: The actual broker instance to wrap
        """
        self._broker = underlying_broker

    # -------------------------------------------------------------------------
    # Read Operations (Delegated)
    # -------------------------------------------------------------------------

    def get_ticker(self, *args: Any, **kwargs: Any) -> Any:
        """Get ticker data from underlying broker."""
        return self._broker.get_ticker(*args, **kwargs)

    def get_quote(self, *args: Any, **kwargs: Any) -> Any:
        """Get quote data from underlying broker."""
        return self._broker.get_quote(*args, **kwargs)

    def get_candles(self, *args: Any, **kwargs: Any) -> Any:
        """Get historical candles from underlying broker."""
        return self._broker.get_candles(*args, **kwargs)

    def list_products(self, *args: Any, **kwargs: Any) -> Any:
        """List available products from underlying broker."""
        return self._broker.list_products(*args, **kwargs)

    def get_product(self, *args: Any, **kwargs: Any) -> Any:
        """Get product information from underlying broker."""
        return self._broker.get_product(*args, **kwargs)

    def list_positions(self, *args: Any, **kwargs: Any) -> Any:
        """List positions from underlying broker."""
        return self._broker.list_positions(*args, **kwargs)

    def get_positions(self, *args: Any, **kwargs: Any) -> Any:
        """Get positions from underlying broker."""
        return self._broker.get_positions(*args, **kwargs)

    def list_balances(self, *args: Any, **kwargs: Any) -> Any:
        """List balances from underlying broker."""
        return self._broker.list_balances(*args, **kwargs)

    def get_balances(self, *args: Any, **kwargs: Any) -> Any:
        """Get balances from underlying broker."""
        return self._broker.get_balances(*args, **kwargs)

    def get_equity(self, *args: Any, **kwargs: Any) -> Any:
        """Get total equity from underlying broker."""
        return self._broker.get_equity(*args, **kwargs)

    def get_market_product(self, *args: Any, **kwargs: Any) -> Any:
        """Get market product information from underlying broker."""
        return self._broker.get_market_product(*args, **kwargs)

    def get_market_product_ticker(self, *args: Any, **kwargs: Any) -> Any:
        """Get market product ticker from underlying broker."""
        return self._broker.get_market_product_ticker(*args, **kwargs)

    def get_market_product_candles(self, *args: Any, **kwargs: Any) -> Any:
        """Get market product candles from underlying broker."""
        return self._broker.get_market_product_candles(*args, **kwargs)

    def get_market_products(self, *args: Any, **kwargs: Any) -> Any:
        """Get all market products from underlying broker."""
        return self._broker.get_market_products(*args, **kwargs)

    def start_market_data(self, *args: Any, **kwargs: Any) -> Any:
        """Start market data streaming (read-only operation)."""
        return self._broker.start_market_data(*args, **kwargs)

    def stop_market_data(self, *args: Any, **kwargs: Any) -> Any:
        """Stop market data streaming (read-only operation)."""
        return self._broker.stop_market_data(*args, **kwargs)

    def is_connected(self, *args: Any, **kwargs: Any) -> Any:
        """Check if broker is connected."""
        return self._broker.is_connected(*args, **kwargs)

    def is_stale(self, *args: Any, **kwargs: Any) -> Any:
        """Check if data is stale."""
        return self._broker.is_stale(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Write Operations (Blocked)
    # -------------------------------------------------------------------------

    def place_order(self, *args: Any, **kwargs: Any) -> Any:
        """Block order placement in read-only mode."""
        raise PermissionError(
            "Order execution is disabled in read-only observation mode. "
            "Switch to paper or live mode to place orders."
        )

    def cancel_order(self, *args: Any, **kwargs: Any) -> Any:
        """Block order cancellation in read-only mode."""
        raise PermissionError(
            "Order cancellation is disabled in read-only observation mode. "
            "Switch to paper or live mode to cancel orders."
        )

    def modify_order(self, *args: Any, **kwargs: Any) -> Any:
        """Block order modification in read-only mode."""
        raise PermissionError(
            "Order modification is disabled in read-only observation mode. "
            "Switch to paper or live mode to modify orders."
        )

    def close_position(self, *args: Any, **kwargs: Any) -> Any:
        """Block position closing in read-only mode."""
        raise PermissionError(
            "Position closing is disabled in read-only observation mode. "
            "Switch to paper or live mode to close positions."
        )


__all__ = ["ReadOnlyBroker"]
