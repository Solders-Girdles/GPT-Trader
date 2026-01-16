"""
WebSocket Client Mixin for streaming market data.

Provides stream_orderbook and stream_trades methods that bridge the
CoinbaseWebSocket class with the broker interface, enabling real-time
data consumption via blocking generators.
"""

import queue
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth

logger = get_logger(__name__, component="websocket_mixin")

# Sentinel value to signal stream termination
_STREAM_STOP = object()

# Default timeout for queue.get() to allow periodic stop checks
_QUEUE_TIMEOUT = 1.0


class WebSocketClientMixin:
    """
    Mixin providing WebSocket streaming capabilities for CoinbaseClient.

    Implements stream_orderbook and stream_trades methods that return
    blocking generators suitable for `for msg in stream:` consumption.

    The mixin maintains a singleton WebSocket connection per client instance.
    Messages are buffered through a queue and yielded to consumers.
    """

    # Type hints for attributes provided by base class
    auth: "SimpleAuth | None"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ws: CoinbaseWebSocket | None = None
        self._ws_lock = threading.Lock()
        self._message_queue: queue.Queue[dict | object] = queue.Queue()
        self._stream_active = False

    def _ensure_ws_state(self) -> None:
        """Ensure required WebSocket attributes exist.

        Some client classes use non-cooperative multiple inheritance and may not
        call this mixin's ``__init__``. This guard keeps streaming safe by
        lazily initialising the expected attributes.
        """
        if not hasattr(self, "_ws"):
            self._ws = None
        if not hasattr(self, "_ws_lock"):
            self._ws_lock = threading.Lock()
        if not hasattr(self, "_message_queue"):
            self._message_queue = queue.Queue()
        if not hasattr(self, "_stream_active"):
            self._stream_active = False

    def _get_websocket(self) -> CoinbaseWebSocket:
        """Get or create the singleton WebSocket instance."""
        self._ensure_ws_state()
        with self._ws_lock:
            if self._ws is None:
                api_key = None
                private_key = None

                # Extract credentials from auth if available
                if self.auth is not None:
                    if hasattr(self.auth, "api_key"):
                        api_key = self.auth.api_key
                    if hasattr(self.auth, "private_key"):
                        private_key = self.auth.private_key

                self._ws = CoinbaseWebSocket(
                    api_key=api_key,
                    private_key=private_key,
                    on_message=self._on_websocket_message,
                )
            return self._ws

    def _on_websocket_message(self, message: dict) -> None:
        """Callback for WebSocket messages - routes to the queue."""
        self._ensure_ws_state()
        if self._stream_active:
            self._message_queue.put(message)

    def _stream_messages(self, stop_event: threading.Event | None = None) -> Iterator[dict]:
        """
        Generator that yields messages from the queue.

        Args:
            stop_event: Optional threading.Event to signal stream termination.

        Yields:
            Message dicts from the WebSocket.
        """
        self._ensure_ws_state()
        while self._stream_active:
            try:
                msg = self._message_queue.get(timeout=_QUEUE_TIMEOUT)
                if msg is _STREAM_STOP:
                    break
                if isinstance(msg, dict):
                    yield msg
            except queue.Empty:
                # Check if we should continue
                if stop_event is not None and stop_event.is_set():
                    break
                continue

    def stream_orderbook(
        self,
        symbols: list[str],
        level: int = 1,
        stop_event: threading.Event | None = None,
        include_trades: bool = False,
        include_user_events: bool = False,
    ) -> Iterator[dict]:
        """
        Stream orderbook updates for the given symbols.

        Args:
            symbols: List of product IDs to subscribe to (e.g., ["BTC-USD", "ETH-USD"]).
            level: Orderbook depth level (1 for top-of-book, 2 for full depth).
            stop_event: Optional threading.Event to signal stream termination.
            include_trades: If True, also subscribe to market_trades channel for
                volume analysis and trade flow data.
            include_user_events: If True, also subscribe to private user events
                (order updates, fills). Requires API credentials.

        Yields:
            Orderbook update messages as dicts. If include_trades=True, also yields
            market trade messages.

        Example:
            >>> for msg in client.stream_orderbook(["BTC-USD"], level=2, include_trades=True):
            ...     print(msg["type"], msg.get("product_id"))
        """
        websocket = self._get_websocket()

        # Clear any stale messages
        self._clear_queue()

        # Determine channel based on level
        # level2 provides full orderbook, ticker provides best bid/ask
        channel = "level2" if level >= 2 else "ticker"

        # Build channel list
        channels = [channel]
        if include_trades:
            channels.append("market_trades")

        logger.info(
            "Starting orderbook stream",
            symbols=symbols,
            level=level,
            channels=channels,
            include_trades=include_trades,
            include_user_events=include_user_events,
        )

        self._stream_active = True

        try:
            # Connect and subscribe
            websocket.connect()
            websocket.subscribe(symbols, channels)
            if include_user_events:
                websocket.subscribe_user_events(symbols)

            # Yield messages
            yield from self._stream_messages(stop_event)

        finally:
            self._stream_active = False
            logger.info("Orderbook stream stopped", symbols=symbols)

    def stream_trades(
        self,
        symbols: list[str],
        stop_event: threading.Event | None = None,
    ) -> Iterator[dict]:
        """
        Stream trade updates for the given symbols.

        Args:
            symbols: List of product IDs to subscribe to (e.g., ["BTC-USD", "ETH-USD"]).
            stop_event: Optional threading.Event to signal stream termination.

        Yields:
            Trade messages as dicts.

        Example:
            >>> for msg in client.stream_trades(["BTC-USD"]):
            ...     print(msg["type"], msg.get("price"))
        """
        self._ensure_ws_state()
        websocket = self._get_websocket()

        # Clear any stale messages
        self._clear_queue()

        logger.info("Starting trades stream", symbols=symbols)

        self._stream_active = True

        try:
            # Connect and subscribe
            websocket.connect()
            websocket.subscribe(symbols, ["market_trades"])

            # Yield messages
            yield from self._stream_messages(stop_event)

        finally:
            self._stream_active = False
            logger.info("Trades stream stopped", symbols=symbols)

    def _clear_queue(self) -> None:
        """Clear any pending messages from the queue."""
        self._ensure_ws_state()
        while True:
            try:
                self._message_queue.get_nowait()
            except queue.Empty:
                break

    def stop_streaming(self) -> None:
        """Stop any active stream and disconnect the WebSocket."""
        self._ensure_ws_state()
        self._stream_active = False
        self._message_queue.put(_STREAM_STOP)

        with self._ws_lock:
            if self._ws is not None:
                try:
                    self._ws.disconnect()
                except Exception as e:
                    logger.warning("Error disconnecting WebSocket", error=str(e))
                self._ws = None

    def is_streaming(self) -> bool:
        """Check if a stream is currently active."""
        self._ensure_ws_state()
        return self._stream_active

    def get_ws_health(self) -> dict:
        """
        Get WebSocket health metrics.

        Returns:
            Dict with health metrics from the underlying WebSocket,
            or empty dict if no WebSocket is active.
        """
        self._ensure_ws_state()
        with self._ws_lock:
            if self._ws is not None:
                return self._ws.get_health()
            return {}


__all__ = ["WebSocketClientMixin"]
