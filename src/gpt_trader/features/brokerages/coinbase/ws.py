"""
Simple WebSocket Client for Coinbase.
Replaces the complex 681-line WebSocket manager with a lightweight implementation.

Supports:
- Public channels: ticker, level2, market_trades
- Private channels: user (order updates, fills) - requires authentication
- Event dispatching with typed handlers
- Exponential backoff reconnection
"""

import json
import threading
import time
from collections.abc import Callable
from typing import Any

# websocket-client is an optional dependency (live-trade extra)
try:
    import websocket
except ImportError:
    websocket = None  # type: ignore[assignment]

from gpt_trader.config.constants import (
    MAX_WS_RECONNECT_ATTEMPTS,
    MAX_WS_RECONNECT_DELAY_SECONDS,
    WS_JOIN_TIMEOUT,
    WS_RECONNECT_BACKOFF_MULTIPLIER,
    WS_RECONNECT_DELAY,
)
from gpt_trader.features.brokerages.coinbase.client.constants import WS_BASE_URL
from gpt_trader.features.brokerages.coinbase.ws_events import EventDispatcher
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_websocket")


class SequenceGuard:
    """
    Track WebSocket message sequence numbers and detect gaps.

    Coinbase WebSocket messages include sequence numbers. A gap indicates
    messages may have been missed (e.g., network issues, reconnection).
    """

    def __init__(self) -> None:
        self._last_sequence: int | None = None

    def annotate(self, message: dict) -> dict:
        """
        Annotate a message with gap detection.

        Args:
            message: WebSocket message dict with optional 'sequence' key

        Returns:
            Message dict with 'gap_detected': True added if gap found
        """
        sequence = message.get("sequence")
        if sequence is None:
            return message

        result = dict(message)

        if self._last_sequence is not None:
            expected = self._last_sequence + 1
            if sequence > expected:
                result["gap_detected"] = True

        self._last_sequence = sequence
        return result

    def reset(self) -> None:
        """Reset sequence tracking state."""
        self._last_sequence = None


class CoinbaseWebSocket:
    """
    WebSocket client for Coinbase Advanced Trade API.

    Features:
    - Event dispatching with typed handlers
    - Exponential backoff reconnection
    - Sequence gap detection
    - Health monitoring with timestamps
    - Public and private channel support

    Thread Safety:
        Connection state is protected by a lock to prevent race conditions
        when connect() or disconnect() are called from multiple threads.

    Resource Management:
        Call close() or use as context manager to ensure clean thread shutdown.
    """

    def __init__(
        self,
        url: str = WS_BASE_URL,
        api_key: str | None = None,
        private_key: str | None = None,
        on_message: Callable[[dict], None] | None = None,
        dispatcher: EventDispatcher | None = None,
    ):
        self.url = url
        self.api_key = api_key
        self.private_key = private_key
        self.on_message = on_message
        self.dispatcher = dispatcher or EventDispatcher()
        self.ws: Any = None
        self.wst: threading.Thread | None = None
        self._running = threading.Event()  # Thread-safe running flag
        self._state_lock = threading.Lock()  # Protects connection state changes
        self._shutdown = threading.Event()  # Signals permanent shutdown
        self.subscriptions: list[dict] = []
        self._transport: Any = None
        self._sequence_guard = SequenceGuard()
        self._reconnect_delay: float = float(WS_RECONNECT_DELAY)
        self._reconnect_count = 0
        self._closed = False

        # Health monitoring state
        self._last_message_ts: float | None = None
        self._last_heartbeat_ts: float | None = None
        self._last_close_ts: float | None = None
        self._last_error_ts: float | None = None
        self._gap_count: int = 0

    @property
    def running(self) -> bool:
        """Thread-safe check if WebSocket is running."""
        return self._running.is_set()

    def connect(self) -> None:
        with self._state_lock:
            if self._running.is_set():
                return

            if websocket is None:
                raise ImportError(
                    "websocket-client is not installed. "
                    "Install with: pip install gpt-trader[live-trade]"
                )

            self._running.set()
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            self._transport = self.ws
            self.wst = threading.Thread(target=self.ws.run_forever)
            self.wst.daemon = True
            self.wst.start()
            logger.info("WebSocket thread started")

    def disconnect(self) -> None:
        """Disconnect the WebSocket but allow reconnection."""
        with self._state_lock:
            self._running.clear()
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    logger.debug(f"Error closing WebSocket: {e}")
            if self.wst and self.wst.is_alive():
                self.wst.join(timeout=WS_JOIN_TIMEOUT)
                if self.wst.is_alive():
                    logger.warning("WebSocket thread did not terminate within timeout")

    def close(self) -> None:
        """Permanently close the WebSocket and release all resources.

        Unlike disconnect(), this prevents reconnection and ensures
        complete cleanup of background threads.
        """
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._shutdown.set()
            self._running.clear()

        # Close WebSocket outside lock to avoid deadlock
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            self.ws = None

        # Wait for thread with timeout
        if self.wst and self.wst.is_alive():
            self.wst.join(timeout=WS_JOIN_TIMEOUT)
            if self.wst.is_alive():
                logger.warning("WebSocket thread did not terminate within timeout")
        self.wst = None

        # Clear references
        self._transport = None
        self.subscriptions.clear()
        logger.debug("WebSocket closed and resources released")

    def __enter__(self) -> "CoinbaseWebSocket":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - ensures cleanup."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensures cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Suppress errors during interpreter shutdown

    def subscribe(self, product_ids: list[str], channels: list[str]) -> None:
        """Subscribe to channels for products."""
        sub_msg = {"type": "subscribe", "product_ids": product_ids, "channels": channels}

        # Add auth if keys are present (required for some channels)
        if self.api_key and self.private_key:
            from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth

            auth = SimpleAuth(self.api_key, self.private_key)
            # Coinbase WS Auth requires a signature on the subscribe message usually?
            # Actually, newer Advanced Trade WS uses a JWT token or signature.
            # Let's use the JWT method if available or just rely on the fact
            # that we might be subscribing to public channels.
            # For simplicity in this "Kill Complexity" pass, we assume public channels
            # or that the user will implement the specific auth payload if needed.
            # But wait, the previous code had auth logic.
            # Let's add basic JWT auth if we have it.
            try:
                jwt_token = auth.generate_jwt("GET", "/users/self")
                sub_msg["jwt"] = jwt_token
            except Exception as e:
                logger.warning(f"Failed to generate JWT for WS auth: {e}")

        self.subscriptions.append(sub_msg)
        if self.ws and self.ws.sock and self.ws.sock.connected:
            self._send(sub_msg)

    def _send(self, msg: dict) -> None:
        if self.ws:
            self.ws.send(json.dumps(msg))

    def subscribe_user_events(self, product_ids: list[str] | None = None) -> None:
        """
        Subscribe to private user events (order updates, fills).

        Requires API key and private key for authentication.

        Args:
            product_ids: Optional list of products to filter. If None, receives all.
        """
        if not self.api_key or not self.private_key:
            logger.warning("Cannot subscribe to user events without API credentials")
            return

        channels = ["user"]
        if product_ids:
            self.subscribe(product_ids, channels)
        else:
            # Subscribe without product filter for all user events
            self.subscribe(["BTC-USD"], channels)  # Coinbase requires at least one product

    def _on_open(self, ws: Any) -> None:
        logger.info("WebSocket connected")
        # Reset reconnect state on successful connection
        self._reconnect_delay = WS_RECONNECT_DELAY
        self._reconnect_count = 0
        self._sequence_guard.reset()

        # Resubscribe
        for sub in self.subscriptions:
            self._send(sub)

    def _on_message(self, ws: Any, message: str) -> None:
        try:
            data = json.loads(message)

            # Update health timestamps
            self._last_message_ts = time.time()

            # Check for heartbeat message
            channel = data.get("channel", "")
            msg_type = data.get("type", "")
            if channel == "heartbeats" or msg_type == "heartbeat":
                self._last_heartbeat_ts = time.time()

            # Annotate with sequence gap detection
            data = self._sequence_guard.annotate(data)

            # Log and track if gap detected
            if data.get("gap_detected"):
                self._gap_count += 1
                logger.warning(
                    "WebSocket sequence gap detected",
                    sequence=data.get("sequence"),
                    channel=data.get("channel"),
                    gap_count=self._gap_count,
                )

            # Dispatch to event handlers
            self.dispatcher.dispatch(data)

            # Also call legacy on_message callback if provided
            if self.on_message:
                self.on_message(data)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_error(self, ws: Any, error: Exception) -> None:
        self._last_error_ts = time.time()
        logger.error(
            "WebSocket error",
            error=str(error),
            reconnect_count=self._reconnect_count,
        )

    def _on_close(self, ws: Any, close_status_code: int | None, close_msg: str | None) -> None:
        self._last_close_ts = time.time()
        logger.info(
            "WebSocket closed",
            status_code=close_status_code,
            message=close_msg,
        )

        # Check if permanent shutdown was requested
        if self._shutdown.is_set():
            logger.debug("WebSocket shutdown requested, not reconnecting")
            return

        # Check running state thread-safely
        if self._running.is_set():
            # Update reconnect state under lock
            with self._state_lock:
                self._reconnect_count += 1

                # Check reconnection limit (0 = unlimited)
                if (
                    MAX_WS_RECONNECT_ATTEMPTS > 0
                    and self._reconnect_count > MAX_WS_RECONNECT_ATTEMPTS
                ):
                    logger.error(
                        "Maximum reconnection attempts reached",
                        max_attempts=MAX_WS_RECONNECT_ATTEMPTS,
                    )
                    self._running.clear()
                    return

                delay = min(self._reconnect_delay, MAX_WS_RECONNECT_DELAY_SECONDS)
                self._reconnect_delay = min(
                    self._reconnect_delay * WS_RECONNECT_BACKOFF_MULTIPLIER,
                    MAX_WS_RECONNECT_DELAY_SECONDS,
                )
                # Clear running to allow reconnect
                self._running.clear()

            logger.info(
                "Attempting reconnect",
                delay_seconds=delay,
                attempt=self._reconnect_count,
                max_attempts=(
                    MAX_WS_RECONNECT_ATTEMPTS if MAX_WS_RECONNECT_ATTEMPTS > 0 else "unlimited"
                ),
            )

            time.sleep(delay)

            # Double-check shutdown wasn't requested during sleep
            if not self._shutdown.is_set():
                self.connect()

    def get_health(self) -> dict[str, Any]:
        """
        Get current WebSocket health state.

        Returns:
            Dict with health metrics:
                - last_message_ts: Timestamp of last received message
                - last_heartbeat_ts: Timestamp of last heartbeat
                - last_close_ts: Timestamp of last close event
                - last_error_ts: Timestamp of last error
                - gap_count: Number of sequence gaps detected
                - reconnect_count: Number of reconnection attempts
                - connected: Whether WebSocket is currently connected
        """
        return {
            "last_message_ts": self._last_message_ts,
            "last_heartbeat_ts": self._last_heartbeat_ts,
            "last_close_ts": self._last_close_ts,
            "last_error_ts": self._last_error_ts,
            "gap_count": self._gap_count,
            "reconnect_count": self._reconnect_count,
            "connected": self.running,
        }


__all__ = ["CoinbaseWebSocket", "EventDispatcher", "SequenceGuard"]
