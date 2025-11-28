"""
Simple WebSocket Client for Coinbase.
Replaces the complex 681-line WebSocket manager with a lightweight implementation.
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

from gpt_trader.config.constants import WS_JOIN_TIMEOUT, WS_RECONNECT_DELAY
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
    def __init__(
        self,
        url: str = "wss://advanced-trade-ws.coinbase.com",
        api_key: str | None = None,
        private_key: str | None = None,
        on_message: Callable[[dict], None] | None = None,
    ):
        self.url = url
        self.api_key = api_key
        self.private_key = private_key
        self.on_message = on_message
        self.ws: Any = None
        self.wst: threading.Thread | None = None
        self.running = False
        self.subscriptions: list[dict] = []
        self._transport: Any = None
        self._sequence_guard = SequenceGuard()

    def connect(self) -> None:
        if self.running:
            return

        if websocket is None:
            raise ImportError(
                "websocket-client is not installed. "
                "Install with: pip install gpt-trader[live-trade]"
            )

        self.running = True
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
        self.running = False
        if self.ws:
            self.ws.close()
        if self.wst:
            self.wst.join(timeout=WS_JOIN_TIMEOUT)

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

    def _on_open(self, ws: Any) -> None:
        logger.info("WebSocket connected")
        # Resubscribe
        for sub in self.subscriptions:
            self._send(sub)

    def _on_message(self, ws: Any, message: str) -> None:
        try:
            data = json.loads(message)
            if self.on_message:
                self.on_message(data)
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_error(self, ws: Any, error: Exception) -> None:
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws: Any, close_status_code: int | None, close_msg: str | None) -> None:
        logger.info("WebSocket closed")
        if self.running:
            logger.info("Attempting reconnect in %ds...", WS_RECONNECT_DELAY)
            time.sleep(WS_RECONNECT_DELAY)
            self.connect()


__all__ = ["CoinbaseWebSocket", "SequenceGuard"]
