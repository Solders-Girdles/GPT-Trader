"""
WebSocket client for Coinbase streaming (ticker, trades, order book).

Design goals:
- Pluggable transport for unit tests (no network dependency)
- Reconnect with exponential backoff and resubscribe
- Simple subscription model (channels + product_ids)
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from bot_v2.features.monitor import LogLevel, get_logger

from .transports import NoopTransport, RealTransport

logger = logging.getLogger(__name__)


@dataclass
class WSSubscription:
    channels: list[str]
    product_ids: list[str]
    auth_data: dict[str, Any] | None = None  # For authenticated channels


class CoinbaseWebSocket:
    def __init__(
        self,
        url: str,
        max_retries: int = 5,
        base_delay: float = 1.0,
        transport: Any | None = None,
        liveness_timeout: float = 30.0,
        ws_auth_provider: Callable[[], dict[str, Any]] | None = None,
        metrics_emitter: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.url = url
        self.connected = False
        self._sub: WSSubscription | None = None
        self._transport = transport  # Can be injected for testing
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._liveness_timeout = liveness_timeout
        self._ws_auth_provider = ws_auth_provider
        self._last_message_time: float | None = None
        self._sequence_guard = SequenceGuard()
        self._metrics_emitter = metrics_emitter

    def set_transport(self, transport: Any) -> None:
        """Set a custom transport (for testing)."""
        self._transport = transport

    def connect(self, headers: dict[str, str] | None = None) -> None:
        logger.info(f"Connecting WS to {self.url}")
        try:
            get_logger().log_event(level=LogLevel.INFO, event_type="ws_connect", message=f"Connecting to {self.url}")
        except Exception as exc:  # pragma: no cover - telemetry optional
            logger.debug("ws_connect event emit failed", exc_info=exc)

        # Initialize default transport if not set
        if self._transport is None:
            # Allow disabling streaming via env for tests/CI without import hacks
            disable = os.getenv('DISABLE_WS_STREAMING')
            enable_flag = os.getenv('PERPS_ENABLE_STREAMING')
            streaming_disabled = (disable or '').lower() in ('1', 'true', 'yes', 'on') or (enable_flag or '').lower() in ('0', 'false', 'no', 'off')
            if streaming_disabled:
                self._transport = NoopTransport()
                logger.info("Initialized NoopTransport (streaming disabled)")
            else:
                self._transport = RealTransport()
                logger.info("Initialized RealTransport for WebSocket")

        # Pass headers if transport supports it
        if hasattr(self._transport, "connect"):
            if headers and hasattr(self._transport.connect, '__code__') and \
               self._transport.connect.__code__.co_argcount > 2:
                self._transport.connect(self.url, headers)
            else:
                self._transport.connect(self.url)
        self.connected = True
        self._last_message_time = time.time()

    def disconnect(self) -> None:
        logger.info("Disconnecting WS")
        try:
            get_logger().log_event(level=LogLevel.WARNING, event_type="ws_disconnect", message="Disconnecting")
        except Exception as exc:  # pragma: no cover
            logger.debug("ws_disconnect event emit failed", exc_info=exc)
        self.connected = False
        if self._transport and hasattr(self._transport, "disconnect"):
            try:
                self._transport.disconnect()
            except Exception as exc:  # pragma: no cover - transport optional
                logger.debug("transport disconnect failed", exc_info=exc)

    def subscribe(self, sub: WSSubscription) -> None:
        logger.info(f"Subscribing: {sub}")
        self._sub = sub
        if self._transport and hasattr(self._transport, "subscribe"):
            payload = {
                "type": "subscribe",
                "channels": sub.channels,
                "product_ids": sub.product_ids,
            }

            # Add auth data if provided (for user channel)
            if sub.auth_data:
                payload.update(sub.auth_data)
            elif self._ws_auth_provider and "user" in sub.channels:
                # Generate auth data for user channel
                auth_data = self._ws_auth_provider()
                if auth_data:
                    payload.update(auth_data)

            self._transport.subscribe(payload)

    def stream_messages(self) -> Iterable[dict[str, Any]]:
        if not self.connected:
            self.connect()

        if self._transport is None:
            raise RuntimeError(
                "No transport available for WebSocket. "
                "This should not happen after connect(). "
                "Please report this issue."
            )

        attempt = 0
        while True:
            try:
                # Begin streaming with liveness check
                for msg in self._transport.stream():
                    self._last_message_time = time.time()
                    # Annotate with sequence guard for gap detection across all channels
                    try:
                        msg = self._sequence_guard.annotate(msg)
                    except Exception as exc:  # pragma: no cover - defensive guard
                        logger.debug("sequence guard annotation failed", exc_info=exc)
                    # Emit latency metric in debug mode when message has timestamp
                    try:
                        if os.getenv('PERPS_DEBUG') in ('1', 'true', 'yes', 'on'):
                            ts = msg.get('time') or msg.get('timestamp')
                            if isinstance(ts, str):
                                # Normalize Z suffix
                                ts_norm = ts.replace('Z', '+00:00') if ts.endswith('Z') else ts
                                try:
                                    tmsg = datetime.fromisoformat(ts_norm)
                                    latency_ms = (datetime.utcnow() - tmsg.replace(tzinfo=None)).total_seconds() * 1000.0
                                    get_logger().log_ws_latency(stream='coinbase_ws', latency_ms=latency_ms)
                                except Exception as exc_latency:  # pragma: no cover - metrics optional
                                    logger.debug("log_ws_latency failed", exc_info=exc_latency)
                    except Exception as exc_metrics:  # pragma: no cover - metrics optional
                        logger.debug("ws latency metric failed", exc_info=exc_metrics)

                    # Check for liveness timeout
                    if self._liveness_timeout > 0:
                        elapsed = time.time() - (self._last_message_time or time.time())
                        if elapsed > self._liveness_timeout:
                            raise TimeoutError(f"No messages for {elapsed:.1f}s")

                    yield msg
                # If stream ended normally, break
                break
            except Exception as e:
                attempt += 1
                if attempt > self._max_retries:
                    logger.error(f"WS max retries exceeded: {e}")
                    try:
                        get_logger().log_event(level=LogLevel.ERROR, event_type="ws_error", message=f"max retries exceeded: {e}")
                    except Exception as exc_event:  # pragma: no cover - telemetry optional
                        logger.debug("log_event ws_error failed", exc_info=exc_event)
                    break
                # Backoff with jitter-free simple scheme for determinism in tests
                delay = self._base_delay * (2 ** (attempt - 1))
                logger.warning(f"WS error: {e}; reconnecting in {delay:.2f}s (attempt {attempt})")
                try:
                    get_logger().log_event(level=LogLevel.WARNING, event_type="ws_reconnect", message=f"{e}; in {delay:.2f}s (attempt {attempt})")
                except Exception as exc_event:  # pragma: no cover
                    logger.debug("log_event ws_reconnect failed", exc_info=exc_event)
                # Emit reconnect attempt metric if provided
                try:
                    if self._metrics_emitter:
                        self._metrics_emitter({'event_type': 'ws_reconnect_attempt', 'attempt': attempt, 'reason': str(e)})
                except Exception as exc_emit:  # pragma: no cover
                    logger.debug("ws_reconnect_attempt metric failed", exc_info=exc_emit)
                time.sleep(delay)
                # Reconnect
                try:
                    self.disconnect()
                    self.connect()
                    # Reset sequence guard on reconnect
                    self._sequence_guard.reset()
                    # Resubscribe
                    if self._sub:
                        self.subscribe(self._sub)
                    # Emit reconnect success metric
                    try:
                        if self._metrics_emitter:
                            self._metrics_emitter({'event_type': 'ws_reconnect_success', 'attempt': attempt})
                    except Exception as exc_emit:  # pragma: no cover
                        logger.debug("ws_reconnect_success metric failed", exc_info=exc_emit)
                except Exception as e2:
                    logger.error(f"WS reconnect failed: {e2}")
                    continue


class SequenceGuard:
    """Attach simple gap detection based on a monotonically increasing sequence field.

    Looks for any of: 'sequence', 'seq', 'sequence_num' in incoming dicts.
    On detected gap, annotates the message with 'gap_detected': True and 'last_seq'.
    """

    def __init__(self) -> None:
        self.last_seq: int | None = None

    def reset(self) -> None:
        """Reset the sequence guard (e.g., after reconnection)."""
        self.last_seq = None
        logger.debug("SequenceGuard reset")

    def annotate(self, msg: dict[str, Any]) -> dict[str, Any]:
        # Extract sequence
        seq = msg.get("sequence") or msg.get("seq") or msg.get("sequence_num")
        if seq is not None:
            try:
                seq = int(seq)
            except (ValueError, TypeError):
                return msg

            if self.last_seq is not None and seq != self.last_seq + 1:
                # Gap detected
                msg["gap_detected"] = True
                msg["last_seq"] = self.last_seq
                logger.warning(f"WS sequence gap: expected {self.last_seq + 1}, got {seq}")

            self.last_seq = seq
        return msg


def normalize_market_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Normalize market data messages with Decimal prices/sizes.

    Converts price and size fields to Decimal for precision.
    Ensures consistent timestamp field.
    """
    # Convert common price/size fields to Decimal
    for key in ['price', 'size', 'best_bid', 'best_ask', 'bid', 'ask',
                'last', 'volume', 'open', 'high', 'low', 'close']:
        if key in msg and msg[key] is not None:
            try:
                value_str = str(msg[key])
                if value_str and value_str != "":
                    msg[key] = Decimal(value_str)
            except (ValueError, TypeError, InvalidOperation) as exc:
                logger.debug("Failed to normalize %s field", key, exc_info=exc)

    # Ensure timestamp field exists
    if 'timestamp' not in msg and 'time' in msg:
        msg['timestamp'] = msg['time']

    return msg
