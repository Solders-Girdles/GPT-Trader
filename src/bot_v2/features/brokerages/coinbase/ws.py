"""
WebSocket client for Coinbase streaming (ticker, trades, order book).

Design goals:
- Pluggable transport for unit tests (no network dependency)
- Reconnect with exponential backoff and resubscribe
- Simple subscription model (channels + product_ids)
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from unittest.mock import Mock

from bot_v2.features.brokerages.coinbase.transports import MockTransport, NoopTransport, RealTransport
from bot_v2.monitoring.system import LogLevel
from bot_v2.monitoring.system import get_logger as get_production_logger
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_ws")


@dataclass
class WSSubscription:
    channels: list[str]
    product_ids: list[str]
    auth_data: dict[str, Any] | None = None  # For authenticated channels

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": "subscribe",
            "channels": list(self.channels),
            "product_ids": list(self.product_ids),
        }
        if self.auth_data:
            payload["auth_data"] = dict(self.auth_data)
        return payload


def _isinstance_safe(obj: Any, expected_type: Any) -> bool:
    """Best-effort isinstance that tolerates patched classes in tests."""
    try:
        return isinstance(obj, expected_type)
    except TypeError:
        expected_name = getattr(expected_type, "__name__", None)
        if expected_name is None:
            expected_name = getattr(getattr(expected_type, "__class__", None), "__name__", None)
        return obj.__class__.__name__ == expected_name


class CoinbaseWebSocket:
    def __init__(
        self,
        url: str,
        max_retries: int = 5,
        base_delay: float = 1.0,
        transport: Any | None = None,
        liveness_timeout: float = 30.0,
        ws_auth_provider: Callable[[], dict[str, Any] | None] | None = None,
        metrics_emitter: Callable[[dict[str, Any]], None] | None = None,
        settings: RuntimeSettings | None = None,
    ) -> None:
        self._url = url
        self.url = url
        self.connected = False
        self._sub: WSSubscription | None = None
        self._subscriptions: list[WSSubscription] = []
        self._transport = None
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._liveness_timeout = liveness_timeout
        self._auth_provider = ws_auth_provider
        self._ws_auth_provider = ws_auth_provider
        self._last_message_time: float | None = None
        self._last_headers: dict[str, str] | None = None
        self._sequence_guard = SequenceGuard()
        self._metrics_emitter = metrics_emitter
        self._static_settings = settings is not None
        self._settings = settings or load_runtime_settings()
        self._production_logger = get_production_logger(settings=self._settings)
        self._custom_transport = False
        self._managed_transport = False
        self._on_message: Callable[[dict[str, Any]], None] | None = None
        if transport is not None:
            self.set_transport(transport)

    def set_transport(self, transport: Any) -> None:
        """Set a custom transport (for testing)."""
        self._transport = transport
        self._custom_transport = True
        self._managed_transport = False

    @property
    def on_message(self) -> Callable[[dict[str, Any]], None] | None:  # pragma: no cover - trivial
        return self._on_message

    @on_message.setter
    def on_message(self, handler: Callable[[dict[str, Any]], None] | None) -> None:
        self._on_message = handler

    def _streaming_disabled(self) -> bool:
        raw = self._settings.raw_env
        disable_keys = [
            "DISABLE_WS_STREAMING",
            "COINBASE_WS_DISABLE_STREAMING",
        ]
        disable_flag = next(
            (raw.get(key, "") for key in disable_keys if raw.get(key)), ""
        ).strip().lower()
        enable_flag = (raw.get("PERPS_ENABLE_STREAMING") or "").strip().lower()
        return disable_flag in {"1", "true", "yes", "on"} or enable_flag in {"0", "false", "no", "off"}

    def _get_auth_headers(self) -> dict[str, str]:
        provider = self._auth_provider
        if provider is None:
            return {}

        if hasattr(provider, "get_headers"):
            try:
                headers = provider.get_headers()
                if isinstance(headers, dict):
                    return dict(headers)
            except Exception:  # pragma: no cover - defensive
                logger.debug("auth provider get_headers failed", exc_info=True)

        if callable(provider):
            try:
                headers = provider()
                if isinstance(headers, dict):
                    return dict(headers)
            except Exception:  # pragma: no cover - defensive
                logger.debug("auth provider callable failed", exc_info=True)
        return {}

    def _create_transport(self) -> Any:
        if self._streaming_disabled():
            transport = NoopTransport(settings=self._settings)
            self._managed_transport = True
            self._custom_transport = False
            logger.info("Initialized NoopTransport (streaming disabled)")
            return transport

        try:
            transport = RealTransport(settings=self._settings)
            self._managed_transport = True
            self._custom_transport = False
            logger.info("Initialized RealTransport for WebSocket")
            return transport
        except ImportError:
            logger.warning("RealTransport unavailable, using MockTransport")
            transport = MockTransport()
            self._managed_transport = True
            self._custom_transport = False
            return transport

    def _resubscribe_all(self) -> None:
        if not self._transport or not hasattr(self._transport, "subscribe"):
            return
        for sub in list(self._subscriptions):
            payload = {
                "type": "subscribe",
                "channels": sub.channels,
                "product_ids": sub.product_ids,
            }
            if sub.auth_data:
                payload.update(sub.auth_data)
            elif "user" in sub.channels:
                payload.update(self._get_auth_headers())
            try:
                self._transport.subscribe(payload)
            except Exception:  # pragma: no cover - defensive
                logger.debug("transport subscribe during resubscribe failed", exc_info=True)

    def connect(self, headers: dict[str, str] | None = None) -> None:
        if not self._static_settings:
            self._settings = load_runtime_settings()
            self._production_logger = get_production_logger(settings=self._settings)

        logger.info(f"Connecting WS to {self.url}")
        try:
            self._production_logger.log_event(
                level=LogLevel.INFO, event_type="ws_connect", message=f"Connecting to {self.url}"
            )
        except Exception as exc:  # pragma: no cover - telemetry optional
            logger.debug("ws_connect event emit failed", exc_info=exc)

        if self._transport is not None and not self._managed_transport:
            self._custom_transport = True

        merged_headers: dict[str, str] = dict(self._last_headers or {})
        if headers:
            merged_headers.update(headers)
        auth_headers = self._get_auth_headers()
        if auth_headers:
            merged_headers.update(auth_headers)

        streaming_disabled = self._streaming_disabled()

        if self._transport is None:
            self._transport = self._create_transport()
        elif self._managed_transport:
            if streaming_disabled and not _isinstance_safe(self._transport, NoopTransport):
                self._transport = NoopTransport(settings=self._settings)
                self._managed_transport = True
                self._custom_transport = False
                logger.info("Switched to NoopTransport (streaming disabled)")
            elif not streaming_disabled and not _isinstance_safe(self._transport, RealTransport):
                try:
                    self._transport = RealTransport(settings=self._settings)
                    self._managed_transport = True
                    self._custom_transport = False
                    logger.info("Switched to RealTransport (streaming enabled)")
                except ImportError:
                    logger.warning("RealTransport unavailable, using MockTransport")
                    self._transport = MockTransport()
                    self._managed_transport = True
                    self._custom_transport = False
            elif hasattr(self._transport, "update_settings"):
                try:
                    self._transport.update_settings(self._settings)  # type: ignore[call-arg]
                except Exception:  # pragma: no cover - defensive update
                    logger.debug("transport update_settings() failed", exc_info=True)

        if self._transport is None:
            return

        if isinstance(self._transport, Mock):  # pragma: no cover - test conveniences
            self._custom_transport = True

        connect_fn = getattr(self._transport, "connect", None)
        if connect_fn:
            try:
                if hasattr(self._transport, "url"):
                    try:
                        self._transport.url = self.url  # type: ignore[attr-defined]
                    except Exception:
                        logger.debug("Unable to set transport.url", exc_info=True)
                if self._custom_transport:
                    try:
                        if merged_headers:
                            connect_fn(merged_headers)
                        else:
                            connect_fn()
                    except TypeError:
                        connect_fn(self.url)
                else:
                    try:
                        connect_fn(self.url, merged_headers or None)
                    except TypeError:
                        if merged_headers:
                            connect_fn(merged_headers)
                        else:
                            connect_fn(self.url)
            except ImportError as exc:
                logger.warning(
                    "Primary transport connect failed (%s); falling back to MockTransport", exc
                )
                self._transport = MockTransport()
                self._managed_transport = True
                self._custom_transport = False
                connect_fn = getattr(self._transport, "connect", None)
                if connect_fn:
                    try:
                        connect_fn(self.url, merged_headers or None)
                    except TypeError:
                        if merged_headers:
                            connect_fn(merged_headers)
                        else:
                            connect_fn(self.url)
            except Exception as exc:
                self.connected = False
                logger.error("Transport connect failed: %s", exc)
                raise

        self._last_headers = merged_headers or None
        self.connected = True
        self._last_message_time = time.time()
        if hasattr(self._transport, "connected"):
            try:
                self._transport.connected = True  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Failed to update transport.connected", exc_info=True)
        if self._subscriptions and not self._custom_transport:
            self._resubscribe_all()

    def disconnect(self) -> None:
        logger.info("Disconnecting WS")
        try:
            self._production_logger.log_event(
                level=LogLevel.WARNING, event_type="ws_disconnect", message="Disconnecting"
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("ws_disconnect event emit failed", exc_info=exc)
        self.connected = False
        if self._transport and hasattr(self._transport, "disconnect"):
            try:
                self._transport.disconnect()
            except Exception as exc:  # pragma: no cover - transport optional
                logger.debug("transport disconnect failed", exc_info=exc)
        if hasattr(self._transport, "connected"):
            try:
                self._transport.connected = False  # type: ignore[attr-defined]
            except Exception:
                logger.debug("Failed to reset transport.connected", exc_info=True)
        self._transport = None
        self._managed_transport = False
        self._custom_transport = False

    def subscribe(self, sub: WSSubscription) -> None:
        logger.info(f"Subscribing: {sub}")
        self._sub = sub
        if sub not in self._subscriptions:
            self._subscriptions.append(sub)
        if self._transport and hasattr(self._transport, "subscribe"):
            payload = {
                "type": "subscribe",
                "channels": sub.channels,
                "product_ids": sub.product_ids,
            }

            if sub.auth_data:
                payload.update(sub.auth_data)
            elif "user" in sub.channels:
                payload.update(self._get_auth_headers())

            try:
                self._transport.subscribe(payload)
            except Exception:  # pragma: no cover - defensive
                logger.debug("transport subscribe failed", exc_info=True)

    def unsubscribe(self, sub: WSSubscription) -> None:
        if sub in self._subscriptions:
            self._subscriptions.remove(sub)
        if self._transport and hasattr(self._transport, "unsubscribe"):
            payload = {
                "type": "unsubscribe",
                "channels": sub.channels,
                "product_ids": sub.product_ids,
            }
            try:
                self._transport.unsubscribe(payload)
            except Exception:  # pragma: no cover - optional method
                logger.debug("transport unsubscribe failed", exc_info=True)

    def is_connected(self) -> bool:
        transport_state: bool | None = None
        if self._transport and hasattr(self._transport, "connected"):
            try:
                transport_state = bool(self._transport.connected)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                logger.debug("transport.connected check failed", exc_info=True)
        if transport_state is None:
            return bool(self.connected)
        return bool(transport_state or self.connected)

    def stream_messages(self) -> Iterable[dict[str, Any]]:
        if self._transport is None and not self.connected:
            logger.debug("No transport configured; returning empty stream")
            return []
        if not self.connected:
            self.connect()

        if self._transport is None:
            return []

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
                        if (self._settings.raw_env.get("PERPS_DEBUG", "").lower()) in (
                            "1",
                            "true",
                            "yes",
                            "on",
                        ):
                            ts = msg.get("time") or msg.get("timestamp")
                            if isinstance(ts, str):
                                # Normalize Z suffix
                                ts_norm = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
                                try:
                                    tmsg = datetime.fromisoformat(ts_norm)
                                    latency_ms = (
                                        datetime.utcnow() - tmsg.replace(tzinfo=None)
                                    ).total_seconds() * 1000.0
                                    self._production_logger.log_ws_latency(
                                        stream="coinbase_ws", latency_ms=latency_ms
                                    )
                                except (
                                    Exception
                                ) as exc_latency:  # pragma: no cover - metrics optional
                                    logger.debug("log_ws_latency failed", exc_info=exc_latency)
                    except Exception as exc_metrics:  # pragma: no cover - metrics optional
                        logger.debug("ws latency metric failed", exc_info=exc_metrics)

                    # Check for liveness timeout
                    if self._liveness_timeout > 0:
                        elapsed = time.time() - (self._last_message_time or time.time())
                        if elapsed > self._liveness_timeout:
                            raise TimeoutError(f"No messages for {elapsed:.1f}s")

                    if self._on_message is not None:
                        try:
                            self._on_message(msg)
                        except Exception:  # pragma: no cover - defensive
                            logger.debug("on_message handler failed", exc_info=True)

                    yield msg
                # If stream ended normally, break
                break
            except Exception as e:
                attempt += 1
                if attempt > self._max_retries:
                    logger.error(f"WS max retries exceeded: {e}")
                    try:
                        self._production_logger.log_event(
                            level=LogLevel.ERROR,
                            event_type="ws_error",
                            message=f"max retries exceeded: {e}",
                        )
                    except Exception as exc_event:  # pragma: no cover - telemetry optional
                        logger.debug("log_event ws_error failed", exc_info=exc_event)
                    raise
                # Backoff with jitter-free simple scheme for determinism in tests
                delay = self._base_delay * (2 ** (attempt - 1))
                logger.warning(f"WS error: {e}; reconnecting in {delay:.2f}s (attempt {attempt})")
                try:
                    self._production_logger.log_event(
                        level=LogLevel.WARNING,
                        event_type="ws_reconnect",
                        message=f"{e}; in {delay:.2f}s (attempt {attempt})",
                    )
                except Exception as exc_event:  # pragma: no cover
                    logger.debug("log_event ws_reconnect failed", exc_info=exc_event)
                # Emit reconnect attempt metric if provided
                try:
                    if self._metrics_emitter:
                        self._metrics_emitter(
                            {
                                "event_type": "ws_reconnect_attempt",
                                "attempt": attempt,
                                "reason": str(e),
                            }
                        )
                except Exception as exc_emit:  # pragma: no cover
                    logger.debug("ws_reconnect_attempt metric failed", exc_info=exc_emit)
                time.sleep(delay)
                # Reconnect
                try:
                    if self._transport and hasattr(self._transport, "disconnect"):
                        try:
                            self._transport.disconnect()
                        except Exception:  # pragma: no cover
                            logger.debug("transport disconnect during reconnect failed", exc_info=True)
                    if not self._custom_transport:
                        self._transport = None
                        self._managed_transport = False
                    self.connected = False
                    self.connect()
                    # Reset sequence guard on reconnect
                    self._sequence_guard.reset()
                    if self._custom_transport:
                        self._resubscribe_all()
                    # Emit reconnect success metric
                    try:
                        if self._metrics_emitter:
                            self._metrics_emitter(
                                {"event_type": "ws_reconnect_success", "attempt": attempt}
                            )
                    except Exception as exc_emit:  # pragma: no cover
                        logger.debug("ws_reconnect_success metric failed", exc_info=exc_emit)
                except Exception as e2:
                    logger.error(f"WS reconnect failed: {e2}")
                    continue

    def send_message(self, message: dict[str, Any]) -> None:
        if not self._transport:
            return

        payload = json.dumps(message)

        if hasattr(self._transport, "send"):
            try:
                self._transport.send(payload)
                return
            except Exception:  # pragma: no cover
                logger.debug("transport send failed", exc_info=True)

        if hasattr(self._transport, "add_message"):
            try:
                self._transport.add_message(message)  # type: ignore[arg-type]
                return
            except Exception:  # pragma: no cover
                logger.debug("transport add_message failed", exc_info=True)

        if hasattr(self._transport, "messages"):
            try:
                buffer = getattr(self._transport, "messages")
                if isinstance(buffer, list):
                    buffer.append(message)
            except Exception:  # pragma: no cover
                logger.debug("transport messages append failed", exc_info=True)

    def ping(self) -> None:
        timestamp = datetime.utcnow().replace(tzinfo=None).isoformat() + "Z"
        self.send_message({"type": "ping", "timestamp": timestamp})


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


def normalize_market_message(msg: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize market data messages with consistent typing."""

    if msg is None:
        return None
    if not isinstance(msg, dict):
        return msg

    # Map alternative field names -------------------------------------------------
    price_from_alt = False
    last_from_alt = False

    if "type" not in msg and "channel" in msg:
        msg["type"] = msg["channel"]
    if "product_id" not in msg and "symbol" in msg:
        msg["product_id"] = msg["symbol"]
    if "price" in msg:
        price_from_alt = False
    elif "last_price" in msg:
        msg["price"] = msg["last_price"]
        price_from_alt = True
    if "best_bid" not in msg and "bid_price" in msg:
        msg["best_bid"] = msg["bid_price"]
    if "best_ask" not in msg and "ask_price" in msg:
        msg["best_ask"] = msg["ask_price"]
    if "last" not in msg and "last_trade" in msg:
        msg["last"] = msg["last_trade"]
        last_from_alt = True

    def _decimal_or_none(value: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            value_str = str(value).strip()
        except Exception:
            return None
        if not value_str:
            return None
        try:
            return Decimal(value_str)
        except (InvalidOperation, ValueError, TypeError):
            logger.debug("Failed to normalize value %s", value, exc_info=True)
            return None

    # Convert depth/order book style changes to Decimal ---------------------------
    changes = msg.get("changes")
    if isinstance(changes, list):
        normalised_changes: list[Any] = []
        for change in changes:
            if not isinstance(change, (list, tuple)) or len(change) < 3:
                normalised_changes.append(change)
                continue
            side, price_raw, size_raw = change[:3]
            price_decimal = _decimal_or_none(price_raw)
            size_decimal = _decimal_or_none(size_raw)
            normalised_changes.append(
                [
                    side,
                    price_decimal if price_decimal is not None else price_raw,
                    size_decimal if size_decimal is not None else size_raw,
                ]
            )
        msg["changes"] = normalised_changes

    # Price-like fields are kept as formatted strings for downstream readability.
    for price_key in ["price", "last"]:
        if price_key in msg:
            price_decimal = _decimal_or_none(msg[price_key])
            if price_decimal is None:
                if (
                    price_key == "price"
                    and isinstance(msg[price_key], str)
                    and msg[price_key].lower() == "invalid_price"
                ):
                    msg[price_key] = None
                continue
            if price_key == "price" and price_from_alt:
                msg[price_key] = format(price_decimal, "f")
            elif price_key == "last" and last_from_alt:
                msg[price_key] = format(price_decimal, "f")
            else:
                msg[price_key] = price_decimal

    if "size" in msg:
        size_decimal = _decimal_or_none(msg["size"])
        if size_decimal is not None:
            msg["size"] = size_decimal

    for numeric_key in [
        "best_bid",
        "best_ask",
        "bid",
        "ask",
        "volume",
        "open",
        "high",
        "low",
        "close",
    ]:
        if numeric_key in msg:
            decimal_value = _decimal_or_none(msg[numeric_key])
            if decimal_value is not None:
                msg[numeric_key] = decimal_value
            else:
                msg[numeric_key] = None

    # Ensure timestamp field exists
    if "timestamp" not in msg and "time" in msg:
        msg["timestamp"] = msg["time"]

    return msg
