"""
WebSocket transport implementations for Coinbase.

Provides both real and mock transports for production and testing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
import sys
from typing import TYPE_CHECKING, Any, cast

from bot_v2.utilities import empty_stream
from bot_v2.utilities.logging_patterns import get_logger

try:  # pragma: no cover - optional dependency shim
    import websocket as websocket  # type: ignore[import-not-found, assignment]
    _HAS_WEBSOCKET = True
except ImportError:  # pragma: no cover - fallback stub for tests
    class _WebsocketStub:
        WebSocketException = Exception

        @staticmethod
        def enableTrace(_flag: bool) -> None:
            raise ModuleNotFoundError(
                "websocket-client is not installed. Install with "
                "`pip install gpt-trader[live-trade]` or `poetry install --with live-trade`."
            )

        @staticmethod
        def create_connection(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError(
                "websocket-client is not installed. Install with "
                "`pip install gpt-trader[live-trade]` or `poetry install --with live-trade`."
            )

    websocket = _WebsocketStub()  # type: ignore[assignment]
    sys.modules.setdefault("websocket", websocket)  # allow unittest.mock.patch
    _HAS_WEBSOCKET = False

if TYPE_CHECKING:
    from bot_v2.orchestration.runtime_settings import RuntimeSettings
else:  # pragma: no cover - runtime type alias
    RuntimeSettings = Any  # type: ignore[misc]

_TRUTHY = {"1", "true", "yes", "on"}

logger = get_logger(__name__, component="coinbase_transport")


def _load_runtime_settings_snapshot() -> RuntimeSettings:
    from bot_v2.orchestration.runtime_settings import load_runtime_settings as _loader

    return _loader()


class RealTransport:
    """Real WebSocket transport using websocket-client library."""

    def __init__(self, *, settings: RuntimeSettings | None = None) -> None:
        self.ws = None
        self.url: str | None = None
        self._static_settings = settings is not None
        self._settings = settings or _load_runtime_settings_snapshot()

    def update_settings(self, settings: RuntimeSettings) -> None:
        """Update runtime settings snapshot used by the transport."""
        self._settings = settings

    def _refresh_settings(self) -> None:
        if not self._static_settings:
            self._settings = _load_runtime_settings_snapshot()

    def connect(self, url: str | dict[str, str] | None = None, headers: dict[str, str] | None = None) -> None:
        """Connect to the WebSocket server with optional headers."""
        self._refresh_settings()

        if isinstance(url, dict) and headers is None:
            headers = url
            url = self.url

        if url is not None:
            self.url = url

        if self.url is None:
            raise ValueError("WebSocket URL is required for connection")

        options: dict[str, Any] = {}
        if headers:
            options["header"] = headers

        timeout_raw = self._settings.raw_env.get("COINBASE_WS_CONNECT_TIMEOUT")
        if timeout_raw:
            try:
                options["timeout"] = float(timeout_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid COINBASE_WS_CONNECT_TIMEOUT=%s (expected float)", timeout_raw
                )

        subprotocols_raw = self._settings.raw_env.get("COINBASE_WS_SUBPROTOCOLS")
        if subprotocols_raw:
            subprotocols = [token.strip() for token in subprotocols_raw.split(",") if token.strip()]
            if subprotocols:
                options["subprotocols"] = subprotocols

        trace_flag = (self._settings.raw_env.get("COINBASE_WS_ENABLE_TRACE") or "").strip().lower()
        if trace_flag in _TRUTHY:
            try:
                websocket.enableTrace(True)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - optional trace feature
                logger.warning("Unable to enable websocket trace output", exc_info=True)

        try:
            self.ws = websocket.create_connection(self.url, **options)
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "websocket-client is not installed. Install the live trading extras with "
                "`pip install gpt-trader[live-trade]` or `poetry install --with live-trade`."
            ) from exc
        logger.info(f"Connected to WebSocket: {self.url}")

    def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self.ws:
            try:
                self.ws.close()
                logger.info("Disconnected from WebSocket")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
            finally:
                self.ws = None

    def subscribe(self, message: dict[str, Any]) -> None:
        """Send a subscription message."""
        if not self.ws:
            raise RuntimeError("Not connected to WebSocket")

        msg_str = json.dumps(message)
        self.ws.send(msg_str)
        logger.debug(f"Sent subscription: {msg_str}")

    def stream(self) -> Iterable[dict[str, Any]]:
        """Stream messages from the WebSocket."""
        if not self.ws:
            raise RuntimeError("Not connected to WebSocket")

        while True:
            try:
                msg = self.ws.recv()
            except StopIteration:
                break
            except Exception as e:
                logger.error("WebSocket stream error: %s", e)
                raise

            try:
                data = json.loads(msg)
            except Exception as exc:  # pragma: no cover - bubble parse errors
                logger.error("WebSocket stream error: %s", exc)
                raise
            yield data


class MockTransport:
    """Mock WebSocket transport for testing."""

    def __init__(self, messages: list[dict[str, Any]] | None = None) -> None:
        self.messages: list[dict[str, Any]] = messages or []
        self.connected = False
        self.subscriptions: list[dict[str, Any]] = []

    def connect(self, url: str, headers: dict[str, str] | None = None) -> None:
        """Mock connection with optional headers."""
        self.connected = True
        self.headers = headers  # Store for testing
        logger.debug(f"Mock connected to: {url}")

    def disconnect(self) -> None:
        """Mock disconnection."""
        self.connected = False
        logger.debug("Mock disconnected")

    def subscribe(self, message: dict[str, Any]) -> None:
        """Record subscription."""
        self.subscriptions.append(message)
        logger.debug(f"Mock subscription: {message}")

    def stream(self) -> Iterable[dict[str, Any]]:
        """Yield predefined messages."""
        yield from self.messages

    def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to the mock stream."""
        self.messages.append(message)


class NoopTransport:
    """No-op transport used when streaming is explicitly disabled.

    Provides the same interface as RealTransport/MockTransport but does nothing.
    Useful for environments where websocket-client isn't installed or
    tests want to avoid network dependencies without monkeypatching imports.
    """

    def __init__(self, *, settings: RuntimeSettings | None = None) -> None:
        self.connected = False
        self._static_settings = settings is not None
        self._settings = settings or _load_runtime_settings_snapshot()

    def connect(self, url: str, headers: dict[str, str] | None = None) -> None:  # noqa: ARG002
        self.connected = True
        logger.info("NoopTransport connect called (streaming disabled)")

    def disconnect(self) -> None:
        self.connected = False
        logger.info("NoopTransport disconnect called")

    def subscribe(self, message: dict[str, Any]) -> None:  # noqa: ARG002
        logger.debug("NoopTransport subscribe called (ignored)")

    def stream(self) -> Iterable[dict[str, Any]]:
        return cast(Iterable[dict[str, Any]], empty_stream())
