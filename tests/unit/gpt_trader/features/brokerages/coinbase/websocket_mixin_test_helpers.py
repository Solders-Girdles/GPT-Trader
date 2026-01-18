from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.features.brokerages.coinbase.client.websocket_mixin import WebSocketClientMixin


class MockClientBase:
    """Mock base class providing auth attribute."""

    def __init__(self, auth: MagicMock | None = None) -> None:
        self.auth = auth


class MockWebSocketClient(WebSocketClientMixin, MockClientBase):
    """Testable client combining mixin with mock base."""

    def __init__(self, auth: MagicMock | None = None) -> None:
        super().__init__(auth=auth)


class _NonCooperativeBase:
    """Base class that does not call super().__init__()."""

    def __init__(self, auth: MagicMock | None = None) -> None:
        self.auth = auth


class _NonCooperativeClient(_NonCooperativeBase, WebSocketClientMixin):
    """Client where WebSocketClientMixin.__init__ is not invoked."""
