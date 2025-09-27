"""
WebSocket transport implementations for Coinbase.

Provides both real and mock transports for production and testing.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, Optional, Any

logger = logging.getLogger(__name__)


class RealTransport:
    """Real WebSocket transport using websocket-client library."""
    
    def __init__(self):
        self.ws = None
        self.url = None
        
    def connect(self, url: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Connect to the WebSocket server with optional headers."""
        try:
            import websocket
        except ImportError:
            raise ImportError(
                "websocket-client is not installed. "
                "Install it with: pip install websocket-client"
            )
        
        self.url = url
        # Pass headers if provided (for auth)
        if headers:
            self.ws = websocket.create_connection(url, header=headers)
        else:
            self.ws = websocket.create_connection(url)
        logger.info(f"Connected to WebSocket: {url}")
    
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
    
    def subscribe(self, message: Dict[str, Any]) -> None:
        """Send a subscription message."""
        if not self.ws:
            raise RuntimeError("Not connected to WebSocket")
        
        msg_str = json.dumps(message)
        self.ws.send(msg_str)
        logger.debug(f"Sent subscription: {msg_str}")
    
    def stream(self) -> Iterable[Dict]:
        """Stream messages from the WebSocket."""
        if not self.ws:
            raise RuntimeError("Not connected to WebSocket")
        
        while True:
            try:
                msg = self.ws.recv()
                if msg:
                    data = json.loads(msg)
                    yield data
            except Exception as e:
                logger.error(f"WebSocket stream error: {e}")
                raise


class MockTransport:
    """Mock WebSocket transport for testing."""
    
    def __init__(self, messages: Optional[list] = None):
        self.messages = messages or []
        self.connected = False
        self.subscriptions = []
        
    def connect(self, url: str, headers: Optional[Dict[str, str]] = None) -> None:
        """Mock connection with optional headers."""
        self.connected = True
        self.headers = headers  # Store for testing
        logger.debug(f"Mock connected to: {url}")
    
    def disconnect(self) -> None:
        """Mock disconnection."""
        self.connected = False
        logger.debug("Mock disconnected")
    
    def subscribe(self, message: Dict[str, Any]) -> None:
        """Record subscription."""
        self.subscriptions.append(message)
        logger.debug(f"Mock subscription: {message}")
    
    def stream(self) -> Iterable[Dict]:
        """Yield predefined messages."""
        for msg in self.messages:
            yield msg
    
    def add_message(self, message: Dict) -> None:
        """Add a message to the mock stream."""
        self.messages.append(message)


class NoopTransport:
    """No-op transport used when streaming is explicitly disabled.

    Provides the same interface as RealTransport/MockTransport but does nothing.
    Useful for environments where websocket-client isn't installed or
    tests want to avoid network dependencies without monkeypatching imports.
    """

    def __init__(self):
        self.connected = False

    def connect(self, url: str, headers: Optional[Dict[str, str]] = None) -> None:  # noqa: ARG002
        self.connected = True
        logger.info("NoopTransport connect called (streaming disabled)")

    def disconnect(self) -> None:
        self.connected = False
        logger.info("NoopTransport disconnect called")

    def subscribe(self, message: Dict[str, Any]) -> None:  # noqa: ARG002
        logger.debug("NoopTransport subscribe called (ignored)")

    def stream(self) -> Iterable[Dict]:
        if False:
            yield {}
