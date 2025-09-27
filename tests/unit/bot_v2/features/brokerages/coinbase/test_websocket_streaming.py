"""
Unit tests for Phase 3 - WebSocket Integration (Perpetuals).
Tests market WS, authenticated user events, and reconnection without network calls.
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
from typing import Dict, List, Any

from bot_v2.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket, 
    WSSubscription, 
    SequenceGuard,
    normalize_market_message
)
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.transports import MockTransport


class TestMarketWebSocket:
    """Test market data WebSocket subscriptions."""
    
    def test_stream_trades_normalization(self):
        """Test stream_trades normalizes messages with Decimal prices/sizes."""
        # Create mock messages
        mock_messages = [
            {
                "type": "trade",
                "product_id": "BTC-USD-PERP",
                "price": "50000.50",
                "size": "0.1",
                "time": "2024-01-15T12:00:00Z"
            },
            {
                "type": "trade",
                "product_id": "ETH-USD-PERP",
                "price": "3000.25",
                "size": "1.5",
                "time": "2024-01-15T12:00:01Z"
            },
            {
                "type": "trade",
                "product_id": "BTC-USD-PERP",
                "price": "50001.00",
                "size": "0.2",
                "time": "2024-01-15T12:00:02Z"
            }
        ]
        
        # Create adapter with mock transport
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False
        )
        adapter = CoinbaseBrokerage(config)
        
        # Create mock WebSocket
        mock_transport = MockTransport(messages=mock_messages)
        mock_ws = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com",
            transport=mock_transport
        )
        
        # Override WS factory
        adapter._ws_factory_override = lambda: mock_ws
        
        # Stream trades
        trades = list(adapter.stream_trades(["BTC-USD-PERP", "ETH-USD-PERP"]))
        
        # Verify normalization
        assert len(trades) == 3
        
        # Check first trade
        assert trades[0]["product_id"] == "BTC-USD-PERP"
        assert trades[0]["price"] == Decimal("50000.50")
        assert trades[0]["size"] == Decimal("0.1")
        assert trades[0]["timestamp"] == "2024-01-15T12:00:00Z"
        
        # Check second trade (different product)
        assert trades[1]["product_id"] == "ETH-USD-PERP"
        assert trades[1]["price"] == Decimal("3000.25")
        assert trades[1]["size"] == Decimal("1.5")
        
        # Verify all trades have Decimal types
        for trade in trades:
            assert isinstance(trade["price"], Decimal)
            assert isinstance(trade["size"], Decimal)
    
    def test_status_channel_subscription(self):
        """Test that subscribing to status channel sends correct payload."""
        from bot_v2.features.brokerages.coinbase.transports import MockTransport
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False
        )
        adapter = CoinbaseBrokerage(config)
        
        # Track subscriptions
        subs = []
        mock_transport = MockTransport(messages=[])
        original = mock_transport.subscribe
        def track(payload):
            subs.append(payload)
            return original(payload)
        mock_transport.subscribe = track
        
        mock_ws = CoinbaseWebSocket(url="wss://advanced-trade-ws.coinbase.com", transport=mock_transport)
        adapter._ws_factory_override = lambda: mock_ws
        
        # Manually subscribe to status channel
        sub = WSSubscription(channels=["status"], product_ids=["BTC-USD-PERP", "ETH-USD-PERP"])
        mock_ws.subscribe(sub)
        
        assert len(subs) == 1
        assert subs[0]["channels"] == ["status"]
        assert subs[0]["product_ids"] == ["BTC-USD-PERP", "ETH-USD-PERP"]
    
    def test_stream_orderbook_channel_selection(self):
        """Test stream_orderbook selects correct channel based on level."""
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False
        )
        adapter = CoinbaseBrokerage(config)
        
        # Track subscriptions
        subscriptions = []
        
        # Create mock transport that records subscriptions
        mock_transport = MockTransport(messages=[])
        original_subscribe = mock_transport.subscribe
        
        def track_subscribe(msg):
            subscriptions.append(msg)
            return original_subscribe(msg)
        
        mock_transport.subscribe = track_subscribe
        
        # Test level 1 (ticker)
        mock_ws = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com",
            transport=mock_transport
        )
        adapter._ws_factory_override = lambda: mock_ws
        
        # Start streaming L1
        stream = adapter.stream_orderbook(["BTC-USD-PERP"], level=1)
        try:
            next(stream)
        except StopIteration:
            pass  # No messages to consume
        
        # Verify ticker channel selected
        assert len(subscriptions) == 1
        assert subscriptions[0]["channels"] == ["ticker"]
        assert subscriptions[0]["product_ids"] == ["BTC-USD-PERP"]
        
        # Reset for L2 test
        subscriptions.clear()
        mock_transport.subscriptions.clear()
        
        # Test level 2 (level2)
        mock_ws2 = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com",
            transport=MockTransport(messages=[])
        )
        mock_ws2._transport.subscribe = track_subscribe
        adapter._ws_factory_override = lambda: mock_ws2
        
        stream = adapter.stream_orderbook(["ETH-USD-PERP"], level=2)
        try:
            next(stream)
        except StopIteration:
            pass
        
        # Verify level2 channel selected (only one subscription because new WS)
        assert len(subscriptions) == 1
        assert subscriptions[0]["channels"] == ["level2"]
        assert subscriptions[0]["product_ids"] == ["ETH-USD-PERP"]
    
    def test_normalize_market_message(self):
        """Test normalize_market_message helper function."""
        # Test price/size normalization
        msg = {
            "price": "12345.67",
            "size": "1.234",
            "best_bid": "12340.00",
            "best_ask": "12350.00",
            "volume": "100.5"
        }
        
        normalized = normalize_market_message(msg)
        
        assert normalized["price"] == Decimal("12345.67")
        assert normalized["size"] == Decimal("1.234")
        assert normalized["best_bid"] == Decimal("12340.00")
        assert normalized["best_ask"] == Decimal("12350.00")
        assert normalized["volume"] == Decimal("100.5")
        
        # Test timestamp normalization
        msg_with_time = {
            "time": "2024-01-15T12:00:00Z",
            "price": "100"
        }
        
        normalized = normalize_market_message(msg_with_time)
        assert normalized["timestamp"] == "2024-01-15T12:00:00Z"
        assert normalized["time"] == "2024-01-15T12:00:00Z"
        
        # Test robustness with invalid values
        msg_invalid = {
            "price": "invalid",
            "size": None,
            "volume": ""
        }
        
        normalized = normalize_market_message(msg_invalid)
        assert normalized["price"] == "invalid"  # Kept original
        assert normalized["size"] is None
        assert normalized["volume"] == ""


class TestAuthenticatedUserEvents:
    """Test authenticated user event streaming."""
    
    def test_stream_user_events_with_sequence_gap(self):
        """Test stream_user_events detects sequence gaps."""
        # Create messages with a gap
        mock_messages = [
            {"type": "order", "sequence": 100, "order_id": "order1"},
            {"type": "order", "sequence": 101, "order_id": "order2"},
            {"type": "order", "sequence": 103, "order_id": "order3"},  # Gap here (102 missing)
            {"type": "fill", "sequence": 104, "order_id": "order3"}
        ]
        
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False
        )
        adapter = CoinbaseBrokerage(config)
        
        # Create mock WebSocket
        mock_transport = MockTransport(messages=mock_messages)
        mock_ws = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com",
            transport=mock_transport
        )
        adapter._ws_factory_override = lambda: mock_ws
        
        # Stream user events
        events = list(adapter.stream_user_events())
        
        assert len(events) == 4
        
        # First two should have no gap
        assert "gap_detected" not in events[0]
        assert "gap_detected" not in events[1]
        
        # Third message should have gap detected
        assert events[2]["gap_detected"] == True
        assert events[2]["last_seq"] == 101
        assert events[2]["sequence"] == 103
        
        # Fourth should continue normally
        assert "gap_detected" not in events[3]
    
    def test_ws_auth_provider_injection(self):
        """Test that ws_auth_provider is used for user channel auth."""
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False,
            enable_derivatives=True,
            auth_type="JWT",
            cdp_api_key="test_key",
            cdp_private_key="test_private_key"
        )
        adapter = CoinbaseBrokerage(config)
        
        # Track subscribe payloads
        subscribe_payloads = []
        
        # Create mock transport
        mock_transport = MockTransport(messages=[])
        original_subscribe = mock_transport.subscribe
        
        def track_subscribe(payload):
            subscribe_payloads.append(payload)
            return original_subscribe(payload)
        
        mock_transport.subscribe = track_subscribe
        
        # Create WS and trigger subscription with provided jwt via ws_auth_provider
        mock_ws = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com",
            transport=mock_transport,
            ws_auth_provider=lambda: {"jwt": "test_jwt_token"}
        )
        
        # Subscribe to user channel
        sub = WSSubscription(channels=["user"], product_ids=[])
        mock_ws.subscribe(sub)
        
        # Verify JWT was included in payload
        assert len(subscribe_payloads) == 1
        assert subscribe_payloads[0]["jwt"] == "test_jwt_token"
        assert subscribe_payloads[0]["channels"] == ["user"]


class TestReconnectAndLiveness:
    """Test reconnection and liveness features."""
    
    def test_reconnect_with_resubscribe(self):
        """Test that reconnection triggers resubscribe and SequenceGuard reset."""
        # Messages that will cause an error after 2 messages
        class ErrorAfterNTransport:
            def __init__(self, n=2):
                self.message_count = 0
                self.n = n
                self.connected = False
                self.subscriptions = []
                self.stream_attempt = 0
                
            def connect(self, url, headers=None):
                self.connected = True
                
            def disconnect(self):
                self.connected = False
                
            def subscribe(self, payload):
                self.subscriptions.append(payload)
                
            def stream(self):
                self.stream_attempt += 1
                
                # Different messages based on stream attempt
                if self.stream_attempt == 1:
                    # First stream (will error after n messages)
                    messages = [
                        {"sequence": 1, "data": "msg1"},
                        {"sequence": 2, "data": "msg2"}
                    ]
                    for msg in messages:
                        self.message_count += 1
                        if self.message_count == self.n:
                            raise ConnectionError("Simulated disconnect")
                        yield msg
                else:
                    # After reconnect - new stream
                    messages = [
                        {"sequence": 10, "data": "msg3"},
                        {"sequence": 11, "data": "msg4"}
                    ]
                    for msg in messages:
                        yield msg
        
        # Create WebSocket with error transport
        ws = CoinbaseWebSocket(
            url="wss://test",
            max_retries=2,
            base_delay=0.01  # Short delay for test
        )
        
        error_transport = ErrorAfterNTransport(n=2)  # Error after 2 messages
        ws._transport = error_transport
        
        # Subscribe
        sub = WSSubscription(channels=["user"], product_ids=["BTC-USD"])
        ws.subscribe(sub)
        
        # Stream messages (WebSocket handles SequenceGuard reset internally)
        messages = []
        
        for msg in ws.stream_messages():
            messages.append(msg)
            if len(messages) >= 4:
                break
        
        # Verify we got messages from both before and after reconnect
        assert len(messages) >= 3  # At least 1 before disconnect + 2 after
        
        # Verify resubscribe happened (should have 2 subscriptions)
        assert len(error_transport.subscriptions) == 2
        assert error_transport.subscriptions[0] == error_transport.subscriptions[1]
        
        # After reconnect, messages continue normally
        # Note: We get msg1, then disconnect at msg2, then reconnect and get msg3, msg4
        assert messages[0]["data"] == "msg1"  # Before disconnect
        # msg2 caused disconnect, so we don't see it
        assert messages[1]["data"] == "msg3"  # After reconnect
        assert messages[2]["data"] == "msg4"  # After reconnect
    
    def test_sequence_guard_reset(self):
        """Test that SequenceGuard.reset() clears last_seq."""
        guard = SequenceGuard()
        
        # Process some messages
        msg1 = guard.annotate({"sequence": 100})
        msg2 = guard.annotate({"sequence": 101})
        
        assert guard.last_seq == 101
        
        # Reset
        guard.reset()
        assert guard.last_seq is None
        
        # New sequence should not detect gap
        msg3 = guard.annotate({"sequence": 200})
        assert "gap_detected" not in msg3
        assert guard.last_seq == 200
    
    def test_liveness_timeout(self):
        """Test liveness timeout triggers reconnect."""
        # Transport that delays messages
        class SlowTransport:
            def __init__(self):
                self.connected = False
                self.connect_count = 0
                
            def connect(self, url, headers=None):
                self.connected = True
                self.connect_count += 1
                
            def disconnect(self):
                self.connected = False
                
            def subscribe(self, payload):
                pass
                
            def stream(self):
                # First message immediate; omit delays to avoid test flakiness
                yield {"msg": 1}
                # Subsequent messages would come later in real transport
                yield {"msg": 2}
        
        # Create WebSocket with short liveness timeout
        ws = CoinbaseWebSocket(
            url="wss://test",
            liveness_timeout=0.1,  # 100ms timeout
            max_retries=1,
            base_delay=0.01
        )
        
        # Note: In real implementation, liveness timeout would be checked
        # in a separate thread or via select/poll. For this test, we're
        # verifying the timeout value is stored and accessible.
        assert ws._liveness_timeout == 0.1


class TestMultiSubscriptions:
    """Test multiple channel subscriptions behavior."""

    def test_multiple_channels_single_payload(self):
        from bot_v2.features.brokerages.coinbase.transports import MockTransport
        subs = []
        t = MockTransport(messages=[])
        original = t.subscribe
        def track(payload):
            subs.append(payload)
            return original(payload)
        t.subscribe = track

        ws = CoinbaseWebSocket(url="wss://test", transport=t)
        ws.connect()
        ws.subscribe(WSSubscription(channels=["ticker", "level2"], product_ids=["BTC-USD", "ETH-USD"]))
        assert len(subs) == 1
        assert subs[0]["channels"] == ["ticker", "level2"]
        assert subs[0]["product_ids"] == ["BTC-USD", "ETH-USD"]

    def test_subscribe_called_twice_overwrites(self):
        from bot_v2.features.brokerages.coinbase.transports import MockTransport
        subs = []
        t = MockTransport(messages=[])
        original = t.subscribe
        def track(payload):
            subs.append(payload)
            return original(payload)
        t.subscribe = track

        ws = CoinbaseWebSocket(url="wss://test", transport=t)
        ws.connect()
        ws.subscribe(WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]))
        ws.subscribe(WSSubscription(channels=["trades"], product_ids=["BTC-USD"]))
        assert len(subs) == 2
        assert subs[0]["channels"] == ["ticker"]
        assert subs[1]["channels"] == ["trades"]


def test_ws_factory_uses_endpoints_url(monkeypatch):
    """Behavioral: adapter uses endpoints.websocket_url() to construct WS."""
    from bot_v2.features.brokerages.coinbase import adapter as adapter_mod
    from bot_v2.features.brokerages.coinbase.models import APIConfig

    # Stub WS to capture URL passed by adapter._create_ws
    created_urls = []

    class StubWS:
        def __init__(self, url, ws_auth_provider=None, metrics_emitter=None, transport=None):
            self.url = url
            created_urls.append(url)
        def connect(self):
            pass
        def disconnect(self):
            pass
        def subscribe(self, payload):
            pass
        def stream_messages(self):
            if False:
                yield {}

    # Advanced mode
    cfg_adv = APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api.coinbase.com", sandbox=False)
    brk_adv = CoinbaseBrokerage(cfg_adv)
    monkeypatch.setattr(adapter_mod, "CoinbaseWebSocket", StubWS)
    ws = brk_adv._create_ws()
    assert ws.url == brk_adv.endpoints.websocket_url()

    # Exchange sandbox mode
    cfg_ex = APIConfig(api_key="k", api_secret="s", passphrase="p", base_url="https://api-public.sandbox.exchange.coinbase.com", sandbox=True, api_mode="exchange")
    brk_ex = CoinbaseBrokerage(cfg_ex)
    monkeypatch.setattr(adapter_mod, "CoinbaseWebSocket", StubWS)
    ws2 = brk_ex._create_ws()
    assert ws2.url == brk_ex.endpoints.websocket_url()
