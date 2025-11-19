"""
WebSocket subscription management tests.
"""

from __future__ import annotations

from unittest.mock import Mock

from bot_v2.features.brokerages.coinbase.transports import (
    MockTransport,
)
from bot_v2.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket,
    WSSubscription,
)


class TestCoinbaseWebSocketSubscription:
    """Test WebSocket subscription management."""

    def test_subscribe_single_subscription(self, mock_runtime_settings):
        """Test subscribing to a single channel."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])
        ws.subscribe(subscription)

        assert subscription in ws._subscriptions
        mock_transport.subscribe.assert_called_once()

    def test_subscribe_multiple_subscriptions(self, mock_runtime_settings):
        """Test subscribing to multiple channels."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        subscriptions = [
            WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]),
            WSSubscription(channels=["matches"], product_ids=["ETH-USD"]),
        ]

        for sub in subscriptions:
            ws.subscribe(sub)

        assert len(ws._subscriptions) == 2
        assert all(sub in ws._subscriptions for sub in subscriptions)
        assert mock_transport.subscribe.call_count == 2

    def test_subscribe_without_transport(self, mock_runtime_settings):
        """Test subscribing without active transport."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])

        # Should not raise exception
        ws.subscribe(subscription)
        assert subscription in ws._subscriptions

    def test_subscribe_duplicate_handling(self, mock_runtime_settings):
        """Test handling of duplicate subscriptions."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])

        # Subscribe twice
        ws.subscribe(subscription)
        ws.subscribe(subscription)

        # Should only have one subscription
        assert ws._subscriptions.count(subscription) == 1
        assert mock_transport.subscribe.call_count == 2  # Still called twice

    def test_unsubscribe_existing(self, mock_runtime_settings):
        """Test unsubscribing from existing subscription."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = MockTransport()
        ws._transport = mock_transport

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])
        ws.subscribe(subscription)
        ws.unsubscribe(subscription)

        assert subscription not in ws._subscriptions

    def test_unsubscribe_nonexistent(self, mock_runtime_settings):
        """Test unsubscribing from non-existent subscription."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])

        # Should not raise exception
        ws.unsubscribe(subscription)
        assert len(ws._subscriptions) == 0

    def test_get_subscription_message_format(self, mock_runtime_settings):
        """Test subscription message format."""
        subscription = WSSubscription(
            channels=["ticker", "matches"], product_ids=["BTC-USD", "ETH-USD"]
        )

        message = subscription.to_dict()

        expected = {
            "type": "subscribe",
            "channels": ["ticker", "matches"],
            "product_ids": ["BTC-USD", "ETH-USD"],
        }

        assert message == expected

    def test_subscription_error_handling(self, mock_runtime_settings):
        """Test error handling in subscription operations."""
        ws = CoinbaseWebSocket(
            url="wss://test.example.com",
            ws_auth_provider=None,
            settings=mock_runtime_settings,
        )

        mock_transport = Mock()
        mock_transport.subscribe.side_effect = RuntimeError("Subscription failed")
        ws._transport = mock_transport

        subscription = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])

        # Should not raise exception
        ws.subscribe(subscription)
        assert subscription in ws._subscriptions
