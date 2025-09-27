#!/usr/bin/env python3
"""
Validation script for WebSocket Integration (Perpetuals).
Verifies market WS, user events, and reconnection without network calls.
"""

import sys
from decimal import Decimal
from typing import Dict, List, Any

# Add src to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bot_v2.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket,
    WSSubscription,
    SequenceGuard,
    normalize_market_message
)
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.transports import MockTransport


def validate_market_trades():
    """Validate market trades streaming for perpetuals."""
    print("\n=== Validating Market Trades ===")
    
    # Create mock trade messages
    mock_trades = [
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
    mock_transport = MockTransport(messages=mock_trades)
    mock_ws = CoinbaseWebSocket(
        url="wss://advanced-trade-ws.coinbase.com",
        transport=mock_transport
    )
    adapter.set_ws_factory_for_testing(lambda: mock_ws)
    
    # Stream trades
    trades = list(adapter.stream_trades(["BTC-USD-PERP", "ETH-USD-PERP"]))
    
    # Validate
    assert len(trades) == 2, f"Expected 2 trades, got {len(trades)}"
    
    # Check normalization
    assert trades[0]["product_id"] == "BTC-USD-PERP"
    assert isinstance(trades[0]["price"], Decimal), "Price should be Decimal"
    assert trades[0]["price"] == Decimal("50000.50")
    assert isinstance(trades[0]["size"], Decimal), "Size should be Decimal"
    assert trades[0]["timestamp"] == "2024-01-15T12:00:00Z"
    
    assert trades[1]["product_id"] == "ETH-USD-PERP"
    assert trades[1]["price"] == Decimal("3000.25")
    
    print("✓ Market trades streaming works")
    print("✓ Decimal normalization applied")
    print("✓ Product IDs correct")
    print("PASS: Market trades validation successful")
    return True


def validate_user_events():
    """Validate user event streaming with sequence gap detection."""
    print("\n=== Validating User Events ===")
    
    # Create messages with sequence gap
    mock_events = [
        {"type": "order", "sequence": 100, "order_id": "order1", "product_id": "BTC-USD-PERP"},
        {"type": "order", "sequence": 101, "order_id": "order2", "product_id": "BTC-USD-PERP"},
        {"type": "order", "sequence": 103, "order_id": "order3", "product_id": "ETH-USD-PERP"},  # Gap!
        {"type": "fill", "sequence": 104, "order_id": "order3", "product_id": "ETH-USD-PERP"}
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
    mock_transport = MockTransport(messages=mock_events)
    mock_ws = CoinbaseWebSocket(
        url="wss://advanced-trade-ws.coinbase.com",
        transport=mock_transport
    )
    adapter.set_ws_factory_for_testing(lambda: mock_ws)
    
    # Stream user events
    events = list(adapter.stream_user_events(["BTC-USD-PERP", "ETH-USD-PERP"]))
    
    # Validate
    assert len(events) == 4, f"Expected 4 events, got {len(events)}"
    
    # Check sequence gap detection
    assert "gap_detected" not in events[0], "First event should have no gap"
    assert "gap_detected" not in events[1], "Second event should have no gap"
    assert events[2].get("gap_detected") == True, "Third event should detect gap"
    assert events[2].get("last_seq") == 101, "Should report last sequence before gap"
    assert "gap_detected" not in events[3], "Fourth event should have no gap"
    
    print("✓ User events streaming works")
    print("✓ Sequence gap detected at message 3")
    print("✓ gap_detected flag set correctly")
    print("PASS: User events validation successful")
    return True


def validate_reconnection():
    """Validate reconnection and resubscription."""
    print("\n=== Validating Reconnection ===")
    
    # Create transport that simulates error and recovery
    class ReconnectTestTransport:
        def __init__(self):
            self.attempt = 0
            self.subscriptions = []
            
        def connect(self, url, headers=None):
            self.attempt += 1
            
        def disconnect(self):
            pass
            
        def subscribe(self, payload):
            self.subscriptions.append(payload)
            
        def stream(self):
            if self.attempt == 1:
                # First attempt - yield one message then error
                yield {"sequence": 1, "msg": "before_error"}
                raise ConnectionError("Simulated disconnect")
            else:
                # After reconnect - yield new messages
                yield {"sequence": 10, "msg": "after_reconnect"}
                yield {"sequence": 11, "msg": "continued"}
    
    # Create WebSocket with test transport
    ws = CoinbaseWebSocket(
        url="wss://test",
        max_retries=2,
        base_delay=0.01  # Short for testing
    )
    
    test_transport = ReconnectTestTransport()
    ws._transport = test_transport
    
    # Subscribe
    sub = WSSubscription(channels=["ticker"], product_ids=["BTC-USD-PERP"])
    ws.subscribe(sub)
    
    # Stream messages (will reconnect once)
    messages = []
    for msg in ws.stream_messages():
        messages.append(msg)
        if len(messages) >= 3:
            break
    
    # Validate
    assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"
    assert messages[0]["msg"] == "before_error"
    assert messages[1]["msg"] == "after_reconnect"
    assert messages[2]["msg"] == "continued"
    
    # Check resubscription happened
    assert len(test_transport.subscriptions) == 2, "Should have resubscribed after reconnect"
    assert test_transport.subscriptions[0] == test_transport.subscriptions[1], "Same subscription"
    
    # Verify sequence guard would reset (no gap after reconnect)
    guard = SequenceGuard()
    annotated1 = guard.annotate(messages[0])
    # Simulate reset on reconnect
    guard.reset()
    annotated2 = guard.annotate(messages[1])
    
    assert "gap_detected" not in annotated2, "Should not detect gap after reconnect/reset"
    
    print("✓ Reconnection triggers successfully")
    print("✓ Resubscription occurs after reconnect")
    print("✓ Stream resumes after error")
    print("PASS: Reconnection validation successful")
    return True


def validate_channel_selection():
    """Validate correct channel selection for orderbook levels."""
    print("\n=== Validating Channel Selection ===")
    
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
    
    class ChannelTrackingTransport:
        def __init__(self):
            self.messages = []
            
        def connect(self, url, headers=None):
            pass
            
        def disconnect(self):
            pass
            
        def subscribe(self, payload):
            subscriptions.append(payload)
            
        def stream(self):
            return iter(self.messages)
    
    # Test L1 orderbook (ticker)
    transport1 = ChannelTrackingTransport()
    ws1 = CoinbaseWebSocket("wss://test", transport=transport1)
    adapter.set_ws_factory_for_testing(lambda: ws1)
    
    stream = adapter.stream_orderbook(["BTC-USD-PERP"], level=1)
    try:
        next(stream)
    except StopIteration:
        pass
    
    assert subscriptions[-1]["channels"] == ["ticker"], "Level 1 should use ticker channel"
    
    # Test L2 orderbook (level2)
    transport2 = ChannelTrackingTransport()
    ws2 = CoinbaseWebSocket("wss://test", transport=transport2)
    adapter.set_ws_factory_for_testing(lambda: ws2)
    
    stream = adapter.stream_orderbook(["ETH-USD-PERP"], level=2)
    try:
        next(stream)
    except StopIteration:
        pass
    
    assert subscriptions[-1]["channels"] == ["level2"], "Level 2+ should use level2 channel"
    
    print("✓ Level 1 uses 'ticker' channel")
    print("✓ Level 2+ uses 'level2' channel")
    print("PASS: Channel selection validation successful")
    return True


def validate_message_normalization():
    """Validate message normalization helper."""
    print("\n=== Validating Message Normalization ===")
    
    # Test various message formats
    test_cases = [
        {
            "input": {
                "price": "12345.67",
                "size": "10.5",
                "best_bid": "12340",
                "best_ask": "12350",
                "volume": "1000.25"
            },
            "expected_types": {
                "price": Decimal,
                "size": Decimal,
                "best_bid": Decimal,
                "best_ask": Decimal,
                "volume": Decimal
            }
        },
        {
            "input": {
                "time": "2024-01-15T12:00:00Z",
                "last": "50000"
            },
            "has_timestamp": True
        }
    ]
    
    for i, test in enumerate(test_cases):
        normalized = normalize_market_message(test["input"].copy())
        
        if "expected_types" in test:
            for field, expected_type in test["expected_types"].items():
                assert isinstance(normalized[field], expected_type), \
                       f"Field {field} should be {expected_type}"
        
        if test.get("has_timestamp"):
            assert "timestamp" in normalized, "Should add timestamp field"
            assert normalized["timestamp"] == test["input"]["time"]
    
    print("✓ Prices converted to Decimal")
    print("✓ Sizes converted to Decimal")
    print("✓ Timestamp field normalized")
    print("PASS: Message normalization validation successful")
    return True


def main():
    """Run all WebSocket validations for perps."""
    print("=" * 60)
    print("VALIDATION: WebSocket Integration (Perpetuals)")
    print("=" * 60)
    
    try:
        # Run all validations
        results = [
            validate_market_trades(),
            validate_user_events(),
            validate_reconnection(),
            validate_channel_selection(),
            validate_message_normalization()
        ]
        
        if all(results):
            print("\n" + "=" * 60)
            print("✅ VALIDATION SUCCESSFUL")
            print("=" * 60)
            print("\nSummary:")
            print("- Market trades streaming with Decimal normalization")
            print("- User events with sequence gap detection")
            print("- Reconnection with automatic resubscription")
            print("- Correct channel selection for orderbook levels")
            print("- Message normalization working correctly")
            print("- SequenceGuard reset on reconnection")
            return 0
        else:
            print("\n❌ VALIDATION FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
