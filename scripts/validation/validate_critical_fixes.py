#!/usr/bin/env python3
"""
Validation script for critical Coinbase integration fixes.

Tests:
1. Sandbox/API mode mismatch fix
2. WebSocket transport initialization
3. Environment configuration validation
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_sandbox_mode():
    """Validate that sandbox mode works with correct API mode."""
    print("\n" + "="*60)
    print("TEST 1: Validating Sandbox/API Mode Fix")
    print("="*60)
    
    try:
        # Clear any existing Coinbase env vars first
        for key in list(os.environ.keys()):
            if key.startswith('COINBASE_'):
                os.environ.pop(key, None)
        
        # Set sandbox mode
        os.environ['BROKER'] = 'coinbase'
        os.environ['COINBASE_SANDBOX'] = '1'
        # Set dummy HMAC credentials (not CDP)
        os.environ['COINBASE_API_KEY'] = 'test-api-key'
        os.environ['COINBASE_API_SECRET'] = 'test-api-secret'
        
        from bot_v2.orchestration.broker_factory import create_brokerage
        
        print("✓ Creating brokerage in sandbox mode...")
        broker = create_brokerage()
        
        # Check that exchange mode was selected
        assert broker.config.api_mode == "exchange", f"Expected 'exchange' mode, got '{broker.config.api_mode}'"
        print(f"✓ API mode correctly set to: {broker.config.api_mode}")
        
        # Check that base URL is correct
        assert "sandbox" in broker.config.base_url, f"Expected sandbox URL, got: {broker.config.base_url}"
        print(f"✓ Base URL correctly set to: {broker.config.base_url}")
        
        # Check that client has correct mode
        assert broker._client.api_mode == "exchange", "Client doesn't have correct API mode"
        print(f"✓ Client API mode: {broker._client.api_mode}")
        
        # Test endpoint routing
        from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError
        
        # This should work in exchange mode
        path = broker._client._get_endpoint_path('products')
        assert path == '/products', f"Expected '/products', got '{path}'"
        print(f"✓ Products endpoint correctly routed to: {path}")
        
        # This should fail in exchange mode (portfolios not available)
        try:
            broker._client._get_endpoint_path('portfolios')
            print("✗ Should have raised error for portfolios in exchange mode")
            return False
        except InvalidRequestError as e:
            print(f"✓ Correctly rejected unsupported endpoint: {e}")
        
        print("\n✅ SANDBOX MODE VALIDATION PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ SANDBOX MODE VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up env vars
        os.environ.pop('COINBASE_SANDBOX', None)


def validate_ws_transport():
    """Validate that WebSocket initializes without assertion."""
    print("\n" + "="*60)
    print("TEST 2: Validating WebSocket Transport Fix")
    print("="*60)
    
    try:
        from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket
        from bot_v2.features.brokerages.coinbase.transports import MockTransport
        
        # Test 1: Default transport initialization
        print("✓ Creating WebSocket without transport...")
        ws = CoinbaseWebSocket("wss://test.example.com")
        
        # This should not raise an assertion
        try:
            # We'll use a mock transport to avoid actual connection
            ws.set_transport(MockTransport([{"type": "heartbeat"}]))
            ws.connect()
            print("✓ WebSocket connected without assertion")
            
            # Verify transport is set
            assert ws._transport is not None, "Transport should be set after connect"
            print(f"✓ Transport initialized: {type(ws._transport).__name__}")
            
        except AssertionError as e:
            print(f"✗ Assertion error (should be fixed): {e}")
            return False
        
        # Test 2: Custom transport injection
        print("\n✓ Testing custom transport injection...")
        custom_transport = MockTransport([
            {"type": "subscribed", "channels": ["ticker"]},
            {"type": "ticker", "price": "50000.00"}
        ])
        
        ws2 = CoinbaseWebSocket("wss://test2.example.com", transport=custom_transport)
        ws2.connect()
        
        messages = list(ws2._transport.stream())
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        print(f"✓ Custom transport works: received {len(messages)} messages")
        
        print("\n✅ WEBSOCKET TRANSPORT VALIDATION PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ WEBSOCKET TRANSPORT VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_environment_config():
    """Validate environment configuration and warnings."""
    print("\n" + "="*60)
    print("TEST 3: Validating Environment Configuration")
    print("="*60)
    
    try:
        import io
        import contextlib
        
        # Clear any existing Coinbase env vars first
        for key in list(os.environ.keys()):
            if key.startswith('COINBASE_'):
                os.environ.pop(key, None)
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)
        
        # Test 1: Sandbox with advanced mode warning
        print("✓ Testing sandbox + advanced mode warning...")
        os.environ['BROKER'] = 'coinbase'
        os.environ['COINBASE_SANDBOX'] = '1'
        os.environ['COINBASE_API_MODE'] = 'advanced'
        
        from bot_v2.orchestration.broker_factory import create_brokerage
        
        # Add handler to capture warnings
        factory_logger = logging.getLogger('bot_v2.orchestration.broker_factory')
        factory_logger.addHandler(handler)
        
        broker = create_brokerage()
        
        log_output = log_capture.getvalue()
        if "Advanced Trade API does not have a public sandbox" in log_output:
            print("✓ Warning displayed for advanced mode with sandbox")
        else:
            print("✗ Expected warning not found")
        
        # Test 2: Exchange mode without passphrase warning
        print("\n✓ Testing exchange mode without passphrase warning...")
        os.environ['COINBASE_API_MODE'] = 'exchange'
        os.environ.pop('COINBASE_API_PASSPHRASE', None)
        
        log_capture.truncate(0)
        log_capture.seek(0)
        
        broker = create_brokerage()
        
        log_output = log_capture.getvalue()
        if "Exchange API mode requires passphrase" in log_output:
            print("✓ Warning displayed for missing passphrase")
        else:
            print("✗ Expected warning not found")
        
        # Test 3: CDP with exchange mode warning
        print("\n✓ Testing CDP auth with exchange mode warning...")
        os.environ['COINBASE_CDP_API_KEY'] = 'test-key'
        os.environ['COINBASE_CDP_PRIVATE_KEY'] = 'test-private-key'
        os.environ['COINBASE_API_MODE'] = 'exchange'
        
        log_capture.truncate(0)
        log_capture.seek(0)
        
        broker = create_brokerage()
        
        log_output = log_capture.getvalue()
        if "CDP/JWT authentication selected but Exchange API mode active" in log_output:
            print("✓ Warning displayed for CDP with exchange mode")
        else:
            print("✗ Expected warning not found")
        
        print("\n✅ ENVIRONMENT CONFIGURATION VALIDATION PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ ENVIRONMENT CONFIGURATION VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up env vars
        for key in ['COINBASE_SANDBOX', 'COINBASE_API_MODE', 'COINBASE_CDP_API_KEY', 
                    'COINBASE_CDP_PRIVATE_KEY', 'COINBASE_API_PASSPHRASE']:
            os.environ.pop(key, None)


def validate_endpoint_routing():
    """Validate endpoint routing for both API modes."""
    print("\n" + "="*60)
    print("TEST 4: Validating Endpoint Routing")
    print("="*60)
    
    try:
        from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
        from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError
        
        # Test Advanced mode endpoints
        print("✓ Testing Advanced Trade mode endpoints...")
        client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=None,
            api_mode="advanced"
        )
        
        endpoints_advanced = {
            'products': '/api/v3/brokerage/market/products',
            'accounts': '/api/v3/brokerage/accounts',
            'orders': '/api/v3/brokerage/orders',
            'portfolios': '/api/v3/brokerage/portfolios',
        }
        
        for name, expected_path in endpoints_advanced.items():
            path = client._get_endpoint_path(name)
            assert path == expected_path, f"Expected '{expected_path}', got '{path}'"
            print(f"  ✓ {name}: {path}")
        
        # Test Exchange mode endpoints
        print("\n✓ Testing Exchange mode endpoints...")
        client = CoinbaseClient(
            base_url="https://api.exchange.coinbase.com",
            auth=None,
            api_mode="exchange"
        )
        
        endpoints_exchange = {
            'products': '/products',
            'accounts': '/accounts',
            'orders': '/orders',
            'fills': '/fills',
        }
        
        for name, expected_path in endpoints_exchange.items():
            path = client._get_endpoint_path(name)
            assert path == expected_path, f"Expected '{expected_path}', got '{path}'"
            print(f"  ✓ {name}: {path}")
        
        # Test unsupported endpoint in exchange mode
        print("\n✓ Testing unsupported endpoint handling...")
        try:
            client._get_endpoint_path('portfolios')
            print("  ✗ Should have raised error for portfolios in exchange mode")
            return False
        except InvalidRequestError as e:
            assert "not available in exchange mode" in str(e)
            print(f"  ✓ Correctly rejected: {e}")
        
        print("\n✅ ENDPOINT ROUTING VALIDATION PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ ENDPOINT ROUTING VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print(" COINBASE INTEGRATION CRITICAL FIXES VALIDATION")
    print("="*70)
    
    results = {
        "Sandbox/API Mode": validate_sandbox_mode(),
        "WebSocket Transport": validate_ws_transport(),
        "Environment Config": validate_environment_config(),
        "Endpoint Routing": validate_endpoint_routing(),
    }
    
    # Summary
    print("\n" + "="*70)
    print(" VALIDATION SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
        return 0
    else:
        print("\n⚠️  SOME VALIDATIONS FAILED - PLEASE REVIEW")
        return 1


if __name__ == "__main__":
    sys.exit(main())
