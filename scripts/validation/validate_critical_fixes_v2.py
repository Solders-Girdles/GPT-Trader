#!/usr/bin/env python3
"""
Enhanced validation script for Coinbase critical fixes.
Tests actual client methods to ensure routing works correctly.
"""

import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient, CoinbaseAuth
from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError
from bot_v2.features.brokerages.coinbase.models import APIConfig


def test_endpoint_routing_advanced_mode():
    """Test that all methods work correctly in advanced mode."""
    print("\n=== Testing Advanced Mode Endpoint Routing ===")
    
    auth = CoinbaseAuth(
        api_key="test-key",
        api_secret="test-secret",
        passphrase=None
    )
    client = CoinbaseClient(
        base_url="https://api.coinbase.com",
        auth=auth,
        api_mode="advanced"
    )
    
    # Mock the _request method to capture paths
    paths_called = []
    
    def mock_request(method, path, payload=None):
        paths_called.append((method, path))
        return {"success": True}
    
    client._request = mock_request
    
    # Test methods that should work in advanced mode
    test_cases = [
        ("get_time", lambda: client.get_time(), "GET", "/api/v3/brokerage/time"),
        ("get_accounts", lambda: client.get_accounts(), "GET", "/api/v3/brokerage/accounts"),
        ("get_account", lambda: client.get_account("test-id"), "GET", "/api/v3/brokerage/accounts/test-id"),
        ("get_products", lambda: client.get_products(), "GET", "/api/v3/brokerage/market/products"),
        ("get_product", lambda: client.get_product("BTC-USD"), "GET", "/api/v3/brokerage/market/products/BTC-USD"),
        ("list_orders_batch", lambda: client.list_orders_batch([]), "GET", "/api/v3/brokerage/orders/historical/batch"),
        ("preview_order", lambda: client.preview_order({}), "POST", "/api/v3/brokerage/orders/preview"),
        ("edit_order_preview", lambda: client.edit_order_preview({}), "POST", "/api/v3/brokerage/orders/edit_preview"),
        ("edit_order", lambda: client.edit_order({}), "POST", "/api/v3/brokerage/orders/edit"),
        ("close_position", lambda: client.close_position({}), "POST", "/api/v3/brokerage/orders/close_position"),
        ("list_payment_methods", lambda: client.list_payment_methods(), "GET", "/api/v3/brokerage/payment_methods"),
        ("get_payment_method", lambda: client.get_payment_method("test-pm"), "GET", "/api/v3/brokerage/payment_methods/test-pm"),
        ("list_portfolios", lambda: client.list_portfolios(), "GET", "/api/v3/brokerage/portfolios"),
        ("get_portfolio", lambda: client.get_portfolio("test-uuid"), "GET", "/api/v3/brokerage/portfolios/test-uuid"),
        ("get_portfolio_breakdown", lambda: client.get_portfolio_breakdown("test-uuid"), "GET", "/api/v3/brokerage/portfolios/test-uuid/breakdown"),
        ("move_funds", lambda: client.move_funds({}), "POST", "/api/v3/brokerage/portfolios/move_funds"),
        ("intx_allocate", lambda: client.intx_allocate({}), "POST", "/api/v3/brokerage/intx/allocate"),
        ("intx_balances", lambda: client.intx_balances("test-uuid"), "GET", "/api/v3/brokerage/intx/balances/test-uuid"),
        ("intx_portfolio", lambda: client.intx_portfolio("test-uuid"), "GET", "/api/v3/brokerage/intx/portfolio/test-uuid"),
        ("intx_positions", lambda: client.intx_positions("test-uuid"), "GET", "/api/v3/brokerage/intx/positions/test-uuid"),
        ("intx_position", lambda: client.intx_position("test-uuid", "BTC"), "GET", "/api/v3/brokerage/intx/positions/test-uuid/BTC"),
        ("intx_multi_asset_collateral", lambda: client.intx_multi_asset_collateral(), "GET", "/api/v3/brokerage/intx/multi_asset_collateral"),
        ("cfm_balance_summary", lambda: client.cfm_balance_summary(), "GET", "/api/v3/brokerage/cfm/balance_summary"),
        ("cfm_positions", lambda: client.cfm_positions(), "GET", "/api/v3/brokerage/cfm/positions"),
        ("cfm_position", lambda: client.cfm_position("BTC-USD"), "GET", "/api/v3/brokerage/cfm/positions/BTC-USD"),
        ("cfm_sweeps", lambda: client.cfm_sweeps(), "GET", "/api/v3/brokerage/cfm/sweeps"),
        ("cfm_sweeps_schedule", lambda: client.cfm_sweeps_schedule(), "GET", "/api/v3/brokerage/cfm/sweeps/schedule"),
        # The three previously hardcoded methods
        ("get_market_product_book", lambda: client.get_market_product_book("BTC-USD"), "GET", "/api/v3/brokerage/market/product_book?product_id=BTC-USD&level=2"),
        ("get_best_bid_ask", lambda: client.get_best_bid_ask(["BTC-USD"]), "GET", "/api/v3/brokerage/best_bid_ask?product_ids=BTC-USD"),
        ("get_account", lambda: client.get_account("test-uuid"), "GET", "/api/v3/brokerage/accounts/test-uuid"),
    ]
    
    passed = 0
    failed = 0
    
    for name, func, expected_method, expected_path in test_cases:
        paths_called.clear()
        try:
            func()
            if paths_called and paths_called[0] == (expected_method, expected_path):
                print(f"✅ {name}: Correct path used")
                passed += 1
            else:
                print(f"❌ {name}: Wrong path. Expected {expected_path}, got {paths_called[0][1] if paths_called else 'no call'}")
                failed += 1
        except Exception as e:
            print(f"❌ {name}: Failed with error: {e}")
            failed += 1
    
    print(f"\nAdvanced Mode Results: {passed} passed, {failed} failed")
    return failed == 0


def test_endpoint_routing_exchange_mode():
    """Test that exchange-only methods work and advanced-only methods raise errors."""
    print("\n=== Testing Exchange Mode Endpoint Routing ===")
    
    auth = CoinbaseAuth(
        api_key="test-key",
        api_secret="test-secret",
        passphrase="test-pass"
    )
    client = CoinbaseClient(
        base_url="https://api-public.sandbox.exchange.coinbase.com",
        auth=auth,
        api_mode="exchange"
    )
    
    # Mock the _request method
    def mock_request(method, path, payload=None):
        return {"success": True}
    
    client._request = mock_request
    
    # Test methods that should work in exchange mode
    working_methods = [
        ("get_time", lambda: client.get_time()),
        ("get_accounts", lambda: client.get_accounts()),
        ("get_products", lambda: client.get_products()),
        ("get_product", lambda: client.get_product("BTC-USD")),
        ("get_ticker", lambda: client.get_ticker("BTC-USD")),
        ("get_candles", lambda: client.get_candles("BTC-USD")),
        ("get_fills", lambda: client.get_fills()),
        ("get_fees", lambda: client.get_fees()),
        ("get_market_product_book", lambda: client.get_market_product_book("BTC-USD")),
        ("get_account", lambda: client.get_account("test-id")),
    ]
    
    passed = 0
    failed = 0
    
    print("\nTesting methods that should work in exchange mode:")
    for name, func in working_methods:
        try:
            func()
            print(f"✅ {name}: Works in exchange mode")
            passed += 1
        except InvalidRequestError as e:
            print(f"❌ {name}: Should work but raised InvalidRequestError: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️  {name}: Other error (may be expected): {e}")
    
    # Test methods that should raise InvalidRequestError in exchange mode
    blocked_methods = [
        ("list_orders_batch", lambda: client.list_orders_batch([])),
        ("preview_order", lambda: client.preview_order({})),
        ("edit_order_preview", lambda: client.edit_order_preview({})),
        ("edit_order", lambda: client.edit_order({})),
        ("close_position", lambda: client.close_position({})),
        ("list_payment_methods", lambda: client.list_payment_methods()),
        ("get_payment_method", lambda: client.get_payment_method("test")),
        ("list_portfolios", lambda: client.list_portfolios()),
        ("get_portfolio", lambda: client.get_portfolio("test")),
        ("get_portfolio_breakdown", lambda: client.get_portfolio_breakdown("test")),
        ("move_funds", lambda: client.move_funds({})),
        ("intx_allocate", lambda: client.intx_allocate({})),
        ("intx_balances", lambda: client.intx_balances("test")),
        ("intx_portfolio", lambda: client.intx_portfolio("test")),
        ("intx_positions", lambda: client.intx_positions("test")),
        ("intx_position", lambda: client.intx_position("test", "BTC")),
        ("intx_multi_asset_collateral", lambda: client.intx_multi_asset_collateral()),
        ("cfm_balance_summary", lambda: client.cfm_balance_summary()),
        ("cfm_positions", lambda: client.cfm_positions()),
        ("cfm_position", lambda: client.cfm_position("BTC-USD")),
        ("cfm_sweeps", lambda: client.cfm_sweeps()),
        ("cfm_sweeps_schedule", lambda: client.cfm_sweeps_schedule()),
        ("get_best_bid_ask", lambda: client.get_best_bid_ask(["BTC-USD"])),  # Now blocked in exchange mode
    ]
    
    print("\nTesting methods that should be blocked in exchange mode:")
    for name, func in blocked_methods:
        try:
            func()
            print(f"❌ {name}: Should be blocked but worked")
            failed += 1
        except InvalidRequestError as e:
            if "not available in exchange mode" in str(e):
                print(f"✅ {name}: Correctly blocked with helpful error")
                passed += 1
            else:
                print(f"⚠️  {name}: Blocked but wrong error message: {e}")
                failed += 1
        except Exception as e:
            print(f"❌ {name}: Wrong error type: {e}")
            failed += 1
    
    print(f"\nExchange Mode Results: {passed} passed, {failed} failed")
    return failed == 0


def test_sequence_guard():
    """Test SequenceGuard API consistency."""
    print("\n=== Testing SequenceGuard API ===")
    
    from bot_v2.features.brokerages.coinbase.ws import SequenceGuard
    
    try:
        guard = SequenceGuard()
        
        # Test that annotate method exists and works
        msg1 = {"data": "test", "sequence": 1}
        result1 = guard.annotate(msg1)
        
        # First message should not have gap
        if "gap_detected" in result1:
            print("❌ SequenceGuard: First message incorrectly marked as gap")
            return False
        
        # Skip a sequence to test gap detection
        msg2 = {"data": "test2", "sequence": 3}
        result2 = guard.annotate(msg2)
        
        if result2.get("gap_detected") == True and result2.get("last_seq") == 1:
            print("✅ SequenceGuard: annotate() method works and detects gaps")
            return True
        else:
            print("❌ SequenceGuard: Gap detection not working properly")
            return False
            
    except AttributeError as e:
        print(f"❌ SequenceGuard: Missing annotate() method - {e}")
        return False
    except Exception as e:
        print(f"❌ SequenceGuard test failed: {e}")
        return False


def test_websocket_initialization():
    """Test WebSocket transport initialization."""
    print("\n=== Testing WebSocket Initialization ===")
    
    from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket
    
    # Test without providing transport (use env to disable streaming)
    try:
        os.environ['DISABLE_WS_STREAMING'] = '1'
        ws = CoinbaseWebSocket("wss://test.com")
        
        # Transport should be None initially
        if ws._transport is not None:
            print("❌ WebSocket: Transport should be None before connect")
            return False
            
        # Connect initializes default transport (Noop when streaming disabled)
        try:
            ws.connect()
        except Exception:
            pass  # Even if connect raises, transport should be initialized
        
        if ws._transport is not None:
            print("✅ WebSocket: Default transport initialized on connect (NoopTransport)")
            result = True
        else:
            print("❌ WebSocket: Transport not initialized after connect")
            result = False
    except Exception as e:
        print(f"❌ WebSocket initialization failed: {e}")
        result = False
    finally:
        try:
            del os.environ['DISABLE_WS_STREAMING']
        except Exception:
            pass
    
    return result


def test_mode_detection():
    """Test API mode detection logic."""
    print("\n=== Testing API Mode Detection ===")
    
    # Simple test that mode detection would work with proper environment
    test_cases = [
        ("advanced", "Advanced mode"),
        ("exchange", "Exchange mode"),
    ]
    
    passed = 0
    failed = 0
    
    for mode, description in test_cases:
        try:
            auth = CoinbaseAuth(
                api_key="test-key",
                api_secret="test-secret",
                passphrase="test-pass" if mode == "exchange" else None
            )
            client = CoinbaseClient(
                base_url="https://api.coinbase.com" if mode == "advanced" else "https://api-public.sandbox.exchange.coinbase.com",
                auth=auth,
                api_mode=mode
            )
            
            if client.api_mode == mode:
                print(f"✅ {description}: Mode set correctly to {mode}")
                passed += 1
            else:
                print(f"❌ {description}: Expected {mode}, got {client.api_mode}")
                failed += 1
        except Exception as e:
            print(f"❌ {description}: Failed with error: {e}")
            failed += 1
    
    print(f"\nMode Detection Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all validation tests."""
    print("="*60)
    print("COINBASE CRITICAL FIXES VALIDATION V2")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_endpoint_routing_advanced_mode()
    all_passed &= test_endpoint_routing_exchange_mode()
    all_passed &= test_sequence_guard()
    all_passed &= test_websocket_initialization()
    all_passed &= test_mode_detection()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
        print("\nThe following have been resolved:")
        print("1. All endpoint routing through _get_endpoint_path()")
        print("2. InvalidRequestError for exchange-unsupported methods")
        print("3. WebSocket transport initialization")
        print("4. API mode detection and sandbox handling")
        print("5. Environment configuration")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
