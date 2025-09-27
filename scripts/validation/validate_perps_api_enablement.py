#!/usr/bin/env python3
"""
Validation script for Derivatives API Enablement.
Quick checks to verify CFM endpoints are properly routed and gated.
"""

import sys
import os
from typing import Tuple, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_advanced_mode_routing() -> Tuple[bool, str]:
    """Test that CFM methods route correctly in advanced mode."""
    try:
        from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
        
        # Create client in advanced mode
        client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            api_mode="advanced"
        )
        
        # Mock transport to capture paths
        captured_paths = []
        
        def mock_transport(method, url, headers, body, timeout):
            path = url.replace("https://api.coinbase.com", "")
            captured_paths.append(path)
            return (200, {}, '{"success": true}')
        
        client.set_transport_for_testing(mock_transport)
        
        # Test cfm_positions routing
        client.cfm_positions()
        if "/api/v3/brokerage/cfm/positions" not in captured_paths:
            return False, f"cfm_positions routed incorrectly: {captured_paths[-1]}"
        
        # Test close_position routing
        client.close_position({"product_id": "BTC-PERP"})
        if "/api/v3/brokerage/orders/close_position" not in captured_paths:
            return False, f"close_position routed incorrectly: {captured_paths[-1]}"
        
        return True, "Advanced mode routing: PASSED"
        
    except Exception as e:
        return False, f"Advanced mode routing: FAILED - {e}"


def test_exchange_mode_gating() -> Tuple[bool, str]:
    """Test that CFM methods are blocked in exchange mode."""
    try:
        from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
        from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError
        
        # Create client in exchange mode
        client = CoinbaseClient(
            base_url="https://api-public.sandbox.exchange.coinbase.com",
            api_mode="exchange"
        )
        
        # Test cfm_positions is blocked
        try:
            client.cfm_positions()
            return False, "cfm_positions should be blocked in exchange mode"
        except InvalidRequestError as e:
            if "not available in exchange mode" not in str(e):
                return False, f"Wrong error message: {e}"
        
        # Test close_position is blocked
        try:
            client.close_position({"product_id": "BTC-PERP"})
            return False, "close_position should be blocked in exchange mode"
        except InvalidRequestError as e:
            if "not available in exchange mode" not in str(e):
                return False, f"Wrong error message: {e}"
        
        return True, "Exchange mode gating: PASSED"
        
    except Exception as e:
        return False, f"Exchange mode gating: FAILED - {e}"


def test_adapter_smoke() -> Tuple[bool, str]:
    """Test adapter can handle derivatives operations with mocks."""
    try:
        from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
        from bot_v2.features.brokerages.coinbase.models import APIConfig
        from unittest.mock import MagicMock
        
        # Create adapter with derivatives enabled
        config = APIConfig(
            api_key="test",
            api_secret="test",
            passphrase=None,
            base_url="https://api.coinbase.com",
            sandbox=False,
            enable_derivatives=True
        )
        
        adapter = CoinbaseBrokerage(config)
        
        # Mock cfm_positions response
        mock_positions = {
            "positions": [
                {
                    "product_id": "BTC-PERP",
                    "side": "LONG",
                    "qty": "1.0",
                    "entry_price": "50000"
                }
            ]
        }
        adapter._client.cfm_positions = MagicMock(return_value=mock_positions)
        
        # Test list_positions
        positions = adapter.list_positions()
        if not isinstance(positions, list):
            return False, f"list_positions should return list, got {type(positions)}"
        
        # Test close_position exists
        if not hasattr(adapter, 'close_position'):
            return False, "Adapter missing close_position method"
        
        # Mock close_position response
        adapter._client.close_position = MagicMock(return_value={"order_id": "123"})
        
        # Test close_position
        result = adapter.close_position("BTC-PERP")
        if "order_id" not in result:
            return False, "close_position should return order_id"
        
        return True, "Adapter smoke test: PASSED"
        
    except Exception as e:
        return False, f"Adapter smoke test: FAILED - {e}"


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Derivatives API Enablement Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("Advanced Mode Routing", test_advanced_mode_routing),
        ("Exchange Mode Gating", test_exchange_mode_gating),
        ("Adapter Smoke Test", test_adapter_smoke)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        passed, message = test_func()
        
        if passed:
            print(f"  ✅ {message}")
        else:
            print(f"  ❌ {message}")
            all_passed = False
        print()
    
    # Summary
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED - Review and fix")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
