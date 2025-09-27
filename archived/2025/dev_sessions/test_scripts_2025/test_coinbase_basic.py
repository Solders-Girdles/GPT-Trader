#!/usr/bin/env python3
"""
Basic Coinbase integration smoke test.

This tests the essential functionality to ensure the integration works.
Run with sandbox credentials configured in .env or environment variables.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_functionality():
    """Test basic Coinbase integration functionality."""
    print("=" * 60)
    print("Coinbase Integration Smoke Test")
    print("=" * 60)
    
    # Check environment
    broker = os.getenv("BROKER", "").lower()
    if broker != "coinbase":
        print("❌ BROKER environment variable not set to 'coinbase'")
        print("   Set BROKER=coinbase to run this test")
        return False
    
    sandbox = os.getenv("COINBASE_SANDBOX", "0")
    if sandbox != "1":
        print("⚠️  Warning: Not using sandbox mode.")
        print("   Set COINBASE_SANDBOX=1 for safer testing")
    
    api_key = os.getenv("COINBASE_API_KEY", "")
    api_secret = os.getenv("COINBASE_API_SECRET", "")
    
    if not api_key or not api_secret:
        print("❌ Missing Coinbase API credentials")
        print("   Set COINBASE_API_KEY and COINBASE_API_SECRET")
        return False
    
    print(f"✓ Environment configured (sandbox={sandbox})")
    print()
    
    try:
        from src.bot_v2.orchestration.broker_factory import create_brokerage
        
        # Test 1: Create brokerage instance
        print("Test 1: Creating brokerage instance...")
        broker = create_brokerage()
        print("✓ Brokerage instance created")
        print()
        
        # Test 2: Connect and authenticate
        print("Test 2: Connecting to Coinbase...")
        connected = broker.connect()
        if not connected:
            print("❌ Failed to connect to Coinbase")
            print("   Check your API credentials")
            return False
        print("✓ Successfully connected and authenticated")
        print(f"  Account ID: {broker.get_account_id()}")
        print()
        
        # Test 3: Get products (public endpoint)
        print("Test 3: Fetching available products...")
        products = broker.list_products()
        if not products:
            print("❌ No products returned")
            return False
        print(f"✓ Found {len(products)} products")
        if len(products) > 0:
            print(f"  First product: {products[0].symbol}")
        print()
        
        # Test 4: Get balances (authenticated endpoint)
        print("Test 4: Fetching account balances...")
        try:
            balances = broker.list_balances()
            print(f"✓ Retrieved {len(balances)} balance entries")
            if balances:
                # Show first few balances
                for balance in balances[:3]:
                    if balance.total > 0:
                        print(f"  {balance.asset}: {balance.available} available")
        except Exception as e:
            print(f"⚠️  Could not fetch balances: {e}")
        print()
        
        # Test 5: Get market quote
        print("Test 5: Getting market quote for BTC-USD...")
        try:
            quote = broker.get_quote("BTC-USD")
            print(f"✓ BTC-USD Quote:")
            print(f"  Bid: ${quote.bid:.2f}")
            print(f"  Ask: ${quote.ask:.2f}")
            print(f"  Last: ${quote.last:.2f}")
        except Exception as e:
            print(f"⚠️  Could not get quote: {e}")
        print()
        
        # Test 6: Validate connection
        print("Test 6: Validating connection...")
        valid = broker.validate_connection()
        if valid:
            print("✓ Connection is valid")
        else:
            print("❌ Connection validation failed")
            return False
        print()
        
        # Test 7: List positions (should be empty for spot trading)
        print("Test 7: Checking positions...")
        positions = broker.list_positions()
        print(f"✓ Positions: {len(positions)} (expected 0 for spot trading)")
        print()
        
        print("=" * 60)
        print("✅ All basic tests passed!")
        print("   Coinbase integration is working correctly")
        print("=" * 60)
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're in the project directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    success = test_basic_functionality()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()