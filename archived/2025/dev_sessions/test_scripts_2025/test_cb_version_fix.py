#!/usr/bin/env python3
"""
Test CB-VERSION Header Fix for Coinbase CDP Authentication
This script tests that the CB-VERSION header is properly included in all requests.
"""

import os
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load production environment manually
env_file = Path(__file__).parent.parent / '.env.production'
if not env_file.exists():
    env_file = Path(__file__).parent.parent / '.env'

if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"')
                if key == 'COINBASE_CDP_PRIVATE_KEY':
                    private_key_lines = [value] if value else []
                    for next_line in f:
                        next_line = next_line.strip()
                        private_key_lines.append(next_line)
                        if 'END EC PRIVATE KEY' in next_line:
                            break
                    value = '\n'.join(private_key_lines)
                os.environ[key] = value

from src.bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from src.bot_v2.features.brokerages.coinbase.models import APIConfig
from src.bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from src.bot_v2.features.brokerages.coinbase.cdp_auth_v2 import create_cdp_auth_v2

def test_api_versions():
    """Test different API versions with CB-VERSION header."""
    
    print("=" * 80)
    print("CB-VERSION HEADER FIX TEST")
    print("=" * 80)
    
    # API versions to test
    versions_to_test = [
        "2024-10-24",  # Latest stable
        "2023-06-01",  # Widely compatible
        "2022-01-06",  # Legacy version
    ]
    
    # Get CDP credentials
    cdp_key = os.getenv('COINBASE_CDP_API_KEY')
    cdp_private = os.getenv('COINBASE_CDP_PRIVATE_KEY')
    
    if not cdp_key or not cdp_private:
        print("❌ CDP credentials not found in environment")
        print("   Please set COINBASE_CDP_API_KEY and COINBASE_CDP_PRIVATE_KEY")
        return
    
    print(f"\n📋 CDP Key: {cdp_key[:50]}...")
    
    # Create auth
    auth = create_cdp_auth_v2(
        api_key_name=cdp_key,
        private_key_pem=cdp_private
    )
    
    base_url = os.getenv('COINBASE_API_BASE', 'https://api.coinbase.com')
    
    # Test each version
    for version in versions_to_test:
        print(f"\n" + "=" * 60)
        print(f"Testing API Version: {version}")
        print("=" * 60)
        
        # Create client with specific version
        client = CoinbaseClient(
            base_url=base_url,
            auth=auth,
            api_version=version
        )
        
        # Test 1: Time endpoint (should always work)
        print("\n1. Testing /api/v3/brokerage/time...")
        try:
            result = client.get_time()
            if result and 'iso' in result:
                print(f"   ✅ Time endpoint works: {result['iso']}")
            else:
                print(f"   ⚠️  Time response: {result}")
        except Exception as e:
            print(f"   ❌ Time failed: {e}")
        
        # Test 2: Accounts endpoint (main test)
        print("\n2. Testing /api/v3/brokerage/accounts...")
        try:
            result = client.get_accounts()
            if result:
                if 'accounts' in result:
                    print(f"   ✅ SUCCESS! Retrieved {len(result['accounts'])} accounts")
                    print(f"   🎉 CB-VERSION {version} WORKS!")
                    
                    # Show first account
                    if result['accounts']:
                        acc = result['accounts'][0]
                        print(f"\n   First account:")
                        print(f"   - Name: {acc.get('name')}")
                        print(f"   - Currency: {acc.get('currency')}")
                        print(f"   - UUID: {acc.get('uuid', 'N/A')[:20]}...")
                    
                    # This version works, return success
                    print(f"\n" + "=" * 60)
                    print(f"🎯 SOLUTION FOUND: Use CB-VERSION = {version}")
                    print("=" * 60)
                    
                    # Test 3: Products (market data)
                    print("\n3. Testing /api/v3/brokerage/products...")
                    try:
                        products = client.get_products()
                        if products and 'products' in products:
                            print(f"   ✅ Retrieved {len(products['products'])} products")
                    except Exception as e:
                        print(f"   ❌ Products failed: {e}")
                    
                    return version  # Return the working version
                    
                elif 'error' in result:
                    print(f"   ❌ Error: {result['error']}")
                    if 'message' in result:
                        print(f"   Message: {result['message']}")
                else:
                    print(f"   ⚠️  Unexpected response: {json.dumps(result, indent=2)[:200]}")
            else:
                print("   ❌ No response")
                
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg:
                print(f"   ❌ Authentication failed (401)")
            elif "403" in error_msg:
                print(f"   ❌ Permission denied (403)")
            else:
                print(f"   ❌ Error: {error_msg[:100]}")
    
    print("\n" + "=" * 60)
    print("❌ No working API version found")
    print("Possible issues:")
    print("1. CDP key may not be properly provisioned")
    print("2. Account may not have Advanced Trade API access")
    print("3. Contact Coinbase support for CDP key activation")
    print("=" * 60)
    
    return None

def test_adapter_integration():
    """Test the full adapter integration with CB-VERSION."""
    
    print("\n" + "=" * 80)
    print("TESTING ADAPTER INTEGRATION")
    print("=" * 80)
    
    # Use environment variable or default
    api_version = os.getenv('COINBASE_API_VERSION', '2024-10-24')
    
    config = APIConfig(
        api_key="",
        api_secret="",
        passphrase=None,
        base_url=os.getenv('COINBASE_API_BASE', 'https://api.coinbase.com'),
        sandbox=False,
        ws_url=os.getenv('COINBASE_WS_URL', 'wss://advanced-trade-ws.coinbase.com'),
        cdp_api_key=os.getenv('COINBASE_CDP_API_KEY'),
        cdp_private_key=os.getenv('COINBASE_CDP_PRIVATE_KEY'),
        api_version=api_version
    )
    
    print(f"\nUsing API Version: {api_version}")
    
    # Create brokerage
    broker = CoinbaseBrokerage(config)
    
    # Try to connect
    print("\nConnecting to Coinbase...")
    if broker.connect():
        print("✅ Connection successful!")
        
        # Get account
        account_id = broker.get_account_id()
        print(f"   Account ID: {account_id}")
        
        # List balances
        print("\nGetting balances...")
        try:
            balances = broker.list_balances()
            if balances:
                print(f"✅ Retrieved {len(balances)} balances")
                for bal in balances[:5]:
                    if bal.total > 0:
                        print(f"   {bal.asset}: {bal.total}")
        except Exception as e:
            print(f"❌ Failed to get balances: {e}")
        
        # Get quote
        print("\nGetting BTC-USD quote...")
        try:
            quote = broker.get_quote("BTC-USD")
            if quote:
                print(f"✅ BTC-USD: Bid=${quote.bid}, Ask=${quote.ask}")
        except Exception as e:
            print(f"❌ Failed to get quote: {e}")
        
        broker.disconnect()
    else:
        print("❌ Connection failed")
        print("\nTrying different API versions...")
        working_version = test_api_versions()
        
        if working_version:
            print(f"\n✅ Update your .env file:")
            print(f"   COINBASE_API_VERSION={working_version}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CB-VERSION header fix")
    parser.add_argument('--test-versions', action='store_true', 
                       help='Test different API versions')
    parser.add_argument('--version', type=str, 
                       help='Test specific API version')
    
    args = parser.parse_args()
    
    if args.version:
        # Test specific version
        os.environ['COINBASE_API_VERSION'] = args.version
        print(f"Testing with CB-VERSION = {args.version}")
        test_adapter_integration()
    elif args.test_versions:
        # Test multiple versions
        test_api_versions()
    else:
        # Default: test adapter with current config
        test_adapter_integration()