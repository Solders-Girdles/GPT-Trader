#!/usr/bin/env python3
"""
Test authentication using the official Coinbase Advanced Trade SDK.
This will help determine if the issue is with our implementation or CDP configuration.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Clear any environment variables first
for key in ['COINBASE_PROD_CDP_API_KEY', 'COINBASE_PROD_CDP_PRIVATE_KEY', 
            'COINBASE_CDP_API_KEY', 'COINBASE_CDP_PRIVATE_KEY']:
    if key in os.environ:
        del os.environ[key]

# Load from .env
load_dotenv()

def test_with_official_sdk():
    """Test using Coinbase's official SDK."""
    
    print("=" * 70)
    print("TESTING WITH OFFICIAL COINBASE SDK")
    print("=" * 70)
    
    # Get credentials
    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not api_key or not private_key:
        print("❌ Missing credentials in .env")
        return False
    
    # Verify we're using the new key
    key_id = api_key.split('/')[-1]
    print(f"\n📋 Using API Key ID: {key_id}")
    if key_id == "68c0297a-d2e1-428f-bcfa-e6496ca33e8e":
        print("   ✅ This is your NEW key with all permissions")
    else:
        print("   ❌ Wrong key loaded")
        return False
    
    print("\n" + "=" * 70)
    print("INITIALIZING OFFICIAL SDK")
    print("=" * 70)
    
    try:
        from coinbase.rest import RESTClient
        
        # Initialize the official client
        client = RESTClient(
            api_key=api_key,
            api_secret=private_key,
            verbose=True  # Enable verbose output for debugging
        )
        
        print("✅ SDK client initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize SDK: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("TESTING ENDPOINTS WITH OFFICIAL SDK")
    print("=" * 70)
    
    # Test 1: Get accounts (requires View permission)
    print("\n📍 Test 1: List Accounts")
    try:
        accounts = client.get_accounts()
        if accounts:
            print(f"   ✅ SUCCESS! Found {len(accounts.accounts)} accounts")
            for acc in accounts.accounts[:3]:
                print(f"      - {acc.name} ({acc.currency}): {acc.available_balance.value} {acc.available_balance.currency}")
        else:
            print("   ⚠️ No accounts returned")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("      → This confirms the portfolio access issue")
    
    # Test 2: Get portfolios
    print("\n📍 Test 2: List Portfolios")
    try:
        portfolios = client.get_portfolios()
        if portfolios:
            print(f"   ✅ SUCCESS! Found portfolios")
            for portfolio in portfolios.portfolios[:3]:
                print(f"      - {portfolio.name} (UUID: {portfolio.uuid})")
        else:
            print("   ⚠️ No portfolios returned")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: Get products (public endpoint)
    print("\n📍 Test 3: Get Products (Public)")
    try:
        products = client.get_products()
        if products:
            print(f"   ✅ Found {len(products.products)} products")
            # Show a few
            for product in products.products[:3]:
                print(f"      - {product.product_id}: ${product.price}")
        else:
            print("   ⚠️ No products returned")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: Get specific product (public)
    print("\n📍 Test 4: Get BTC-USD Product (Public)")
    try:
        btc = client.get_product("BTC-USD")
        if btc:
            print(f"   ✅ BTC-USD Price: ${btc.price}")
            print(f"      24h Volume: {btc.volume_24h}")
        else:
            print("   ⚠️ No data returned")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: Server time (public)
    print("\n📍 Test 5: Server Time (Public)")
    try:
        time_response = client.get_unix_time()
        if time_response:
            print(f"   ✅ Server time: {time_response.iso}")
        else:
            print("   ⚠️ No time returned")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print("\nBased on the official SDK results:")
    print("\n1. If public endpoints work but account endpoints fail:")
    print("   → Your CDP portfolio access is not configured properly")
    print("   → Go to CDP console and explicitly select your portfolio")
    print("")
    print("2. If nothing works with the SDK:")
    print("   → There may be an issue with the credentials themselves")
    print("   → Try regenerating the API key")
    print("")
    print("3. If everything works with the SDK:")
    print("   → Our JWT implementation has an issue")
    print("   → We'll need to debug our code further")
    
    return True


def compare_implementations():
    """Compare our implementation with the SDK."""
    
    print("\n" + "=" * 70)
    print("COMPARING IMPLEMENTATIONS")
    print("=" * 70)
    
    print("\n📋 Key Differences:")
    print("1. SDK uses their official JWT generation")
    print("2. SDK may handle edge cases we're missing")
    print("3. SDK has been tested with CDP's backend")
    
    print("\n📋 Next Steps:")
    print("1. If SDK works: We'll align our code with SDK behavior")
    print("2. If SDK fails: Focus on CDP console configuration")
    print("3. Contact Coinbase support if needed")


if __name__ == "__main__":
    success = test_with_official_sdk()
    if success:
        compare_implementations()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)