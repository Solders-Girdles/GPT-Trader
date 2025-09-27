#!/usr/bin/env python3
"""
Coinbase API Connection Tester

This script helps diagnose and fix Coinbase API connection issues.
"""

import os
import sys
import json
import base64
import hmac
import hashlib
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_environment():
    """Check environment variables are set."""
    print("=" * 60)
    print("1. CHECKING ENVIRONMENT VARIABLES")
    print("=" * 60)
    
    required_vars = {
        'BROKER': 'coinbase',
        'COINBASE_API_KEY': None,
        'COINBASE_API_SECRET': None,
    }
    
    optional_vars = {
        'COINBASE_SANDBOX': '1',
        'COINBASE_API_PASSPHRASE': '',
        'COINBASE_API_BASE': None,
    }
    
    all_good = True
    
    # Check required variables
    for var, expected in required_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"❌ {var}: NOT SET (required)")
            all_good = False
        elif expected and value != expected:
            print(f"⚠️  {var}: '{value}' (expected '{expected}')")
            all_good = False
        else:
            if 'SECRET' in var or 'KEY' in var:
                # Hide sensitive values
                display = value[:8] + "..." if len(value) > 8 else "***"
                print(f"✅ {var}: {display}")
            else:
                print(f"✅ {var}: {value}")
    
    # Check optional variables
    print("\nOptional settings:")
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        if 'PASSPHRASE' in var and value:
            display = "***" if value else "(empty)"
        else:
            display = value if value else "(not set)"
        print(f"   {var}: {display}")
    
    return all_good


def check_api_key_format():
    """Validate API key and secret format."""
    print("\n" + "=" * 60)
    print("2. VALIDATING CREDENTIAL FORMAT")
    print("=" * 60)
    
    api_key = os.getenv('COINBASE_API_KEY', '')
    api_secret = os.getenv('COINBASE_API_SECRET', '')
    
    issues = []
    
    # Check API key
    if not api_key:
        issues.append("API key is empty")
    elif len(api_key) < 20:
        issues.append(f"API key seems too short ({len(api_key)} chars)")
    elif ' ' in api_key or '\n' in api_key:
        issues.append("API key contains spaces or newlines")
    else:
        print(f"✅ API key format looks valid ({len(api_key)} chars)")
    
    # Check API secret
    if not api_secret:
        issues.append("API secret is empty")
    else:
        # Try to decode as base64
        try:
            decoded = base64.b64decode(api_secret, validate=True)
            if decoded:
                print(f"✅ API secret is valid base64 ({len(api_secret)} chars)")
            else:
                issues.append("API secret decoded to empty")
        except Exception as e:
            # Some keys might not be base64
            print(f"⚠️  API secret may not be base64 (this might be OK)")
            print(f"    Length: {len(api_secret)} chars")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    return True


def test_api_signature():
    """Test HMAC signature generation."""
    print("\n" + "=" * 60)
    print("3. TESTING SIGNATURE GENERATION")
    print("=" * 60)
    
    try:
        from src.bot_v2.features.brokerages.coinbase.client import CoinbaseAuth
        
        api_key = os.getenv('COINBASE_API_KEY', 'test')
        api_secret = os.getenv('COINBASE_API_SECRET', 'test')
        passphrase = os.getenv('COINBASE_API_PASSPHRASE')
        
        auth = CoinbaseAuth(
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase
        )
        
        # Test signature generation
        headers = auth.sign("GET", "/api/v3/brokerage/accounts", None)
        
        required_headers = ['CB-ACCESS-KEY', 'CB-ACCESS-SIGN', 'CB-ACCESS-TIMESTAMP']
        for header in required_headers:
            if header in headers:
                if header == 'CB-ACCESS-SIGN':
                    print(f"✅ {header}: {headers[header][:20]}...")
                else:
                    print(f"✅ {header}: {headers[header]}")
            else:
                print(f"❌ {header}: Missing")
                return False
        
        if passphrase and 'CB-ACCESS-PASSPHRASE' in headers:
            print(f"✅ CB-ACCESS-PASSPHRASE: ***")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate signature: {e}")
        return False


def test_connection():
    """Test actual API connection."""
    print("\n" + "=" * 60)
    print("4. TESTING API CONNECTION")
    print("=" * 60)
    
    sandbox = os.getenv('COINBASE_SANDBOX', '0') == '1'
    env_name = "SANDBOX" if sandbox else "PRODUCTION"
    print(f"Connecting to: {env_name}")
    
    try:
        from src.bot_v2.orchestration.broker_factory import create_brokerage
        
        # Create brokerage instance
        print("\nCreating brokerage instance...")
        broker = create_brokerage()
        
        # Test connection
        print("Testing authentication...")
        connected = broker.connect()
        
        if connected:
            print(f"✅ Successfully connected!")
            print(f"   Account ID: {broker.get_account_id()}")
            
            # Try to get some data
            print("\nFetching account data...")
            try:
                balances = broker.list_balances()
                print(f"✅ Retrieved {len(balances)} balance entries")
                
                # Show non-zero balances
                non_zero = [b for b in balances if b.total > 0]
                if non_zero:
                    print("\n   Non-zero balances:")
                    for b in non_zero[:5]:  # Show first 5
                        print(f"   - {b.asset}: {b.total}")
            except Exception as e:
                print(f"⚠️  Could not fetch balances: {e}")
            
            # Test public endpoint
            print("\nFetching available products...")
            try:
                products = broker.list_products()
                print(f"✅ Found {len(products)} products")
                if products:
                    print(f"   Sample products: {', '.join(p.symbol for p in products[:5])}")
            except Exception as e:
                print(f"⚠️  Could not fetch products: {e}")
            
            return True
        else:
            print("❌ Connection failed")
            print("\nPossible causes:")
            print("1. Wrong environment (sandbox vs production)")
            print("2. Invalid API credentials")
            print("3. API key doesn't have required permissions")
            print("4. Network connectivity issues")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        
        # Provide specific guidance based on error
        error_str = str(e).lower()
        if "permission" in error_str or "forbidden" in error_str:
            print("\n💡 This looks like a permission issue:")
            print("   - Check your API key has 'View' permission")
            print("   - Make sure you're using the right environment")
        elif "authentication" in error_str or "unauthorized" in error_str:
            print("\n💡 This looks like an authentication issue:")
            print("   - Verify your API key and secret are correct")
            print("   - Check if you need a passphrase")
            print("   - Ensure no extra spaces in credentials")
        elif "not found" in error_str:
            print("\n💡 This might be an endpoint issue:")
            print("   - Make sure you're using Advanced Trade API keys")
            print("   - Not the legacy Exchange API keys")
        
        return False


def test_public_endpoint():
    """Test a public endpoint that doesn't require auth."""
    print("\n" + "=" * 60)
    print("5. TESTING PUBLIC ENDPOINT (No Auth)")
    print("=" * 60)
    
    import urllib.request
    import urllib.error
    
    sandbox = os.getenv('COINBASE_SANDBOX', '0') == '1'
    
    # Pick path based on API family
    base_url = os.getenv('COINBASE_API_BASE')
    if not base_url:
        base_url = 'https://api-public.sandbox.exchange.coinbase.com' if sandbox else 'https://api.coinbase.com'

    if 'api.coinbase.com' in base_url:
        test_url = f"{base_url}/api/v3/brokerage/products"
    else:
        test_url = f"{base_url}/products"
    
    print(f"Testing: {test_url}")
    
    try:
        req = urllib.request.Request(test_url)
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            products = data.get('products', [])
            print(f"✅ Public endpoint works! Found {len(products)} products")
            
            if products:
                print("   Sample products:")
                for p in products[:3]:
                    print(f"   - {p.get('product_id', 'unknown')}")
            return True
            
    except urllib.error.HTTPError as e:
        print(f"❌ HTTP Error {e.code}: {e.reason}")
        if e.code == 404:
            print("   The API endpoint was not found - check your URLs")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def print_summary(results):
    """Print summary and recommendations."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✅ All checks passed! Your Coinbase API is configured correctly.")
        print("\nNext steps:")
        print("1. Run the smoke test: python scripts/test_coinbase_basic.py")
        print("2. Try paper trading in sandbox mode")
        print("3. Monitor for any errors")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nFailed checks:")
        for check, passed in results.items():
            if not passed:
                print(f"  - {check}")
        
        print("\nRecommendations:")
        if not results.get('environment'):
            print("1. Set up your environment variables in .env file")
            print("   See docs/COINBASE_API_SETUP.md for details")
        
        if not results.get('format'):
            print("2. Check your API credentials format")
            print("   - API key should be alphanumeric")
            print("   - API secret should be base64 (for Advanced Trade)")
        
        if not results.get('connection'):
            print("3. Verify your API key permissions and environment")
            print("   - Use sandbox for testing")
            print("   - Ensure 'View' permission is enabled")


def main():
    """Run all connection tests."""
    print("\n" + "🔧 COINBASE API CONNECTION TESTER " + "🔧")
    print("This tool will help diagnose connection issues\n")
    
    results = {}
    
    # Run tests
    results['environment'] = check_environment()
    results['format'] = check_api_key_format()
    results['signature'] = test_api_signature()
    results['public'] = test_public_endpoint()
    results['connection'] = test_connection()
    
    # Print summary
    print_summary(results)
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
