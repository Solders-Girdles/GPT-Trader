#!/usr/bin/env python3
"""
Comprehensive CDP authentication test - checks multiple scenarios.
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_endpoint(auth, method, path, description):
    """Test a specific endpoint."""
    print(f"\n📍 Testing: {description}")
    print(f"   Method: {method}")
    print(f"   Path: {path}")
    
    try:
        jwt_token = auth.generate_jwt(method, path)
        url = f"https://api.coinbase.com{path}"
        
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
            "CB-VERSION": "2024-10-24"
        }
        
        req = urllib.request.Request(url, headers=headers, method=method)
        
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            data = response.read().decode('utf-8')
            
            print(f"   ✅ Status: {status}")
            
            # Parse and show summary
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    keys = list(parsed.keys())[:5]
                    print(f"   Response keys: {keys}")
            except:
                print(f"   Response: {data[:100]}")
            
            return True
            
    except urllib.error.HTTPError as e:
        status = e.code
        error_content = e.read()
        error_body = error_content.decode('utf-8') if error_content else ""
        
        print(f"   ❌ Status: {status}")
        print(f"   Error: {error_body[:200]}")
        
        return False
        
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def main():
    """Run comprehensive tests."""
    
    load_dotenv()
    
    print("=" * 70)
    print("COMPREHENSIVE CDP AUTHENTICATION TEST")
    print("=" * 70)
    
    # Check configuration
    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not api_key or not private_key:
        print("❌ Missing credentials")
        return 1
    
    print(f"\n📋 Configuration:")
    print(f"   API Key: {api_key[:50]}...")
    
    # Import auth
    from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
    
    auth = CDPAuthV2(
        api_key_name=api_key,
        private_key_pem=private_key,
        base_host="api.coinbase.com"
    )
    
    # Test multiple endpoints
    print("\n" + "=" * 70)
    print("TESTING VARIOUS ENDPOINTS")
    print("=" * 70)
    
    results = []
    
    # Test different endpoints to understand permission scope
    test_cases = [
        # Public endpoints (should work without auth issues)
        ("GET", "/api/v3/brokerage/time", "Server Time (public)"),
        
        # Account endpoints (require authentication)
        ("GET", "/api/v3/brokerage/accounts", "List Accounts"),
        ("GET", "/api/v3/brokerage/portfolios", "List Portfolios"),
        
        # Market data (some public, some require auth)
        ("GET", "/api/v3/brokerage/market/products", "List Products"),
        ("GET", "/api/v3/brokerage/market/products/BTC-USD", "Get BTC-USD Product"),
        
        # Best bid/ask (requires auth)
        ("GET", "/api/v3/brokerage/best_bid_ask?product_ids=BTC-USD", "Best Bid/Ask for BTC-USD"),
        
        # INTX/Derivatives endpoints
        ("GET", "/api/v3/brokerage/intx/portfolio/default", "INTX Portfolio (derivatives)"),
        
        # Transaction summary (basic account info)
        ("GET", "/api/v3/brokerage/transaction_summary", "Transaction Summary"),
    ]
    
    for method, path, description in test_cases:
        success = test_endpoint(auth, method, path, description)
        results.append((description, success))
        time.sleep(0.5)  # Rate limiting
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    working = []
    failing = []
    
    for desc, success in results:
        if success:
            working.append(desc)
            print(f"✅ {desc}")
        else:
            failing.append(desc)
            print(f"❌ {desc}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if not working and failing:
        print("🔴 All endpoints failed - likely an authentication issue:")
        print("")
        print("Common causes:")
        print("1. IP not whitelisted - add your IP in CDP console")
        print(f"   Your IP: {get_ip()}")
        print("")
        print("2. API key lacks permissions - check CDP console for:")
        print("   - 'view' permission for account data")
        print("   - 'trade' permission for trading")
        print("   - Proper portfolio/account access")
        print("")
        print("3. Wrong API key type - ensure using CDP key, not Exchange API")
        print("")
        print("4. Key/Private key mismatch - regenerate if needed")
        
    elif working and not failing:
        print("🟢 All endpoints working! Authentication is correct.")
        
    else:
        print("🟡 Mixed results - partial permissions:")
        print("")
        if working:
            print("Working endpoints suggest:")
            print("- Authentication mechanism is correct")
            print("- JWT generation is working")
            print("")
        if failing:
            print("Failed endpoints suggest:")
            print("- Limited permissions on API key")
            print("- Check CDP console for required scopes")
    
    return 0 if working else 1


def get_ip():
    """Get current IP address."""
    try:
        import urllib.request
        with urllib.request.urlopen('https://api.ipify.org', timeout=5) as response:
            return response.read().decode('utf-8')
    except:
        return "Unknown"


if __name__ == "__main__":
    sys.exit(main())