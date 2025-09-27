#!/usr/bin/env python3
"""
Get detailed 401 error information from Coinbase API.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_with_details():
    """Test and capture detailed error response."""
    
    # Clear any env vars and load from .env
    for key in ['COINBASE_PROD_CDP_API_KEY', 'COINBASE_PROD_CDP_PRIVATE_KEY', 
                'COINBASE_CDP_API_KEY', 'COINBASE_CDP_PRIVATE_KEY']:
        if key in os.environ:
            del os.environ[key]
    
    load_dotenv()
    
    print("=" * 70)
    print("DETAILED 401 ERROR ANALYSIS")
    print("=" * 70)
    
    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not api_key or not private_key:
        print("❌ Missing credentials")
        return
    
    # Verify we're using the new key
    key_id = api_key.split('/')[-1]
    print(f"\n📋 Using API Key ID: {key_id}")
    if key_id == "68c0297a-d2e1-428f-bcfa-e6496ca33e8e":
        print("   ✅ This is your NEW key with all permissions")
    else:
        print("   ❌ This is the OLD key - env vars may be cached")
        return
    
    from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
    
    auth = CDPAuthV2(
        api_key_name=api_key,
        private_key_pem=private_key,
        base_host="api.coinbase.com"
    )
    
    # Test accounts endpoint with detailed error capture
    print("\n" + "=" * 70)
    print("TESTING: GET /api/v3/brokerage/accounts")
    print("=" * 70)
    
    method = "GET"
    path = "/api/v3/brokerage/accounts"
    
    try:
        jwt_token = auth.generate_jwt(method, path)
        print(f"\n✅ JWT generated successfully")
        
        # Decode to verify claims
        import jwt
        decoded = jwt.decode(jwt_token, options={"verify_signature": False})
        print(f"\nJWT Claims:")
        print(f"  sub: {decoded.get('sub', 'N/A')}")
        print(f"  iss: {decoded.get('iss', 'N/A')}")
        print(f"  uri: {decoded.get('uri', 'N/A')}")
        
    except Exception as e:
        print(f"❌ JWT generation failed: {e}")
        return
    
    # Make request with detailed error capture
    url = f"https://api.coinbase.com{path}"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
        "CB-VERSION": "2024-10-24"
    }
    
    print(f"\n🌐 Making request to: {url}")
    
    req = urllib.request.Request(url, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            data = response.read().decode('utf-8')
            
            print(f"\n✅ SUCCESS! Status: {status}")
            
            try:
                parsed = json.loads(data)
                if "accounts" in parsed:
                    print(f"\nFound {len(parsed['accounts'])} accounts:")
                    for acc in parsed['accounts'][:3]:
                        print(f"  - {acc.get('name', 'Unknown')} ({acc.get('currency', 'N/A')})")
                        print(f"    UUID: {acc.get('uuid', 'N/A')}")
            except:
                print(f"Response: {data[:200]}")
                
    except urllib.error.HTTPError as e:
        status = e.code
        
        # Read all headers
        print(f"\n❌ Request failed with status: {status}")
        print(f"\n📋 Response Headers:")
        for header, value in e.headers.items():
            if header.lower() in ['x-request-id', 'trace-id', 'cf-ray', 'date']:
                print(f"  {header}: {value}")
        
        # Get error body
        error_content = e.read()
        error_body = error_content.decode('utf-8') if error_content else ""
        
        print(f"\n📋 Error Response Body:")
        print(f"  {error_body}")
        
        # Try to parse JSON error
        try:
            error_json = json.loads(error_body)
            print(f"\n📋 Parsed Error:")
            for key, value in error_json.items():
                print(f"  {key}: {value}")
        except:
            pass
        
        print("\n" + "=" * 70)
        print("TROUBLESHOOTING")
        print("=" * 70)
        
        if status == 401:
            print("\n🔍 401 Unauthorized - Possible causes:\n")
            
            print("1. Portfolio Selection Issue:")
            print("   - In CDP console, click on the portfolio dropdown")
            print("   - Actually SELECT 'Default' (click it, don't just leave it)")
            print("   - Make sure it shows as selected/highlighted")
            print("   - Click Save at the bottom")
            print("")
            print("2. Permission Propagation:")
            print("   - Changes can take 1-2 minutes to propagate")
            print("   - Try again in a moment")
            print("")
            print("3. Account Type Mismatch:")
            print("   - Ensure this is a regular Coinbase account")
            print("   - Not Coinbase One or special account type")
            print("")
            print("4. API Key Restrictions:")
            print("   - Check if there are any account-level restrictions")
            print("   - Some accounts require additional verification")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    test_with_details()