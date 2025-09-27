#!/usr/bin/env python3
"""
Test Coinbase Developer Platform (CDP) API with JWT authentication
"""

import json
import time
import jwt
import urllib.request
import urllib.error
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Your CDP credentials
API_KEY_NAME = "organizations/5184a9ea-2cec-4a66-b00e-7cf6daaf048e/apiKeys/7e24f68f-9e72-4d19-9418-86ee7d65bcb4"
PRIVATE_KEY_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIL1jiXKo+Hpgv0wTx/Bi1ux+Et2DzC76Q3BdE370aMz2oAoGCCqGSM49
AwEHoUQDQgAEMwCNcJ86vEPQimLLPf6IWcSVC0FWiTads4yiGA4qHazVptVAf0eq
8oteBwNt9Y4MQ/uqQn11/Y/VuCWY9rJLkQ==
-----END EC PRIVATE KEY-----"""

def generate_jwt(request_method, request_path):
    """Generate JWT token for CDP API authentication"""
    
    # Load the private key
    private_key = serialization.load_pem_private_key(
        PRIVATE_KEY_PEM.encode(),
        password=None,
        backend=default_backend()
    )
    
    # Create JWT claims
    current_time = int(time.time())
    claims = {
        "sub": API_KEY_NAME,
        "iss": "coinbase-cloud",
        "nbf": current_time,
        "exp": current_time + 120,  # Token expires in 2 minutes
        "aud": ["retail_rest_api_proxy"],
        "uri": f"{request_method} {request_path}"
    }
    
    # Generate JWT token
    token = jwt.encode(
        claims,
        private_key,
        algorithm="ES256",
        headers={"kid": API_KEY_NAME, "nonce": str(int(time.time() * 1000))}
    )
    
    return token

def make_request(method, path, base_url="https://api.coinbase.com"):
    """Make an authenticated request to CDP API"""
    
    # Generate JWT
    jwt_token = generate_jwt(method, path)
    
    # Prepare request
    url = f"{base_url}{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {jwt_token}")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read().decode()
            return json.loads(data) if data else {}
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if hasattr(e, 'read') else ''
        print(f"HTTP {e.code}: {e.reason}")
        if error_body:
            try:
                error_json = json.loads(error_body)
                print(f"Error details: {error_json}")
            except:
                print(f"Error body: {error_body[:200]}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

print("=" * 60)
print("Testing Coinbase Developer Platform (CDP) API")
print("=" * 60)
print(f"API Key: {API_KEY_NAME.split('/')[-1][:8]}...")
print()

# Test endpoints
test_cases = [
    ("List Accounts", "GET", "/api/v3/brokerage/accounts"),
    ("Get Products", "GET", "/api/v3/brokerage/products"),
    ("Get Server Time", "GET", "/api/v3/brokerage/time"),
]

for name, method, path in test_cases:
    print(f"\nTesting: {name}")
    print(f"  {method} {path}")
    
    result = make_request(method, path)
    
    if result:
        if isinstance(result, dict):
            # Handle different response types
            if 'accounts' in result:
                accounts = result['accounts']
                print(f"  ✅ Success! Found {len(accounts)} accounts")
                for acc in accounts[:3]:  # Show first 3
                    currency = acc.get('currency', 'Unknown')
                    balance = acc.get('available_balance', {}).get('value', '0')
                    print(f"     - {currency}: {balance}")
            elif 'products' in result:
                products = result['products']
                print(f"  ✅ Success! Found {len(products)} products")
                if products:
                    print(f"     Sample: {products[0].get('product_id', 'Unknown')}")
            elif 'iso' in result:
                print(f"  ✅ Success! Server time: {result['iso']}")
            else:
                keys = list(result.keys())[:5]
                print(f"  ✅ Success! Response keys: {keys}")
        else:
            print(f"  ✅ Success! Got response")
    else:
        print(f"  ❌ Request failed")

print("\n" + "=" * 60)
print("Test complete!")
print("\nNote: CDP API uses JWT authentication with EC private keys.")
print("This is different from the older Exchange API that uses HMAC.")