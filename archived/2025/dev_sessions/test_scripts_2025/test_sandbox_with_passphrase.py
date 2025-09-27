#!/usr/bin/env python3
"""Test Coinbase sandbox with passphrase using legacy endpoints"""

import hmac
import hashlib
import time
import base64
import json
import urllib.request
import urllib.error
import os

# Get credentials from environment
API_KEY = os.getenv('COINBASE_API_KEY', 'b60e18a142dff4c9c53a9759caee2c8c')
API_SECRET = os.getenv('COINBASE_API_SECRET', 'oF4cWu2fdhiQwkbgF9g/Ak+moFMIDH9x6r1jmlTlsF1j0p0CknXvk+79f8vLTx6A915FRe+X0g81REl+3pXYhQ==')
API_PASSPHRASE = os.getenv('COINBASE_API_PASSPHRASE', 'cvnrvm3xbim1')

def sign_request(method, path, body=''):
    """Generate signature for Coinbase Exchange API"""
    timestamp = str(time.time())
    
    # Create message for signature
    message = timestamp + method.upper() + path + (body or '')
    message = message.encode('ascii')
    
    # Create HMAC signature
    hmac_key = base64.b64decode(API_SECRET)
    signature = hmac.new(hmac_key, message, hashlib.sha256)
    signature_b64 = base64.b64encode(signature.digest()).decode()
    
    return {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature_b64,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-PASSPHRASE': API_PASSPHRASE,
        'Content-Type': 'application/json',
        'User-Agent': 'GPT-Trader/1.0'
    }

print("Testing Coinbase Sandbox with Passphrase")
print("=" * 60)
print(f"API Key: {API_KEY[:8]}...")
print(f"Passphrase: {API_PASSPHRASE}")
print()

# Test with legacy Exchange API endpoints
base_url = "https://api-public.sandbox.exchange.coinbase.com"

# Test endpoints
test_cases = [
    ("Server Time", "/time", "GET", None, False),
    ("Products List", "/products", "GET", None, False),
    ("Accounts", "/accounts", "GET", None, True),
    ("Single Product", "/products/BTC-USD", "GET", None, False),
    ("Order Book", "/products/BTC-USD/book", "GET", None, False),
]

for name, path, method, body, needs_auth in test_cases:
    print(f"\nTesting: {name}")
    print(f"  Endpoint: {method} {path}")
    
    url = f"{base_url}{path}"
    
    try:
        req = urllib.request.Request(url, method=method)
        
        if needs_auth:
            print(f"  Auth: Yes (with passphrase)")
            headers = sign_request(method, path, body)
            for k, v in headers.items():
                if k != 'CB-ACCESS-SIGN':
                    req.add_header(k, v)
                else:
                    req.add_header(k, v)
        else:
            print(f"  Auth: No")
            req.add_header('Content-Type', 'application/json')
            req.add_header('User-Agent', 'GPT-Trader/1.0')
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read().decode()
            
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    print(f"  ✅ Success! Got {len(parsed)} items")
                    if parsed and 'id' in parsed[0]:
                        print(f"     Sample: {parsed[0]['id']}")
                elif isinstance(parsed, dict):
                    if 'epoch' in parsed or 'iso' in parsed:
                        print(f"  ✅ Success! Time: {parsed.get('iso', parsed)}")
                    elif 'bids' in parsed:
                        print(f"  ✅ Success! Order book with {len(parsed.get('bids', []))} bids")
                    else:
                        keys = list(parsed.keys())[:5]
                        print(f"  ✅ Success! Keys: {keys}")
                else:
                    print(f"  ✅ Success!")
                    
            except json.JSONDecodeError:
                print(f"  ✅ Success! (non-JSON response)")
                
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("     Authentication failed - check credentials")
        elif e.code == 403:
            print("     Forbidden - check permissions or URL")
        elif e.code == 400:
            print("     Bad request - check parameters")
            
        # Try to get error details
        try:
            error_body = e.read().decode()
            error_json = json.loads(error_body)
            print(f"     Message: {error_json.get('message', error_body[:100])}")
        except:
            pass
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("\nIf all tests failed with 403, the sandbox URL might have changed.")
print("If only authenticated endpoints failed, check your API key permissions.")