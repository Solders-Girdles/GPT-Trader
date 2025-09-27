#!/usr/bin/env python3
"""
Debug CDP API authentication
"""

import json
import time
import jwt
import urllib.request
import urllib.error
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import base64

# Your CDP credentials
API_KEY_NAME = "organizations/5184a9ea-2cec-4a66-b00e-7cf6daaf048e/apiKeys/7e24f68f-9e72-4d19-9418-86ee7d65bcb4"
PRIVATE_KEY_PEM = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIL1jiXKo+Hpgv0wTx/Bi1ux+Et2DzC76Q3BdE370aMz2oAoGCCqGSM49
AwEHoUQDQgAEMwCNcJ86vEPQimLLPf6IWcSVC0FWiTads4yiGA4qHazVptVAf0eq
8oteBwNt9Y4MQ/uqQn11/Y/VuCWY9rJLkQ==
-----END EC PRIVATE KEY-----"""

print("Debugging CDP Authentication")
print("=" * 60)

# Load and verify the private key
try:
    private_key = serialization.load_pem_private_key(
        PRIVATE_KEY_PEM.encode(),
        password=None,
        backend=default_backend()
    )
    print("✅ Private key loaded successfully")
    print(f"   Key type: {type(private_key).__name__}")
except Exception as e:
    print(f"❌ Failed to load private key: {e}")
    exit(1)

# Create a test JWT
current_time = int(time.time())
request_method = "GET"
request_path = "/api/v3/brokerage/accounts"

claims = {
    "sub": API_KEY_NAME,
    "iss": "coinbase-cloud",
    "nbf": current_time,
    "exp": current_time + 120,
    "aud": ["retail_rest_api_proxy"],
    "uri": f"{request_method} {request_path}"
}

print("\nJWT Claims:")
for key, value in claims.items():
    print(f"  {key}: {value}")

# Generate JWT
try:
    token = jwt.encode(
        claims,
        private_key,
        algorithm="ES256",
        headers={
            "kid": API_KEY_NAME,
            "typ": "JWT",
            "alg": "ES256",
            "nonce": str(int(time.time() * 1000))
        }
    )
    print("\n✅ JWT generated successfully")
    print(f"   Token length: {len(token)} chars")
    
    # Decode to verify structure
    header_b64 = token.split('.')[0]
    # Add padding if needed
    header_b64 += '=' * (4 - len(header_b64) % 4)
    header = json.loads(base64.urlsafe_b64decode(header_b64))
    print(f"\nJWT Header:")
    for key, value in header.items():
        if key == "kid":
            print(f"  {key}: ...{value[-20:]}")
        else:
            print(f"  {key}: {value}")
    
except Exception as e:
    print(f"❌ Failed to generate JWT: {e}")
    exit(1)

# Test the request
print("\n" + "=" * 60)
print("Testing authenticated request...")

url = f"https://api.coinbase.com{request_path}"
req = urllib.request.Request(url, method=request_method)
req.add_header("Authorization", f"Bearer {token}")
req.add_header("Content-Type", "application/json")
req.add_header("User-Agent", "GPT-Trader/1.0")

print(f"URL: {url}")
print(f"Method: {request_method}")
print("Headers:")
print(f"  Authorization: Bearer {token[:50]}...")
print(f"  Content-Type: application/json")

try:
    with urllib.request.urlopen(req, timeout=10) as response:
        data = response.read().decode()
        result = json.loads(data) if data else {}
        print(f"\n✅ Success! Status: {response.status}")
        print(f"Response: {json.dumps(result, indent=2)[:500]}")
except urllib.error.HTTPError as e:
    print(f"\n❌ HTTP {e.code}: {e.reason}")
    if hasattr(e, 'headers'):
        print("Response headers:")
        for key, value in e.headers.items():
            if key.lower() in ['x-request-id', 'cf-ray', 'date']:
                print(f"  {key}: {value}")
    if hasattr(e, 'read'):
        error_body = e.read().decode()
        print(f"Error body: {error_body[:200]}")
except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "=" * 60)
print("\nNotes:")
print("1. CDP API requires specific JWT structure")
print("2. The 'kid' in header should match API key name")
print("3. The 'uri' claim must match the request method and path")
print("4. Check if your API key has the right permissions")