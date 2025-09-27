#!/usr/bin/env python3
"""
Test CDP authentication variations - different headers and claim formats.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# Manual env loading
def load_env_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        if '=' in line:
            key, value = line.split('=', 1)
            if value.startswith('"') and not value.endswith('"'):
                value_lines = [value[1:]]
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.endswith('"'):
                        value_lines.append(next_line[:-1])
                        value = '\n'.join(value_lines)
                        break
                    value_lines.append(next_line)
                    i += 1
            else:
                value = value.strip('"')
            os.environ[key] = value
        i += 1

load_env_file(Path(__file__).parent.parent / ".env")

# Import JWT libraries
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

print("Testing CDP Authentication Variations")
print("=" * 60)

cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

# Load private key
private_key = serialization.load_pem_private_key(
    cdp_private_key.encode() if isinstance(cdp_private_key, str) else cdp_private_key,
    password=None,
    backend=default_backend()
)

def test_jwt_variation(name, claims_modifier=None, headers_modifier=None):
    """Test a JWT variation."""
    print(f"\n{name}")
    print("-" * 40)
    
    method = "GET"
    path = "/api/v3/brokerage/accounts"
    current_time = int(time.time())
    
    # Base claims
    claims = {
        "sub": cdp_api_key,
        "iss": "coinbase-cloud",
        "nbf": current_time,
        "exp": current_time + 120,
        "aud": ["retail_rest_api_proxy"],
        "uri": f"{method} {path}"
    }
    
    # Base headers
    headers = {
        "alg": "ES256",
        "kid": cdp_api_key,
        "typ": "JWT",
        "nonce": str(int(time.time() * 1000))
    }
    
    # Apply modifications
    if claims_modifier:
        claims = claims_modifier(claims)
    if headers_modifier:
        headers = headers_modifier(headers)
    
    # Generate JWT
    token = jwt.encode(claims, private_key, algorithm="ES256", headers=headers)
    if isinstance(token, bytes):
        token = token.decode('utf-8')
    
    # Make request
    url = f"https://api.coinbase.com{path}"
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.getcode()
            data = response.read().decode()
            print(f"  ✅ SUCCESS! Status: {status}")
            try:
                result = json.loads(data)
                if 'accounts' in result:
                    print(f"     Found {len(result['accounts'])} accounts")
            except:
                pass
            return True
                    
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}")
        return False

# Test variations
test_jwt_variation("1. Standard CDP JWT (current implementation)")

test_jwt_variation(
    "2. With 'retail_rest_api_proxy' as string (not array)",
    claims_modifier=lambda c: {**c, "aud": "retail_rest_api_proxy"}
)

test_jwt_variation(
    "3. With additional 'scope' claim",
    claims_modifier=lambda c: {**c, "scope": "view trade"}
)

test_jwt_variation(
    "4. With 'iat' (issued at) claim",
    claims_modifier=lambda c: {**c, "iat": int(time.time())}
)

test_jwt_variation(
    "5. Without nonce in header",
    headers_modifier=lambda h: {k: v for k, v in h.items() if k != "nonce"}
)

test_jwt_variation(
    "6. With lowercase HTTP method in URI",
    claims_modifier=lambda c: {**c, "uri": "get /api/v3/brokerage/accounts"}
)

test_jwt_variation(
    "7. With full URL in URI claim",
    claims_modifier=lambda c: {**c, "uri": "GET https://api.coinbase.com/api/v3/brokerage/accounts"}
)

test_jwt_variation(
    "8. With 'jti' (JWT ID) claim",
    claims_modifier=lambda c: {**c, "jti": str(int(time.time() * 1000))}
)

test_jwt_variation(
    "9. With different audience",
    claims_modifier=lambda c: {**c, "aud": ["advanced_trade_api"]}
)

test_jwt_variation(
    "10. With portfolio_id in claims",
    claims_modifier=lambda c: {**c, "portfolio_id": "default"}
)

print("\n" + "=" * 60)
print("\nConclusions:")
print("If none of these variations work, the issue is definitively on Coinbase's side.")
print("The API key may need additional activation or configuration in their system.")