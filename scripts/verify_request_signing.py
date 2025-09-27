#!/usr/bin/env python3
"""
Verify request signing components match Coinbase's exact requirements.
Tests different path formats and timestamp handling.
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path
from dotenv import load_dotenv

# Clear env vars and reload
for key in ['COINBASE_PROD_CDP_API_KEY', 'COINBASE_PROD_CDP_PRIVATE_KEY',
            'COINBASE_CDP_API_KEY', 'COINBASE_CDP_PRIVATE_KEY']:
    if key in os.environ:
        del os.environ[key]

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_signing_variations():
    """Test different signing variations to find what works."""
    
    print("=" * 70)
    print("REQUEST SIGNING VERIFICATION")
    print("=" * 70)
    
    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not api_key or not private_key:
        print("❌ Missing credentials")
        return
    
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    import secrets
    
    # Load private key
    key_obj = serialization.load_pem_private_key(
        private_key.encode() if isinstance(private_key, str) else private_key,
        password=None,
        backend=default_backend()
    )
    
    # Test variations
    test_cases = [
        {
            "name": "Current Implementation",
            "method": "GET",
            "host": "api.coinbase.com",
            "path": "/api/v3/brokerage/accounts",
            "uri_format": "GET api.coinbase.com/api/v3/brokerage/accounts",
            "description": "What we currently use"
        },
        {
            "name": "Without Host in URI",
            "method": "GET",
            "host": "api.coinbase.com",
            "path": "/api/v3/brokerage/accounts",
            "uri_format": "GET /api/v3/brokerage/accounts",
            "description": "Just method and path, no host"
        },
        {
            "name": "With HTTPS in URI",
            "method": "GET",
            "host": "api.coinbase.com",
            "path": "/api/v3/brokerage/accounts",
            "uri_format": "GET https://api.coinbase.com/api/v3/brokerage/accounts",
            "description": "Full URL with scheme"
        },
        {
            "name": "Path Only in URI",
            "method": "GET",
            "host": "api.coinbase.com",
            "path": "/api/v3/brokerage/accounts",
            "uri_format": "/api/v3/brokerage/accounts",
            "description": "Just the path, no method"
        },
        {
            "name": "Test Public Endpoint",
            "method": "GET",
            "host": "api.coinbase.com",
            "path": "/api/v3/brokerage/time",
            "uri_format": "GET api.coinbase.com/api/v3/brokerage/time",
            "description": "Public endpoint that should work"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n" + "=" * 70)
        print(f"TEST {i}: {test['name']}")
        print(f"Description: {test['description']}")
        print("=" * 70)
        
        current_time = int(time.time())
        
        # Build JWT claims
        claims = {
            "sub": api_key,
            "iss": "cdp",
            "nbf": current_time,
            "exp": current_time + 120,
            "uri": test["uri_format"]
        }
        
        # Headers
        headers = {
            "kid": api_key,
            "nonce": secrets.token_hex(),
            "typ": "JWT"
        }
        
        print(f"\n📋 JWT Details:")
        print(f"   URI Claim: {test['uri_format']}")
        print(f"   Timestamp (nbf): {current_time}")
        print(f"   Expires (exp): {current_time + 120}")
        
        # Generate token
        try:
            token = jwt.encode(claims, key_obj, algorithm="ES256", headers=headers)
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            
            print(f"   ✅ JWT generated successfully")
            print(f"   Token length: {len(token)}")
            
        except Exception as e:
            print(f"   ❌ JWT generation failed: {e}")
            continue
        
        # Test the token
        url = f"https://{test['host']}{test['path']}"
        print(f"\n🌐 Testing request:")
        print(f"   URL: {url}")
        print(f"   Method: {test['method']}")
        
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "CB-VERSION": "2024-10-24"
            },
            method=test['method']
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.getcode()
                data = response.read().decode('utf-8')
                
                print(f"\n   ✅ SUCCESS! Status: {status}")
                
                # Parse response
                try:
                    parsed = json.loads(data)
                    if "accounts" in parsed:
                        print(f"   Found {len(parsed['accounts'])} accounts")
                    elif "iso" in parsed:
                        print(f"   Server time: {parsed['iso']}")
                    else:
                        print(f"   Response keys: {list(parsed.keys())[:5]}")
                except:
                    print(f"   Response: {data[:100]}")
                
                print(f"\n   🎯 THIS SIGNING FORMAT WORKS!")
                print(f"   Use URI format: {test['uri_format']}")
                
        except urllib.error.HTTPError as e:
            status = e.code
            error_body = e.read().decode('utf-8') if e.read() else ""
            
            print(f"\n   ❌ Failed with status: {status}")
            if error_body:
                print(f"   Error: {error_body[:100]}")
            
            if status == 401:
                print(f"   → This URI format doesn't work")
            elif status == 400:
                print(f"   → Bad request - might be malformed")
        
        except Exception as e:
            print(f"\n   ❌ Request error: {e}")
        
        # Small delay between tests
        time.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("TIMESTAMP VERIFICATION")
    print("=" * 70)
    
    # Check if timestamp might be an issue
    print("\n📍 Testing with different timestamp offsets...")
    
    for offset in [0, -30, -60, 30, 60]:
        adjusted_time = int(time.time()) + offset
        
        print(f"\n   Offset: {offset:+d} seconds")
        print(f"   Timestamp: {adjusted_time}")
        
        claims = {
            "sub": api_key,
            "iss": "cdp",
            "nbf": adjusted_time,
            "exp": adjusted_time + 120,
            "uri": "GET api.coinbase.com/api/v3/brokerage/time"  # Use public endpoint
        }
        
        headers = {
            "kid": api_key,
            "nonce": secrets.token_hex(),
        }
        
        try:
            token = jwt.encode(claims, key_obj, algorithm="ES256", headers=headers)
            if isinstance(token, bytes):
                token = token.decode('utf-8')
            
            # Quick test
            req = urllib.request.Request(
                "https://api.coinbase.com/api/v3/brokerage/time",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "CB-VERSION": "2024-10-24"
                },
                method="GET"
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                print(f"      ✅ Works with {offset:+d}s offset")
        except:
            print(f"      ❌ Fails with {offset:+d}s offset")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nKey findings:")
    print("1. Check which URI format works (if any)")
    print("2. Verify timestamp tolerance")
    print("3. If public endpoints work but private don't → CDP issue")
    print("4. If nothing works → Signing or key issue")


if __name__ == "__main__":
    test_signing_variations()