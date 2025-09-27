#!/usr/bin/env python3
"""
Debug JWT token generation and validate against CDP requirements.
"""

import os
import sys
import json
import time
import base64
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def debug_jwt():
    """Debug JWT token generation in detail."""
    
    load_dotenv()
    
    print("=" * 70)
    print("JWT TOKEN DETAILED ANALYSIS")
    print("=" * 70)
    
    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not api_key or not private_key:
        print("❌ Missing credentials")
        return
    
    # Import and create auth
    from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    
    auth = CDPAuthV2(
        api_key_name=api_key,
        private_key_pem=private_key,
        base_host="api.coinbase.com"
    )
    
    # Generate token
    method = "GET"
    path = "/api/v3/brokerage/accounts"
    
    jwt_token = auth.generate_jwt(method, path)
    
    print(f"\n1️⃣ JWT Token Generated:")
    print(f"   Length: {len(jwt_token)} characters")
    print(f"   First 50 chars: {jwt_token[:50]}...")
    
    # Decode without verification
    decoded = jwt.decode(jwt_token, options={"verify_signature": False})
    headers = jwt.get_unverified_header(jwt_token)
    
    print(f"\n2️⃣ JWT Headers:")
    for key, value in headers.items():
        print(f"   {key}: {value}")
    
    print(f"\n3️⃣ JWT Claims:")
    for key, value in decoded.items():
        if key in ['nbf', 'exp']:
            # Convert timestamps to readable format
            timestamp = value
            readable = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(timestamp))
            print(f"   {key}: {value} ({readable})")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n4️⃣ Time Analysis:")
    current_time = int(time.time())
    print(f"   Current time: {current_time}")
    print(f"   Token issued at (nbf): {decoded.get('nbf', 'N/A')}")
    print(f"   Token expires at (exp): {decoded.get('exp', 'N/A')}")
    print(f"   Time until expiry: {decoded.get('exp', 0) - current_time} seconds")
    
    print(f"\n5️⃣ Private Key Analysis:")
    try:
        # Load and analyze the private key
        key_obj = serialization.load_pem_private_key(
            private_key.encode() if isinstance(private_key, str) else private_key,
            password=None,
            backend=default_backend()
        )
        
        # Get key details
        from cryptography.hazmat.primitives.asymmetric import ec
        
        if isinstance(key_obj, ec.EllipticCurvePrivateKey):
            curve = key_obj.curve
            print(f"   ✅ EC Private Key loaded successfully")
            print(f"   Curve: {curve.name}")
            print(f"   Key size: {curve.key_size} bits")
            
            # Check if it's P-256 (required by CDP)
            if curve.name == "secp256r1":
                print(f"   ✅ Using correct P-256 curve (secp256r1)")
            else:
                print(f"   ❌ Wrong curve! CDP requires P-256/secp256r1")
        else:
            print(f"   ❌ Not an EC key: {type(key_obj).__name__}")
            
    except Exception as e:
        print(f"   ❌ Failed to analyze key: {e}")
    
    print(f"\n6️⃣ URI Claim Validation:")
    uri = decoded.get('uri', '')
    expected_uri = f"{method} api.coinbase.com{path}"
    print(f"   Generated URI: {uri}")
    print(f"   Expected URI: {expected_uri}")
    if uri == expected_uri:
        print(f"   ✅ URI claim is correct")
    else:
        print(f"   ❌ URI claim mismatch!")
    
    print(f"\n7️⃣ API Key Format Check:")
    print(f"   Full API Key: {api_key}")
    if api_key.startswith("organizations/") and "/apiKeys/" in api_key:
        print(f"   ✅ API key format is correct")
        parts = api_key.split("/")
        if len(parts) == 4:
            org_id = parts[1]
            key_id = parts[3]
            print(f"   Organization ID: {org_id}")
            print(f"   Key ID: {key_id}")
    else:
        print(f"   ❌ API key format is incorrect!")
        print(f"   Expected: organizations/{{org_id}}/apiKeys/{{key_id}}")
    
    print(f"\n8️⃣ Summary:")
    checks = []
    
    # Check algorithm
    if headers.get('alg') == 'ES256':
        checks.append("✅ Correct ES256 algorithm")
    else:
        checks.append(f"❌ Wrong algorithm: {headers.get('alg')}")
    
    # Check issuer
    if decoded.get('iss') == 'cdp':
        checks.append("✅ Correct issuer (cdp)")
    else:
        checks.append(f"❌ Wrong issuer: {decoded.get('iss')}")
    
    # Check time validity
    if current_time >= decoded.get('nbf', 0) and current_time <= decoded.get('exp', 0):
        checks.append("✅ Token time is valid")
    else:
        checks.append("❌ Token time is invalid (expired or not yet valid)")
    
    # Check kid header
    if headers.get('kid') == api_key:
        checks.append("✅ Correct kid header")
    else:
        checks.append(f"❌ Wrong kid header: {headers.get('kid')}")
    
    for check in checks:
        print(f"   {check}")
    
    return all("✅" in check for check in checks)


if __name__ == "__main__":
    success = debug_jwt()
    sys.exit(0 if success else 1)