#!/usr/bin/env python3
"""
Test CDP authentication with SDK-compatible JWT format.
"""

import os
import sys
import json
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

# Load environment
load_env_file(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bot_v2.features.brokerages.coinbase.cdp_auth import create_cdp_auth
from src.bot_v2.features.brokerages.coinbase.cdp_auth_v2 import create_cdp_auth_v2

print("CDP Authentication Comparison Test")
print("=" * 60)

cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

# Test both versions
test_cases = [
    ("Original implementation", create_cdp_auth(cdp_api_key, cdp_private_key)),
    ("SDK-compatible V2", create_cdp_auth_v2(cdp_api_key, cdp_private_key))
]

for name, auth in test_cases:
    print(f"\n{name}")
    print("-" * 40)
    
    method = "GET"
    path = "/api/v3/brokerage/accounts"
    
    # Generate JWT
    jwt_token = auth.generate_jwt(method, path)
    
    # Decode to compare structure
    import base64
    parts = jwt_token.split('.')
    
    # Decode header
    header_b64 = parts[0]
    header_b64 += '=' * (4 - len(header_b64) % 4)
    header = json.loads(base64.urlsafe_b64decode(header_b64))
    
    # Decode payload
    payload_b64 = parts[1]
    payload_b64 += '=' * (4 - len(payload_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    
    print("JWT Structure:")
    print(f"  Issuer: {payload.get('iss')}")
    print(f"  URI: {payload.get('uri')}")
    print(f"  Audience: {payload.get('aud', 'None')}")
    print(f"  Nonce type: {'hex' if len(header.get('nonce', '')) > 20 else 'timestamp'}")
    
    # Test the request
    headers = auth.sign(method, path)
    
    url = f"https://api.coinbase.com{path}"
    req = urllib.request.Request(url, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    
    print("\nTesting API call...")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.getcode()
            data = response.read().decode()
            print(f"  ✅ SUCCESS! Status: {status}")
            
            result = json.loads(data)
            if 'accounts' in result:
                print(f"     Found {len(result['accounts'])} accounts")
                
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        
        # Check if there's a specific error message
        if hasattr(e, 'read'):
            try:
                error_body = e.read().decode()
                if error_body and error_body != "Unauthorized":
                    print(f"     Error details: {error_body}")
            except:
                pass
                
    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("\nComparison with Official SDK:")
print("-" * 40)

# Show what the SDK generates for comparison
try:
    from coinbase import jwt_generator
    
    uri = "/api/v3/brokerage/accounts"
    jwt_uri = jwt_generator.format_jwt_uri("GET", uri)
    sdk_jwt = jwt_generator.build_rest_jwt(jwt_uri, cdp_api_key, cdp_private_key)
    
    # Decode SDK JWT
    parts = sdk_jwt.split('.')
    payload_b64 = parts[1]
    payload_b64 += '=' * (4 - len(payload_b64) % 4)
    sdk_payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    
    print("Official SDK JWT:")
    print(f"  Issuer: {sdk_payload.get('iss')}")
    print(f"  URI: {sdk_payload.get('uri')}")
    print(f"  Audience: {sdk_payload.get('aud', 'None')}")
    
except Exception as e:
    print(f"Could not generate SDK JWT: {e}")

print("\n" + "=" * 60)
print("\nConclusions:")
if "✅" in str(locals()):
    print("One of our implementations worked!")
else:
    print("Neither implementation worked. The issue is likely:")
    print("1. CDP key not activated for Advanced Trade API")
    print("2. Key created for wrong API type (Platform vs Advanced Trade)")
    print("3. Regional or account type restrictions")
    print("4. CDP bug requiring legacy keys as workaround")