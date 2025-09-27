#!/usr/bin/env python3
"""
Test various Coinbase CDP endpoint variations.
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

load_env_file(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bot_v2.features.brokerages.coinbase.cdp_auth import create_cdp_auth

print("Testing CDP Endpoint Variations")
print("=" * 60)

cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

# Create auth
auth = create_cdp_auth(cdp_api_key, cdp_private_key)

# Test different base URLs and paths
test_cases = [
    # Standard Advanced Trade API
    ("https://api.coinbase.com", "/api/v3/brokerage/accounts"),
    ("https://api.coinbase.com", "/api/v3/brokerage/products"),
    
    # CDP specific endpoints (guessing)
    ("https://api.cdp.coinbase.com", "/api/v3/brokerage/accounts"),
    ("https://api.developer.coinbase.com", "/api/v3/brokerage/accounts"),
    
    # Platform API endpoints
    ("https://api.coinbase.com", "/platform/v1/accounts"),
    ("https://api.coinbase.com", "/platform/accounts"),
    
    # Commerce API
    ("https://api.commerce.coinbase.com", "/accounts"),
    
    # International/Intx endpoints
    ("https://api.international.coinbase.com", "/api/v3/brokerage/accounts"),
    ("https://intx.coinbase.com", "/api/v3/brokerage/accounts"),
    
    # Without /api prefix
    ("https://api.coinbase.com", "/v3/brokerage/accounts"),
    
    # Legacy endpoints
    ("https://api.coinbase.com", "/v2/user"),
    ("https://api.coinbase.com", "/v2/accounts"),
]

for base_url, path in test_cases:
    print(f"\nTesting: {base_url}{path}")
    
    # Generate JWT for the path
    jwt_headers = auth.sign("GET", path)
    
    # Make request
    url = f"{base_url}{path}"
    req = urllib.request.Request(url, method="GET")
    for k, v in jwt_headers.items():
        req.add_header(k, v)
    req.add_header("User-Agent", "GPT-Trader/1.0")
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.getcode()
            data = response.read().decode()
            
            # Check if it's JSON
            try:
                result = json.loads(data)
                if 'accounts' in result or 'data' in result:
                    print(f"  ✅ SUCCESS! Status: {status}")
                    print(f"     Response keys: {list(result.keys())[:5]}")
                else:
                    print(f"  ⚠️  Status: {status} but unexpected response")
                    print(f"     Keys: {list(result.keys())[:5] if isinstance(result, dict) else 'not a dict'}")
            except:
                if "html" in data.lower():
                    print(f"  ⚠️  Status: {status} but got HTML")
                else:
                    print(f"  ⚠️  Status: {status}")
                    print(f"     Response: {data[:100]}")
                    
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  ❌ 404 Not Found (endpoint doesn't exist)")
        elif e.code == 401:
            print(f"  ❌ 401 Unauthorized")
        elif e.code == 403:
            print(f"  ❌ 403 Forbidden")
        else:
            print(f"  ❌ HTTP {e.code}")
            
    except urllib.error.URLError as e:
        # DNS or connection error
        print(f"  ❌ Connection failed (host might not exist)")
    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}")

print("\n" + "=" * 60)
print("\nConclusions:")
print("- If all api.coinbase.com endpoints return 401, the JWT auth is working but the key lacks permissions")
print("- If other domains work, CDP might use a different API endpoint")
print("- 404 errors mean the endpoint doesn't exist")
print("- Connection errors mean the domain doesn't exist")