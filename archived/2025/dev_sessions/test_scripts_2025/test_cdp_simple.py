#!/usr/bin/env python3
"""
Test different CDP endpoints to understand auth issue.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

# Load environment
def load_production_env():
    env_file = Path(__file__).parent.parent / ".env.production"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if value.startswith('"') and not value.endswith('"'):
                            lines = [value[1:]]
                            for next_line in f:
                                next_line = next_line.strip()
                                if next_line.endswith('"'):
                                    lines.append(next_line[:-1])
                                    value = '\n'.join(lines)
                                    break
                                lines.append(next_line)
                        else:
                            value = value.strip('"')
                        os.environ[key] = value

load_production_env()

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bot_v2.features.brokerages.coinbase.cdp_auth import create_cdp_auth

# Get credentials
cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

print("CDP Endpoint Tests")
print("=" * 60)

# Create auth
auth = create_cdp_auth(cdp_api_key, cdp_private_key)

# Test different endpoints
endpoints = [
    ("GET", "/api/v3/brokerage/accounts", "Get accounts (requires authentication)"),
    ("GET", "/api/v3/brokerage/products", "List products (public, but test with auth)"),
    ("GET", "/api/v3/brokerage/products/BTC-USD", "Get BTC-USD product"),
    ("GET", "/api/v3/brokerage/market/products/BTC-USD/ticker", "Get BTC ticker"),
]

for method, path, description in endpoints:
    print(f"\nTesting: {description}")
    print(f"  {method} {path}")
    
    # Generate headers
    headers = auth.sign(method, path)
    
    # Make request
    url = f"https://api.coinbase.com{path}"
    req = urllib.request.Request(url, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            data = response.read().decode()
            result = json.loads(data) if data else {}
            
            # Show limited response
            if isinstance(result, dict):
                if 'accounts' in result:
                    print(f"  ✅ Success! Found {len(result.get('accounts', []))} accounts")
                elif 'products' in result:
                    print(f"  ✅ Success! Found {len(result.get('products', []))} products")
                elif 'product_id' in result:
                    print(f"  ✅ Success! Product: {result.get('product_id')}")
                elif 'trades' in result:
                    print(f"  ✅ Success! Got ticker data")
                else:
                    print(f"  ✅ Success! Status: {status}")
                    print(f"  Response keys: {list(result.keys())[:5]}")
            else:
                print(f"  ✅ Success! Status: {status}")
                
    except urllib.error.HTTPError as e:
        error_body = ""
        if hasattr(e, 'read'):
            error_body = e.read().decode()
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        if error_body:
            try:
                error_json = json.loads(error_body)
                if 'error' in error_json:
                    print(f"     Error: {error_json['error']}")
                if 'message' in error_json:
                    print(f"     Message: {error_json['message']}")
            except:
                print(f"     Body: {error_body[:100]}")
    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("\nNOTE: If all endpoints fail with 401, the API key may need to be:")
print("  1. Activated in Coinbase CDP dashboard")
print("  2. Given proper permissions/scopes")
print("  3. Associated with your account")
print("\nIf public endpoints work but authenticated ones fail, it's a permission issue.")