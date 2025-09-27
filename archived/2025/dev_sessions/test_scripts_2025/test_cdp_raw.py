#!/usr/bin/env python3
"""
Raw test of CDP authentication using urllib directly.
"""

import os
import json
import time
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
                        # Handle multi-line values (like private keys)
                        if value.startswith('"') and not value.endswith('"'):
                            lines = [value[1:]]  # Remove opening quote
                            for next_line in f:
                                next_line = next_line.strip()
                                if next_line.endswith('"'):
                                    lines.append(next_line[:-1])  # Remove closing quote
                                    value = '\n'.join(lines)
                                    break
                                lines.append(next_line)
                        else:
                            value = value.strip('"')
                        os.environ[key] = value

load_production_env()

# Import after env is loaded
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bot_v2.features.brokerages.coinbase.cdp_auth import create_cdp_auth

# Get credentials
cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

print("CDP Raw Authentication Test")
print("=" * 60)
print(f"API Key: ...{cdp_api_key[-30:]}")
print(f"Private Key: {'Present' if cdp_private_key else 'Missing'}")

# Create auth and generate JWT
auth = create_cdp_auth(cdp_api_key, cdp_private_key)
method = "GET"
path = "/api/v3/brokerage/accounts"
headers = auth.sign(method, path)

print(f"\nGenerated headers:")
for k, v in headers.items():
    if k == "Authorization":
        print(f"  {k}: {v[:60]}...")
    else:
        print(f"  {k}: {v}")

# Make request
url = f"https://api.coinbase.com{path}"
req = urllib.request.Request(url, method=method)
for k, v in headers.items():
    req.add_header(k, v)

print(f"\nMaking request to: {url}")
print(f"Method: {method}")

try:
    with urllib.request.urlopen(req, timeout=10) as response:
        status = response.getcode()
        data = response.read().decode()
        result = json.loads(data) if data else {}
        
        print(f"\n✅ Success! Status: {status}")
        print(f"Response preview: {json.dumps(result, indent=2)[:500]}")
        
except urllib.error.HTTPError as e:
    print(f"\n❌ HTTP {e.code}: {e.reason}")
    if hasattr(e, 'headers'):
        print("Response headers:")
        for key, value in e.headers.items():
            if key.lower() in ['x-request-id', 'cf-ray', 'date', 'content-type']:
                print(f"  {key}: {value}")
    if hasattr(e, 'read'):
        error_body = e.read().decode()
        print(f"\nError body: {error_body}")
        try:
            error_json = json.loads(error_body)
            print(f"Parsed error: {json.dumps(error_json, indent=2)}")
        except:
            pass
            
except Exception as e:
    print(f"\n❌ Error: {type(e).__name__}: {e}")