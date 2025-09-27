#!/usr/bin/env python3
"""
Detailed CDP authentication test with verbose error reporting.
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
    """Manually parse .env file."""
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

print("CDP Detailed Authentication Test")
print("=" * 60)

# Get credentials
cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

print(f"API Key Name: {cdp_api_key}")
print(f"Private Key: {'Present' if cdp_private_key else 'Missing'}")
print()

# Parse the API key structure
if cdp_api_key:
    parts = cdp_api_key.split('/')
    if len(parts) >= 4:
        org_id = parts[1]
        key_id = parts[3]
        print(f"Organization ID: {org_id}")
        print(f"Key ID: {key_id}")
    print()

# Create auth
auth = create_cdp_auth(cdp_api_key, cdp_private_key)

# Test different request variations
test_cases = [
    {
        "name": "Standard accounts endpoint",
        "method": "GET",
        "path": "/api/v3/brokerage/accounts",
        "headers": {}
    },
    {
        "name": "Accounts with explicit accept header",
        "method": "GET", 
        "path": "/api/v3/brokerage/accounts",
        "headers": {"Accept": "application/json"}
    },
    {
        "name": "Root API endpoint",
        "method": "GET",
        "path": "/",
        "headers": {}
    },
    {
        "name": "V2 accounts (legacy)",
        "method": "GET",
        "path": "/v2/accounts",
        "headers": {}
    },
]

for test in test_cases:
    print(f"\nTest: {test['name']}")
    print("-" * 40)
    
    # Generate JWT
    jwt_headers = auth.sign(test["method"], test["path"])
    
    # Merge headers
    headers = {**jwt_headers, **test["headers"]}
    
    # Make request
    url = f"https://api.coinbase.com{test['path']}"
    req = urllib.request.Request(url, method=test["method"])
    for k, v in headers.items():
        req.add_header(k, v)
    
    # Add User-Agent
    req.add_header("User-Agent", "GPT-Trader/1.0")
    
    print(f"  URL: {url}")
    print(f"  Method: {test['method']}")
    print(f"  Headers:")
    for k, v in headers.items():
        if k == "Authorization":
            print(f"    {k}: {v[:70]}...")
        else:
            print(f"    {k}: {v}")
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            data = response.read().decode()
            
            print(f"  ✅ Status: {status}")
            
            # Parse response
            try:
                result = json.loads(data)
                if isinstance(result, dict):
                    print(f"  Response keys: {list(result.keys())}")
                    if 'accounts' in result:
                        print(f"  Accounts found: {len(result['accounts'])}")
                    elif 'data' in result:
                        print(f"  Data entries: {len(result.get('data', []))}")
                else:
                    print(f"  Response type: {type(result).__name__}")
            except:
                print(f"  Response: {data[:200]}")
                
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        
        # Get detailed error
        error_body = ""
        if hasattr(e, 'read'):
            try:
                error_body = e.read().decode()
            except:
                pass
        
        # Show headers that might have clues
        if hasattr(e, 'headers'):
            relevant_headers = ['x-request-id', 'cf-ray', 'www-authenticate', 'x-cb-error-code']
            for header in relevant_headers:
                value = e.headers.get(header)
                if value:
                    print(f"    {header}: {value}")
        
        if error_body:
            print(f"    Error body: {error_body[:200]}")
            try:
                error_json = json.loads(error_body)
                if 'error' in error_json:
                    print(f"    Error code: {error_json.get('error')}")
                if 'error_details' in error_json:
                    print(f"    Details: {error_json.get('error_details')}")
                if 'message' in error_json:
                    print(f"    Message: {error_json.get('message')}")
            except:
                pass
                
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("\nDiagnostic Summary:")
print("-" * 40)

# Check time sync
current_time = int(time.time())
print(f"System time (Unix): {current_time}")
print(f"System time (UTC): {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

print("\nIf all requests fail with 401:")
print("1. The API key may need to be 'activated' in CDP dashboard")
print("2. The key might be in a different 'environment' (staging vs production)")
print("3. CDP keys might require additional setup steps")
print("4. Try regenerating the key with the same permissions")