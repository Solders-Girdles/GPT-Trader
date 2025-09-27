#!/usr/bin/env python3
"""
Test if CDP requires OAuth flow or different authentication method.
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

print("CDP OAuth and Alternative Auth Test")
print("=" * 60)

cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

# Create standard auth
auth = create_cdp_auth(cdp_api_key, cdp_private_key)

# Test different authentication endpoints
test_cases = [
    {
        "name": "OAuth token endpoint",
        "url": "https://api.coinbase.com/oauth/token",
        "method": "POST",
        "body": json.dumps({
            "grant_type": "client_credentials",
            "client_id": cdp_api_key
        })
    },
    {
        "name": "Sign In With Coinbase",
        "url": "https://api.coinbase.com/v2/user",
        "method": "GET",
        "body": None
    },
    {
        "name": "CDP Platform endpoint",
        "url": "https://api.coinbase.com/platform/v1/users/me",
        "method": "GET",
        "body": None
    },
    {
        "name": "Institutional API",
        "url": "https://api.coinbase.com/api/v3/brokerage/cfm/accounts",
        "method": "GET",
        "body": None
    },
    {
        "name": "Prime API",
        "url": "https://api.coinbase.com/prime/v1/portfolios",
        "method": "GET",
        "body": None
    }
]

for test in test_cases:
    print(f"\n{test['name']}")
    print(f"  URL: {test['url']}")
    
    # Parse path from URL
    path = test['url'].replace("https://api.coinbase.com", "")
    
    # Get JWT headers
    jwt_headers = auth.sign(test['method'], path, test.get('body'))
    
    # Create request
    req = urllib.request.Request(
        test['url'], 
        method=test['method'],
        data=test['body'].encode() if test['body'] else None
    )
    
    # Add headers
    for k, v in jwt_headers.items():
        req.add_header(k, v)
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.getcode()
            data = response.read().decode()
            
            try:
                result = json.loads(data)
                print(f"  ✅ Status: {status}")
                print(f"     Response keys: {list(result.keys())[:5]}")
                
                # Check for useful info
                if 'access_token' in result:
                    print(f"     🔑 Got access token!")
                if 'user' in result:
                    print(f"     👤 Got user info")
                if 'portfolios' in result:
                    print(f"     📊 Got portfolios")
                    
            except:
                print(f"  ⚠️  Status: {status} (non-JSON)")
                    
    except urllib.error.HTTPError as e:
        error_body = ""
        if hasattr(e, 'read'):
            try:
                error_body = e.read().decode()
            except:
                pass
        
        print(f"  ❌ HTTP {e.code}")
        
        # Show error details if interesting
        if error_body and error_body != "Unauthorized":
            try:
                error_json = json.loads(error_body)
                if 'error' in error_json:
                    print(f"     Error: {error_json['error']}")
                if 'error_description' in error_json:
                    print(f"     Description: {error_json['error_description']}")
            except:
                if len(error_body) < 200:
                    print(f"     Body: {error_body}")
                    
    except Exception as e:
        print(f"  ❌ {type(e).__name__}")

print("\n" + "=" * 60)
print("\nDiagnostic Information:")
print(f"Organization ID: {cdp_api_key.split('/')[1] if cdp_api_key else 'N/A'}")
print(f"Key ID: {cdp_api_key.split('/')[3] if cdp_api_key and len(cdp_api_key.split('/')) > 3 else 'N/A'}")

print("\nRecommendations:")
print("1. If OAuth endpoint works, you may need to exchange JWT for access token first")
print("2. If Prime/Institutional endpoints work, your key might be for those APIs")
print("3. If all fail, contact Coinbase support with this specific error:")
print("   'CDP API key with all permissions linked to Default portfolio")
print("    returns 401 for all Advanced Trade API endpoints'")
print("4. Ask support to verify your key is enabled for 'Advanced Trade API v3'")