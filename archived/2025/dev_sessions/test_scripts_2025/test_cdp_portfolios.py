#!/usr/bin/env python3
"""
Test if CDP requires portfolio-specific endpoints.
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

print("CDP Portfolio and Permission Test")
print("=" * 60)

cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

# Parse organization ID
org_id = cdp_api_key.split('/')[1] if cdp_api_key else None
print(f"Organization ID: {org_id}")

# Create auth
auth = create_cdp_auth(cdp_api_key, cdp_private_key)

# Test endpoints that might reveal permission issues
test_endpoints = [
    # Portfolio-specific endpoints
    ("/api/v3/brokerage/portfolios", "List portfolios"),
    ("/api/v3/brokerage/portfolios/default/accounts", "Default portfolio accounts"),
    
    # Different account variations
    ("/api/v3/brokerage/accounts?limit=1", "Accounts with limit param"),
    ("/api/v3/brokerage/accounts?retail_portfolio_id=default", "Accounts with portfolio param"),
    
    # User and permission endpoints
    ("/api/v3/brokerage/user", "User info"),
    ("/api/v3/brokerage/user/auth", "User auth info"),
    ("/api/v3/brokerage/transaction_summary", "Transaction summary"),
    
    # Best bid/offer (should work with view permission)
    ("/api/v3/brokerage/best_bid_ask?product_ids=BTC-USD", "Best bid/ask"),
    
    # Product book (public-ish)
    ("/api/v3/brokerage/product_book?product_id=BTC-USD&limit=1", "Product book"),
    
    # Simple time endpoint
    ("/api/v3/brokerage/time", "Server time"),
]

success_count = 0
for path, description in test_endpoints:
    print(f"\n{description}")
    print(f"  Path: {path}")
    
    # Generate JWT
    # Extract just the path without query params for signing
    sign_path = path.split('?')[0]
    jwt_headers = auth.sign("GET", sign_path)
    
    # Make request
    url = f"https://api.coinbase.com{path}"
    req = urllib.request.Request(url, method="GET")
    for k, v in jwt_headers.items():
        req.add_header(k, v)
    req.add_header("User-Agent", "GPT-Trader/1.0")
    
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.getcode()
            data = response.read().decode()
            
            try:
                result = json.loads(data)
                print(f"  ✅ SUCCESS! Status: {status}")
                success_count += 1
                
                # Show relevant info
                if isinstance(result, dict):
                    if 'portfolios' in result:
                        portfolios = result['portfolios']
                        print(f"     Found {len(portfolios)} portfolios")
                        for p in portfolios[:2]:
                            print(f"     - {p.get('name', 'unnamed')}: {p.get('uuid', 'no-id')}")
                    elif 'accounts' in result:
                        print(f"     Found {len(result['accounts'])} accounts")
                    elif 'iso' in result:
                        print(f"     Server time: {result.get('iso')}")
                    else:
                        print(f"     Response keys: {list(result.keys())[:5]}")
            except:
                print(f"  ⚠️  Status: {status} but non-JSON response")
                    
    except urllib.error.HTTPError as e:
        error_body = ""
        if hasattr(e, 'read'):
            try:
                error_body = e.read().decode()
            except:
                pass
        
        if e.code == 404:
            print(f"  ❌ 404 Not Found")
        elif e.code == 401:
            print(f"  ❌ 401 Unauthorized")
            # Try to get more details from error
            if error_body:
                try:
                    error_json = json.loads(error_body)
                    if error_json != {"message": "Unauthorized"} and error_json != "Unauthorized":
                        print(f"     Details: {error_json}")
                except:
                    if error_body != "Unauthorized":
                        print(f"     Body: {error_body[:100]}")
        elif e.code == 403:
            print(f"  ❌ 403 Forbidden (insufficient permissions)")
            if error_body:
                print(f"     Details: {error_body[:200]}")
        else:
            print(f"  ❌ HTTP {e.code}")
            
    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}")

print("\n" + "=" * 60)
print(f"\nResults: {success_count} successful endpoints out of {len(test_endpoints)}")

if success_count == 0:
    print("\n⚠️  No endpoints succeeded. Possible issues:")
    print("1. API key not activated - check CDP dashboard")
    print("2. Key not linked to a portfolio - may need to link in settings")
    print("3. Wrong key type - ensure it's an 'Advanced Trade API' key")
    print("4. Key might be for testnet/sandbox only")
    print("\nTry creating a new API key specifically for 'Advanced Trade API'")
else:
    print(f"\n✅ Some endpoints worked! The key has partial access.")
    print("Check which endpoints succeeded to understand permission scope.")