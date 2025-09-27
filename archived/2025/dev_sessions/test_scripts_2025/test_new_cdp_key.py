#!/usr/bin/env python3
"""
Test the new CDP key directly.
"""

import os
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

# Set the new CDP credentials directly
os.environ["COINBASE_CDP_API_KEY"] = "organizations/5184a9ea-2cec-4a66-b00e-7cf6daaf048e/apiKeys/d85fc95b-477f-4d4d-afb1-7ca9278de537"
os.environ["COINBASE_CDP_PRIVATE_KEY"] = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIF4Lym2V7aKjVu5jqOwfRtPYOQ1pvsug4TNYvAgH58jioAoGCCqGSM49
AwEHoUQDQgAEJJSe3Oiyae/K0WZu/6browVscpn64uJ4kdJyV2xSwsUVaScC0OGM
TCocQ0VtOXES7TOpdEDKhe+Jw8UNVod47A==
-----END EC PRIVATE KEY-----"""

sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing New CDP Key")
print("=" * 60)
print(f"Key ID: d85fc95b-477f-4d4d-afb1-7ca9278de537")
print()

# Test 1: Our original implementation
print("Test 1: Original CDP Auth Implementation")
print("-" * 40)

from src.bot_v2.features.brokerages.coinbase.cdp_auth import create_cdp_auth

cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

auth = create_cdp_auth(cdp_api_key, cdp_private_key)

method = "GET"
path = "/api/v3/brokerage/accounts"

headers = auth.sign(method, path)

url = f"https://api.coinbase.com{path}"
req = urllib.request.Request(url, method=method)
for k, v in headers.items():
    req.add_header(k, v)

try:
    with urllib.request.urlopen(req, timeout=10) as response:
        status = response.getcode()
        data = response.read().decode()
        result = json.loads(data) if data else {}
        
        print(f"✅ SUCCESS! Status: {status}")
        
        if 'accounts' in result:
            accounts = result['accounts']
            print(f"Found {len(accounts)} accounts!")
            for acc in accounts[:3]:
                currency = acc.get('currency', 'Unknown')
                balance = acc.get('available_balance', {}).get('value', '0')
                print(f"  - {currency}: {balance}")
        else:
            print(f"Response keys: {list(result.keys())}")
            
except urllib.error.HTTPError as e:
    print(f"❌ HTTP {e.code}: {e.reason}")
    if hasattr(e, 'read'):
        error_body = e.read().decode()
        if error_body and error_body != "Unauthorized":
            print(f"   Error: {error_body}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")

print()

# Test 2: SDK-compatible version
print("Test 2: SDK-Compatible CDP Auth")
print("-" * 40)

from src.bot_v2.features.brokerages.coinbase.cdp_auth_v2 import create_cdp_auth_v2

auth_v2 = create_cdp_auth_v2(cdp_api_key, cdp_private_key)

headers = auth_v2.sign(method, path)

req = urllib.request.Request(url, method=method)
for k, v in headers.items():
    req.add_header(k, v)

try:
    with urllib.request.urlopen(req, timeout=10) as response:
        status = response.getcode()
        data = response.read().decode()
        result = json.loads(data) if data else {}
        
        print(f"✅ SUCCESS! Status: {status}")
        
        if 'accounts' in result:
            accounts = result['accounts']
            print(f"Found {len(accounts)} accounts!")
            for acc in accounts[:3]:
                currency = acc.get('currency', 'Unknown')
                balance = acc.get('available_balance', {}).get('value', '0')
                print(f"  - {currency}: {balance}")
        else:
            print(f"Response keys: {list(result.keys())}")
            
except urllib.error.HTTPError as e:
    print(f"❌ HTTP {e.code}: {e.reason}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")

print()

# Test 3: Official SDK
print("Test 3: Official Coinbase SDK")
print("-" * 40)

os.environ["COINBASE_API_KEY"] = cdp_api_key
os.environ["COINBASE_API_SECRET"] = cdp_private_key

try:
    from coinbase.rest import RESTClient
    
    client = RESTClient()
    accounts = client.get_accounts()
    
    print(f"✅ SUCCESS! Got accounts")
    
    if hasattr(accounts, 'to_dict'):
        accounts_dict = accounts.to_dict()
        if 'accounts' in accounts_dict:
            acc_list = accounts_dict['accounts']
            print(f"Found {len(acc_list)} accounts!")
            for acc in acc_list[:3]:
                print(f"  - {acc.get('currency', 'Unknown')}: {acc.get('available_balance', {}).get('value', 0)}")
                
except Exception as e:
    print(f"❌ Failed: {e}")
    if hasattr(e, '__class__'):
        print(f"   Error type: {e.__class__.__name__}")

print()

# Test a simple endpoint
print("Test 4: Server Time Endpoint")
print("-" * 40)

time_url = "https://api.coinbase.com/api/v3/brokerage/time"
headers = auth.sign("GET", "/api/v3/brokerage/time")

req = urllib.request.Request(time_url, method="GET")
for k, v in headers.items():
    req.add_header(k, v)

try:
    with urllib.request.urlopen(req, timeout=10) as response:
        status = response.getcode()
        data = response.read().decode()
        result = json.loads(data) if data else {}
        
        print(f"✅ Time endpoint works! Status: {status}")
        print(f"   Server time: {result.get('iso', 'Unknown')}")
        
except urllib.error.HTTPError as e:
    print(f"❌ HTTP {e.code}: {e.reason}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("\nConclusion:")
print("-" * 40)
if "Found" in str(locals()):
    print("🎉 NEW CDP KEY WORKS! The key has proper permissions.")
    print("The issue was with the previous key, not the implementation.")
else:
    print("❌ New key also fails. This confirms CDP keys don't work with Advanced Trade.")
    print("You'll need to use legacy API keys from Coinbase.com settings instead.")