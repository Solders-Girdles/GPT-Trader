#!/usr/bin/env python3
"""
Test Coinbase CDP keys using the official SDK.
This will help us determine if the issue is with our implementation or the keys themselves.
"""

import os
import sys
import json
from pathlib import Path

# Load environment manually
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

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
print(f"Loading environment from: {env_path}")
load_env_file(env_path)

# Get CDP credentials
cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

print("=" * 60)
print("Testing Coinbase Official SDK with CDP Keys")
print("=" * 60)

if not cdp_api_key or not cdp_private_key:
    print("❌ CDP credentials not found in environment")
    print("Please ensure COINBASE_CDP_API_KEY and COINBASE_CDP_PRIVATE_KEY are set")
    sys.exit(1)

print(f"CDP API Key: ...{cdp_api_key[-30:]}")
print(f"Private Key: {'Present' if cdp_private_key else 'Missing'}")
print()

# Method 1: Using environment variables
print("Method 1: Testing with environment variables")
print("-" * 40)

# Set the environment variables for the SDK
os.environ["COINBASE_API_KEY"] = cdp_api_key
os.environ["COINBASE_API_SECRET"] = cdp_private_key

try:
    from coinbase.rest import RESTClient
    
    # Create client using environment variables
    client = RESTClient()
    
    print("✅ Client created successfully")
    
    # Try to get accounts
    print("\nFetching accounts...")
    accounts = client.get_accounts()
    
    if accounts:
        print(f"✅ SUCCESS! Retrieved accounts")
        accounts_dict = accounts.to_dict() if hasattr(accounts, 'to_dict') else accounts
        print(f"   Response type: {type(accounts)}")
        
        if isinstance(accounts_dict, dict):
            if 'accounts' in accounts_dict:
                acc_list = accounts_dict['accounts']
                print(f"   Found {len(acc_list)} accounts")
                for acc in acc_list[:3]:
                    print(f"   - {acc.get('currency', 'Unknown')}: {acc.get('available_balance', {}).get('value', 0)}")
            else:
                print(f"   Response keys: {list(accounts_dict.keys())[:5]}")
        
        print("\n✅ CDP KEYS WORK WITH OFFICIAL SDK!")
        
    else:
        print("❌ No data returned")
        
except Exception as e:
    print(f"❌ Failed with environment variables: {e}")
    print(f"   Error type: {type(e).__name__}")
    if hasattr(e, 'response'):
        print(f"   Response: {e.response}")

print()

# Method 2: Direct instantiation with keys
print("Method 2: Testing with direct key instantiation")
print("-" * 40)

try:
    from coinbase.rest import RESTClient
    
    # Create client with explicit keys
    client = RESTClient(
        api_key=cdp_api_key,
        api_secret=cdp_private_key
    )
    
    print("✅ Client created with explicit keys")
    
    # Try a simple endpoint first
    print("\nFetching server time...")
    try:
        # The SDK might not have a direct time method, try accounts
        accounts = client.get_accounts()
        print(f"✅ Got response from API")
    except Exception as e:
        print(f"❌ API call failed: {e}")
        
except Exception as e:
    print(f"❌ Failed with direct keys: {e}")

print()

# Method 3: Check what JWT the SDK generates
print("Method 3: Examining SDK's JWT generation")
print("-" * 40)

try:
    from coinbase import jwt_generator
    
    # Generate a JWT using the SDK's method
    uri = "/api/v3/brokerage/accounts"
    jwt_uri = jwt_generator.format_jwt_uri("GET", uri)
    
    print(f"JWT URI format: {jwt_uri}")
    
    # Generate the JWT
    jwt_token = jwt_generator.build_rest_jwt(jwt_uri, cdp_api_key, cdp_private_key)
    
    print(f"✅ JWT generated successfully")
    print(f"   Token length: {len(jwt_token)} chars")
    print(f"   Token preview: {jwt_token[:50]}...")
    
    # Decode to see structure (without verification)
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
    
    print("\nSDK JWT Header:")
    print(json.dumps(header, indent=2))
    
    print("\nSDK JWT Payload:")
    print(json.dumps(payload, indent=2))
    
except Exception as e:
    print(f"❌ JWT generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Summary:")
print("-" * 40)
print("If the official SDK works, we need to match their JWT generation exactly.")
print("If it also fails, the issue is with the CDP key configuration/permissions.")