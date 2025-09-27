#!/usr/bin/env python3
# NOTE: Archived from tests/; excluded from active test suite.
"""
Full CDP Authentication Test

This script tests all CDP endpoints using the official Coinbase SDK
to verify complete functionality.
"""

import os
import sys
import requests
import json
from pathlib import Path

def load_env_file(env_file):
    """Load environment variables from .env file."""
    if not Path(env_file).exists():
        return {}
    
    env_vars = {}
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Split by lines and handle multi-line values
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Handle multi-line values (like PEM keys)
            if value.startswith('-----BEGIN'):
                # Collect all lines until the end marker
                pem_lines = [value]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('-----END'):
                    pem_lines.append(lines[i].strip())
                    i += 1
                if i < len(lines):
                    pem_lines.append(lines[i].strip())
                value = '\n'.join(pem_lines)
            
            env_vars[key] = value
        i += 1
    
    return env_vars

def test_endpoint_with_sdk(path, method="GET", description=""):
    """Test an endpoint using the official SDK."""
    try:
        from coinbase import jwt_generator
        
        # Load environment variables
        env_vars = load_env_file('.env')
        cdp_key = env_vars.get('COINBASE_CDP_API_KEY', '')
        cdp_private = env_vars.get('COINBASE_CDP_PRIVATE_KEY', '')
        
        if not cdp_key or not cdp_private:
            return False, "Missing credentials"
        
        # Generate JWT using official SDK
        jwt_uri = jwt_generator.format_jwt_uri(method, path)
        jwt_token = jwt_generator.build_rest_jwt(jwt_uri, cdp_key, cdp_private)
        
        # Make the request
        url = f"https://api.coinbase.com{path}"
        headers = {"Authorization": f"Bearer {jwt_token}"}
        
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=15)
        else:
            response = requests.post(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Status {response.status_code}: {response.text}"
            
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("FULL CDP AUTHENTICATION TEST")
    print("=" * 60)
    print()
    
    print("Testing all CDP endpoints using the official SDK:")
    print()
    
    # Test endpoints
    endpoints = [
        {
            'path': '/api/v3/brokerage/key_permissions',
            'method': 'GET',
            'description': 'Key Permissions'
        },
        {
            'path': '/api/v3/brokerage/time',
            'method': 'GET',
            'description': 'Server Time'
        },
        {
            'path': '/api/v3/brokerage/products',
            'method': 'GET',
            'description': 'Products List'
        },
        {
            'path': '/api/v3/brokerage/accounts',
            'method': 'GET',
            'description': 'Accounts'
        },
        {
            'path': '/api/v3/brokerage/products/BTC-USD/ticker',
            'method': 'GET',
            'description': 'BTC-USD Ticker'
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        print(f"🔍 Testing {endpoint['description']}:")
        success, result = test_endpoint_with_sdk(
            endpoint['path'], 
            endpoint['method'], 
            endpoint['description']
        )
        
        if success:
            print(f"   ✅ {endpoint['description']} working")
            if endpoint['description'] == 'Key Permissions':
                print(f"   📋 Permissions: {json.dumps(result, indent=6)}")
            elif endpoint['description'] == 'Products List':
                print(f"   📊 Found {len(result.get('products', []))} products")
            elif endpoint['description'] == 'Accounts':
                print(f"   💰 Found {len(result.get('accounts', []))} accounts")
            elif endpoint['description'] == 'BTC-USD Ticker':
                print(f"   💱 Price: {result.get('price', 'N/A')}")
        else:
            print(f"   ❌ {endpoint['description']} failed: {result}")
        
        results.append({
            'endpoint': endpoint['description'],
            'success': success,
            'result': result
        })
        
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful == total:
        print()
        print("🎉 ALL TESTS PASSED!")
        print("Your CDP setup is working perfectly!")
        print()
        print("📋 Next Steps:")
        print("1. Create a monitor key (read-only) for production")
        print("2. Create demo/sandbox keys for testing")
        print("3. Test with your trading application")
    else:
        print()
        print("⚠️  Some tests failed. Check the specific error messages above.")
        print("Most likely causes:")
        print("• IP allowlist restrictions")
        print("• Portfolio linking issues")
        print("• Specific endpoint permissions")

if __name__ == "__main__":
    main()
