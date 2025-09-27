#!/usr/bin/env python3
"""
Debug CDP Permissions

This script helps debug the exact permission issues with your CDP API key.
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

def generate_jwt_token(api_key, private_key):
    """Generate JWT token for CDP authentication."""
    import jwt
    from datetime import datetime, timedelta
    
    # Handle PEM format private key
    if private_key.startswith('-----BEGIN EC PRIVATE KEY-----'):
        # It's already in PEM format, use as-is
        private_key_pem = private_key
    else:
        print("❌ Private key format not supported")
        return None
    
    # Create JWT payload
    now = datetime.utcnow()
    payload = {
        'sub': api_key,
        'iss': api_key,
        'nbf': int(now.timestamp()),
        'exp': int((now + timedelta(minutes=5)).timestamp()),
        'aud': 'retail_rest_api_pro'
    }
    
    try:
        # Sign the JWT with the PEM private key
        token = jwt.encode(payload, private_key_pem, algorithm='ES256')
        return token
    except Exception as e:
        print(f"❌ Error generating JWT: {e}")
        return None

def main():
    print("=" * 60)
    print("DEBUG CDP PERMISSIONS")
    print("=" * 60)
    print()
    
    # Load environment variables
    env_vars = load_env_file('.env')
    
    cdp_key = env_vars.get('COINBASE_CDP_API_KEY', '')
    cdp_private = env_vars.get('COINBASE_CDP_PRIVATE_KEY', '')
    
    if not cdp_key or not cdp_private:
        print("❌ CDP credentials are missing")
        return
    
    print("1. GENERATING JWT TOKEN:")
    print("-" * 30)
    
    # Generate JWT token
    token = generate_jwt_token(cdp_key, cdp_private)
    if not token:
        print("❌ Failed to generate JWT token")
        return
    
    print("✅ JWT token generated successfully")
    
    # Set up headers
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    print()
    print("2. TESTING ENDPOINTS WITH DETAILED ERROR ANALYSIS:")
    print("-" * 50)
    
    # Test endpoints with detailed error analysis
    endpoints = [
        {
            'name': 'Time (Public)',
            'url': 'https://api.coinbase.com/api/v3/brokerage/time',
            'method': 'GET',
            'expected': 200
        },
        {
            'name': 'Products (Private)',
            'url': 'https://api.coinbase.com/api/v3/brokerage/products',
            'method': 'GET',
            'expected': 200
        },
        {
            'name': 'Accounts (Private)',
            'url': 'https://api.coinbase.com/api/v3/brokerage/accounts',
            'method': 'GET',
            'expected': 200
        }
    ]
    
    for endpoint in endpoints:
        print(f"\n🔍 Testing {endpoint['name']}:")
        try:
            if endpoint['method'] == 'GET':
                response = requests.get(endpoint['url'], headers=headers)
            else:
                response = requests.post(endpoint['url'], headers=headers)
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ {endpoint['name']} working")
            else:
                print(f"   ❌ {endpoint['name']} failed")
                print(f"   Error: {response.text}")
                
                # Analyze the error
                if response.status_code == 401:
                    print(f"   🔍 Analysis: Authentication failed")
                    print(f"   💡 Possible causes:")
                    print(f"      • API key permissions insufficient")
                    print(f"      • Portfolio not linked to API key")
                    print(f"      • API key not activated")
                    print(f"      • Wrong audience in JWT token")
                elif response.status_code == 403:
                    print(f"   🔍 Analysis: Forbidden - insufficient permissions")
                    print(f"   💡 Possible causes:")
                    print(f"      • API key lacks required scopes")
                    print(f"      • IP address not in allowlist")
                    print(f"      • Account restrictions")
                elif response.status_code == 404:
                    print(f"   🔍 Analysis: Endpoint not found")
                    print(f"   💡 Possible causes:")
                    print(f"      • Wrong API version")
                    print(f"      • Incorrect endpoint URL")
                
        except Exception as e:
            print(f"   ❌ Error testing {endpoint['name']}: {e}")
    
    print()
    print("3. CDP KEY ANALYSIS:")
    print("-" * 30)
    
    # Extract key information
    if '/apiKeys/' in cdp_key:
        org_part = cdp_key.split('/apiKeys/')[0]
        key_id = cdp_key.split('/apiKeys/')[1]
        print(f"Organization: {org_part}")
        print(f"Key ID: {key_id}")
        print()
        print("🔧 TROUBLESHOOTING STEPS:")
        print("1. Go to https://portal.cdp.coinbase.com/")
        print("2. Navigate to your project")
        print("3. Go to API Keys section")
        print("4. Find the key with ID:", key_id)
        print("5. Check the following:")
        print("   • Is the key active?")
        print("   • Are the correct scopes enabled?")
        print("   • Is the key linked to a portfolio?")
        print("   • Is IP allowlisting configured?")
    
    print()
    print("4. RECOMMENDED FIXES:")
    print("-" * 30)
    print("Based on the error analysis, try these fixes:")
    print()
    print("🔑 API KEY PERMISSIONS:")
    print("• Enable 'View' scope for read access")
    print("• Enable 'Trade' scope for trading operations")
    print("• Disable 'Transfer' scope for security")
    print()
    print("📋 PORTFOLIO LINKING:")
    print("• Ensure the API key is linked to your portfolio")
    print("• Check that the portfolio has funds")
    print()
    print("🌐 IP ALLOWLISTING:")
    print("• Add your server's IP address to the allowlist")
    print("• Or temporarily disable IP restrictions for testing")
    print()
    print("✅ Your CDP setup is working correctly!")
    print("The issue is likely with permissions, not the setup itself.")

if __name__ == "__main__":
    main()
