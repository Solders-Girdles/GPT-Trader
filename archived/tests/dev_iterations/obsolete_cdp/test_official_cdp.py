#!/usr/bin/env python3
"""
Official CDP Authentication Test

This script uses the official Coinbase JWT generator to test CDP authentication
following the exact guidelines for Advanced Trade API.
"""

import os
import pytest
import sys
import requests
import json
from pathlib import Path

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.getenv('COINBASE_CDP_API_KEY') and os.getenv('COINBASE_CDP_PRIVATE_KEY')),
        reason='CDP credentials not configured',
    ),
]

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

def test_with_manual_jwt():
    """Test with manually constructed JWT following the guidelines."""
    import jwt
    from datetime import datetime, timedelta
    
    # Load environment variables
    env_vars = load_env_file('.env')
    
    cdp_key = env_vars.get('COINBASE_CDP_API_KEY', '')
    cdp_private = env_vars.get('COINBASE_CDP_PRIVATE_KEY', '')
    
    if not cdp_key or not cdp_private:
        print("❌ CDP credentials are missing")
        return
    
    print("1. CHECKING KEY FORMAT:")
    print("-" * 30)
    print(f"API Key: {cdp_key}")
    print(f"Private Key Format: {'PEM' if cdp_private.startswith('-----BEGIN') else 'Unknown'}")
    
    # Check if key name format is correct
    if not cdp_key.startswith('organizations/') or '/apiKeys/' not in cdp_key:
        print("❌ API key format incorrect. Should be: organizations/{org_id}/apiKeys/{key_id}")
        return
    
    print("✅ API key format is correct")
    
    print()
    print("2. TESTING KEY PERMISSIONS ENDPOINT:")
    print("-" * 40)
    
    # Test the key_permissions endpoint (recommended test)
    path = "/api/v3/brokerage/key_permissions"
    
    # Create JWT payload
    now = datetime.utcnow()
    payload = {
        'sub': cdp_key,
        'iss': cdp_key,
        'nbf': int(now.timestamp()),
        'exp': int((now + timedelta(minutes=2)).timestamp()),  # 2-minute TTL
        'aud': 'retail_rest_api_pro'
    }
    
    try:
        # Sign the JWT with the PEM private key
        token = jwt.encode(payload, cdp_private, algorithm='ES256')
        print("✅ JWT token generated successfully")
        print(f"Token: {token[:50]}...")
        
        # Make the request
        url = f"https://api.coinbase.com{path}"
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        print(f"Requesting: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Key permissions endpoint working!")
            print("Permissions:", json.dumps(data, indent=2))
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_with_official_sdk():
    """Test using the official Coinbase SDK if available."""
    try:
        from coinbase import jwt_generator
        print("✅ Official Coinbase SDK available")
        
        # Load environment variables
        env_vars = load_env_file('.env')
        
        cdp_key = env_vars.get('COINBASE_CDP_API_KEY', '')
        cdp_private = env_vars.get('COINBASE_CDP_PRIVATE_KEY', '')
        
        if not cdp_key or not cdp_private:
            print("❌ CDP credentials are missing")
            return
        
        print()
        print("3. TESTING WITH OFFICIAL SDK:")
        print("-" * 30)
        
        # Test key_permissions endpoint
        path = "/api/v3/brokerage/key_permissions"
        jwt_uri = jwt_generator.format_jwt_uri("GET", path)
        jwt_token = jwt_generator.build_rest_jwt(jwt_uri, cdp_key, cdp_private)
        
        print("✅ JWT generated with official SDK")
        
        # Make the request
        url = f"https://api.coinbase.com{path}"
        headers = {"Authorization": f"Bearer {jwt_token}"}
        
        response = requests.get(url, headers=headers, timeout=15)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Official SDK test successful!")
            print("Permissions:", json.dumps(data, indent=2))
        else:
            print(f"❌ Failed: {response.text}")
            
    except ImportError:
        print("⚠️  Official Coinbase SDK not available")
        print("Install with: pip install coinbase-advanced-py")
    except Exception as e:
        print(f"❌ Error with official SDK: {e}")

def main():
    print("=" * 60)
    print("OFFICIAL CDP AUTHENTICATION TEST")
    print("=" * 60)
    print()
    
    print("Testing CDP authentication following the official guidelines:")
    print("• Using ES256 algorithm")
    print("• 2-minute JWT TTL")
    print("• Correct key format")
    print("• Proper endpoint paths")
    print()
    
    # Test with manual JWT
    test_with_manual_jwt()
    
    print()
    
    # Test with official SDK if available
    test_with_official_sdk()
    
    print()
    print("4. TROUBLESHOOTING CHECKLIST:")
    print("-" * 40)
    print("If tests fail, check these common issues:")
    print()
    print("🔑 Key Algorithm:")
    print("• Ensure key is ES256 (ECDSA), not Ed25519")
    print("• Check in CDP portal: https://portal.cdp.coinbase.com/")
    print()
    print("⏰ JWT Timing:")
    print("• JWT expires after 2 minutes")
    print("• Generate fresh JWT for each request")
    print("• Check system clock accuracy")
    print()
    print("🔗 Endpoint URLs:")
    print("• Use: https://api.coinbase.com/api/v3/brokerage/...")
    print("• Don't mix with Exchange/Pro endpoints")
    print()
    print("📋 Permissions:")
    print("• Check key permissions in CDP portal")
    print("• Ensure portfolio is linked")
    print("• Verify IP allowlist settings")
    print()
    print("🎯 Next Steps:")
    print("• If tests pass: Your CDP setup is working!")
    print("• If tests fail: Check the specific error message above")

if __name__ == "__main__":
    main()

