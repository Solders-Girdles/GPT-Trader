#!/usr/bin/env python3
"""
Direct test of CDP authentication against Coinbase API.
Tests the exact authentication flow to identify 401 issues.
"""

import os
import sys
import json
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_authentication():
    """Test CDP authentication directly."""
    
    # Load environment
    load_dotenv()
    
    print("=" * 70)
    print("DIRECT CDP AUTHENTICATION TEST")
    print("=" * 70)
    
    # Get configuration
    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
    sandbox = os.getenv("COINBASE_SANDBOX", "0") == "1"
    auth_type = os.getenv("COINBASE_AUTH_TYPE", "JWT")
    
    print(f"\n📋 Configuration:")
    print(f"   Sandbox Mode: {sandbox}")
    print(f"   Auth Type: {auth_type}")
    print(f"   API Key: {api_key[:30]}..." if api_key else "❌ Not found")
    print(f"   Private Key: {'✅ Found' if private_key else '❌ Not found'}")
    
    if not api_key or not private_key:
        print("\n❌ Missing credentials. Set COINBASE_PROD_CDP_API_KEY and COINBASE_PROD_CDP_PRIVATE_KEY")
        return False
    
    # Import auth module
    try:
        from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
        
        # Create auth instance
        base_url = "https://api.coinbase.com"
        auth = CDPAuthV2(
            api_key_name=api_key,
            private_key_pem=private_key,
            base_host="api.coinbase.com"
        )
        
        print(f"\n✅ Created CDPAuthV2 instance")
        
        # Test 1: Simple accounts endpoint
        print("\n" + "=" * 70)
        print("TEST 1: GET /api/v3/brokerage/accounts")
        print("=" * 70)
        
        method = "GET"
        path = "/api/v3/brokerage/accounts"
        
        # Generate JWT
        try:
            jwt_token = auth.generate_jwt(method, path)
            print(f"✅ Generated JWT token")
            print(f"   Token length: {len(jwt_token)}")
            
            # Decode to inspect (without verification)
            import jwt
            decoded = jwt.decode(jwt_token, options={"verify_signature": False})
            print(f"\n📝 JWT Claims:")
            for key, value in decoded.items():
                print(f"   {key}: {value}")
            
        except Exception as e:
            print(f"❌ Failed to generate JWT: {e}")
            return False
        
        # Make the actual request
        print(f"\n🌐 Making API request...")
        url = f"{base_url}{path}"
        
        # Get headers from auth.sign method to ensure consistency
        auth_headers = auth.sign(method, path, None)
        
        headers = {
            **auth_headers,  # This already includes "Authorization: Bearer {token}"
            "CB-VERSION": "2024-10-24"
        }
        
        print(f"   URL: {url}")
        print(f"   Headers:")
        for k, v in headers.items():
            if k == "Authorization":
                print(f"     {k}: {v[:30]}...")  # Show actual header value
            else:
                print(f"     {k}: {v}")
        
        # Make request
        req = urllib.request.Request(url, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.getcode()
                data = response.read().decode('utf-8')
                
                print(f"\n✅ Request successful!")
                print(f"   Status: {status}")
                
                # Parse response
                try:
                    parsed = json.loads(data)
                    if "accounts" in parsed:
                        print(f"   Found {len(parsed['accounts'])} accounts")
                        for acc in parsed['accounts'][:3]:  # Show first 3
                            print(f"     - {acc.get('name', 'Unknown')} ({acc.get('currency', 'N/A')})")
                    else:
                        print(f"   Response: {json.dumps(parsed, indent=2)[:500]}")
                except:
                    print(f"   Raw response: {data[:500]}")
                    
                return True
                
        except urllib.error.HTTPError as e:
            status = e.code
            error_content = e.read()  # Read only once
            error_body = error_content.decode('utf-8') if error_content else ""
            
            print(f"\n❌ Request failed with status {status}")
            print(f"   Raw response: {error_body}")
            
            # Parse error
            try:
                if error_body:
                    error_data = json.loads(error_body)
                    print(f"   Error JSON: {json.dumps(error_data, indent=2)}")
                else:
                    print("   No error body returned")
                
                # Specific error analysis
                if status == 401:
                    print("\n🔍 401 Authentication Error Analysis:")
                    
                    error_msg = error_data.get('message', '').lower()
                    error_code = error_data.get('error', '')
                    
                    if 'invalid api key' in error_msg:
                        print("   ❌ API key format is invalid")
                        print("   ✅ Fix: Ensure key format is 'organizations/{org_id}/apiKeys/{key_id}'")
                    elif 'signature' in error_msg or 'jwt' in error_msg:
                        print("   ❌ JWT signature verification failed")
                        print("   ✅ Fix: Check private key matches the API key")
                    elif 'expired' in error_msg:
                        print("   ❌ JWT token expired")
                        print("   ✅ Fix: Check system clock is synchronized")
                    elif 'permission' in error_msg:
                        print("   ❌ API key lacks required permissions")
                        print("   ✅ Fix: Add 'view' permission in CDP console")
                    elif 'ip' in error_msg or 'whitelist' in error_msg:
                        print("   ❌ IP address not whitelisted")
                        print("   ✅ Fix: Add your IP to the whitelist in CDP console")
                        print("   💡 Your current IP can be found at: https://api.ipify.org")
                    else:
                        print(f"   ❓ Unknown auth error: {error_msg}")
                        
            except:
                print(f"   Raw error: {error_body[:500]}")
            
            return False
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False
    
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("   Ensure pyjwt and cryptography are installed:")
        print("   poetry install")
        return False
    
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_ip_address():
    """Check and display current IP address."""
    print("\n" + "=" * 70)
    print("IP ADDRESS CHECK")
    print("=" * 70)
    
    try:
        import urllib.request
        with urllib.request.urlopen('https://api.ipify.org', timeout=5) as response:
            ip = response.read().decode('utf-8')
            print(f"🌐 Your current IP address: {ip}")
            print(f"   Add this to your CDP API key whitelist if not already done")
            return ip
    except Exception as e:
        print(f"❌ Could not determine IP: {e}")
        return None


def main():
    """Run the authentication test."""
    print("\n" + "=" * 70)
    print("COINBASE CDP AUTHENTICATION VERIFICATION")
    print("=" * 70)
    
    # Check IP first
    ip = check_ip_address()
    
    # Test authentication
    success = test_authentication()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if success:
        print("✅ Authentication is working correctly!")
        print("\nNext steps:")
        print("1. Try running: poetry run perps-bot --profile dev --dev-fast")
        print("2. If that works: poetry run perps-bot --profile canary --dry-run")
    else:
        print("❌ Authentication failed")
        print("\nTroubleshooting steps:")
        print("1. Verify your IP is whitelisted in CDP console")
        if ip:
            print(f"   Your IP: {ip}")
        print("2. Verify API key has 'view' permissions")
        print("3. Check that private key matches the API key")
        print("4. Ensure .env has correct format (see scripts/test_cdp_auth.py)")
        print("5. Try regenerating the API key if issues persist")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())