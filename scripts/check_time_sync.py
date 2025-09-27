#!/usr/bin/env python3
"""
Check time synchronization between local system and Coinbase servers.
JWT authentication fails if clocks are out of sync by more than 30 seconds.
"""

import os
import sys
import time
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_time_sync():
    """Check time sync with Coinbase servers."""
    
    print("=" * 70)
    print("TIME SYNCHRONIZATION CHECK")
    print("=" * 70)
    
    # Get local time
    local_time = time.time()
    local_dt = datetime.utcnow()
    
    print(f"\n📍 Local System Time:")
    print(f"   UTC Time: {local_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   Epoch: {int(local_time)}")
    
    # Get Coinbase server time
    print(f"\n📍 Fetching Coinbase Server Time...")
    
    try:
        # Use the public time endpoint
        url = "https://api.coinbase.com/api/v3/brokerage/time"
        req = urllib.request.Request(url, headers={"CB-VERSION": "2024-10-24"})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            server_epoch = int(data['epochSeconds'])
            server_iso = data['iso']
            
            print(f"   UTC Time: {server_iso}")
            print(f"   Epoch: {server_epoch}")
            
            # Calculate difference
            time_diff = local_time - server_epoch
            
            print(f"\n📊 Time Difference Analysis:")
            print(f"   Difference: {abs(time_diff):.1f} seconds")
            
            if abs(time_diff) < 5:
                print(f"   ✅ EXCELLENT - Clocks are synchronized")
            elif abs(time_diff) < 30:
                print(f"   ✅ OK - Within acceptable range")
            elif abs(time_diff) < 60:
                print(f"   ⚠️ WARNING - Close to JWT tolerance limit")
                print(f"      JWT tokens might fail intermittently")
            else:
                print(f"   ❌ CRITICAL - Clocks are out of sync!")
                print(f"      JWT authentication will fail")
            
            if time_diff > 0:
                print(f"   Your clock is {time_diff:.1f} seconds AHEAD of Coinbase")
            elif time_diff < 0:
                print(f"   Your clock is {abs(time_diff):.1f} seconds BEHIND Coinbase")
            
            # Test JWT with adjusted time
            if abs(time_diff) > 30:
                print("\n" + "=" * 70)
                print("TESTING WITH TIME ADJUSTMENT")
                print("=" * 70)
                test_with_time_adjustment(time_diff)
            
            return abs(time_diff)
            
    except Exception as e:
        print(f"   ❌ Failed to get server time: {e}")
        return None


def test_with_time_adjustment(time_diff):
    """Test JWT generation with time adjustment."""
    
    # Clear env vars and reload
    for key in ['COINBASE_PROD_CDP_API_KEY', 'COINBASE_PROD_CDP_PRIVATE_KEY',
                'COINBASE_CDP_API_KEY', 'COINBASE_CDP_PRIVATE_KEY']:
        if key in os.environ:
            del os.environ[key]
    
    load_dotenv()
    
    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not api_key or not private_key:
        print("❌ Missing credentials")
        return
    
    print("\n📍 Testing with time-adjusted JWT...")
    
    try:
        import jwt
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend
        
        # Load private key
        key_obj = serialization.load_pem_private_key(
            private_key.encode() if isinstance(private_key, str) else private_key,
            password=None,
            backend=default_backend()
        )
        
        # Create JWT with adjusted time
        adjusted_time = int(time.time() - time_diff)  # Adjust to match server
        
        claims = {
            "sub": api_key,
            "iss": "cdp",
            "nbf": adjusted_time,
            "exp": adjusted_time + 120,
            "uri": "GET api.coinbase.com/api/v3/brokerage/accounts"
        }
        
        import secrets
        headers = {
            "kid": api_key,
            "nonce": secrets.token_hex(),
        }
        
        token = jwt.encode(claims, key_obj, algorithm="ES256", headers=headers)
        
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        
        print(f"   ✅ Generated time-adjusted JWT")
        print(f"   Adjusted nbf: {adjusted_time}")
        print(f"   Current time: {int(time.time())}")
        
        # Test the adjusted token
        url = "https://api.coinbase.com/api/v3/brokerage/accounts"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "CB-VERSION": "2024-10-24"
            },
            method="GET"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                print(f"   ✅ SUCCESS with time-adjusted JWT! Status: {response.getcode()}")
                print(f"   → Time sync was the issue!")
        except urllib.error.HTTPError as e:
            print(f"   ❌ Still failing with status {e.code}")
            print(f"   → Time sync might not be the only issue")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")


def show_time_sync_fixes():
    """Show how to fix time sync issues."""
    
    print("\n" + "=" * 70)
    print("HOW TO FIX TIME SYNC ISSUES")
    print("=" * 70)
    
    import platform
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("\n🍎 macOS:")
        print("1. Open System Settings > General > Date & Time")
        print("2. Enable 'Set time and date automatically'")
        print("3. Or run in Terminal:")
        print("   sudo sntp -sS time.apple.com")
        print("   sudo ntpdate -u time.apple.com")
        
    elif system == "Linux":
        print("\n🐧 Linux:")
        print("1. Install NTP if needed:")
        print("   sudo apt-get install ntp")
        print("2. Sync time:")
        print("   sudo ntpdate -s time.nist.gov")
        print("3. Or with systemd:")
        print("   sudo timedatectl set-ntp true")
        
    elif system == "Windows":
        print("\n🪟 Windows:")
        print("1. Right-click clock > Adjust date/time")
        print("2. Enable 'Set time automatically'")
        print("3. Click 'Sync now'")
        print("4. Or run as Administrator:")
        print("   w32tm /resync")
    
    print("\n📍 Quick fix for testing:")
    print("   If you can't sync the system clock, we can modify")
    print("   the JWT generation code to compensate for the offset.")


if __name__ == "__main__":
    time_diff = check_time_sync()
    
    if time_diff is not None and time_diff > 30:
        show_time_sync_fixes()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if time_diff is not None:
        if time_diff < 30:
            print("✅ Time sync is good - this is not the issue")
            print("   Continue troubleshooting CDP portfolio configuration")
        else:
            print("❌ Time sync issue detected!")
            print("   Fix your system clock to resolve authentication")
    else:
        print("⚠️ Could not determine time sync status")