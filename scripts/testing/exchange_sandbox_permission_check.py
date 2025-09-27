#!/usr/bin/env python
"""
Exchange Sandbox Permission Check - Phase 2 Validation
Treats "Insufficient funds" as permission SUCCESS
"""

import os
import sys
import time
import hmac
import hashlib
import base64
import json
import requests
from datetime import datetime

def get_auth_headers(method, path, body=""):
    """Generate HMAC authentication headers"""
    api_key = os.environ.get('COINBASE_API_KEY')
    api_secret = os.environ.get('COINBASE_API_SECRET')
    api_passphrase = os.environ.get('COINBASE_API_PASSPHRASE')
    
    timestamp = str(int(time.time()))
    message = timestamp + method + path + body
    
    hmac_key = base64.b64decode(api_secret)
    signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
    signature_b64 = base64.b64encode(signature.digest()).decode()
    
    return {
        'CB-ACCESS-KEY': api_key,
        'CB-ACCESS-SIGN': signature_b64,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-PASSPHRASE': api_passphrase,
        'Content-Type': 'application/json'
    }

def check_view_permission():
    """Check VIEW permission by getting accounts"""
    url = "https://api-public.sandbox.exchange.coinbase.com/accounts"
    headers = get_auth_headers('GET', '/accounts')
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return True, "Account data retrieved"
    elif response.status_code == 401:
        return False, "Unauthorized - invalid credentials"
    elif response.status_code == 403:
        return False, "Forbidden - missing VIEW permission"
    else:
        return False, f"Unexpected error: {response.status_code}"

def check_trade_permission():
    """Check TRADE permission by attempting order placement"""
    url = "https://api-public.sandbox.exchange.coinbase.com/orders"
    
    # Minimal order that will fail due to insufficient funds
    order_data = {
        'product_id': 'BTC-USD',
        'side': 'buy',
        'type': 'limit',
        'size': '0.0001',
        'price': '10000.00',
        'time_in_force': 'GTC'
    }
    
    body = json.dumps(order_data)
    headers = get_auth_headers('POST', '/orders', body)
    
    response = requests.post(url, headers=headers, data=body)
    
    if response.status_code == 200:
        # Order placed (unlikely with $0 balance)
        return True, "Order placed successfully", "LIFECYCLE_READY"
    elif response.status_code == 400:
        # Check the error message
        try:
            error_msg = response.json().get('message', '').lower()
            if 'insufficient funds' in error_msg:
                return True, "TRADE permission confirmed (insufficient funds)", "LIFECYCLE_BLOCKED"
            elif 'notional is too small' in error_msg:
                return True, "TRADE permission confirmed (min notional)", "LIFECYCLE_BLOCKED"
            else:
                return False, f"Bad request: {error_msg}", "UNKNOWN"
        except:
            return False, f"Bad request: {response.text}", "UNKNOWN"
    elif response.status_code == 401:
        return False, "Unauthorized - invalid credentials", "AUTH_FAILED"
    elif response.status_code == 403:
        return False, "Forbidden - missing TRADE permission", "PERMISSION_DENIED"
    else:
        return False, f"Unexpected error: {response.status_code}", "UNKNOWN"

def main():
    """Run permission checks"""
    print("=" * 60)
    print("EXCHANGE SANDBOX PERMISSION CHECK")
    print("=" * 60)
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print()
    
    # Verify environment
    api_mode = os.environ.get('COINBASE_API_MODE')
    sandbox = os.environ.get('COINBASE_SANDBOX')
    
    print("📋 Environment Check:")
    print(f"   API_MODE: {api_mode} {'✅' if api_mode == 'exchange' else '❌'}")
    print(f"   SANDBOX: {sandbox} {'✅' if sandbox == '1' else '❌'}")
    print()
    
    if api_mode != 'exchange' or sandbox != '1':
        print("❌ Environment not configured for Exchange Sandbox")
        return 1
    
    # Check VIEW permission
    print("🔍 Checking VIEW Permission...")
    view_ok, view_msg = check_view_permission()
    print(f"   Result: {'✅ PASS' if view_ok else '❌ FAIL'}")
    print(f"   Details: {view_msg}")
    print()
    
    # Check TRADE permission
    print("📝 Checking TRADE Permission...")
    trade_ok, trade_msg, lifecycle_status = check_trade_permission()
    print(f"   Result: {'✅ PASS' if trade_ok else '❌ FAIL'}")
    print(f"   Details: {trade_msg}")
    print(f"   Lifecycle: {lifecycle_status}")
    print()
    
    # Summary
    print("=" * 60)
    print("📊 PERMISSION CHECK SUMMARY")
    print("=" * 60)
    
    if view_ok and trade_ok:
        print("✅ ALL PERMISSIONS VALIDATED")
        print()
        print("Permission Path: PASS")
        print("- VIEW: ✅ Granted")
        print("- TRADE: ✅ Granted")
        print()
        
        if lifecycle_status == "LIFECYCLE_BLOCKED":
            print("⚠️ Order Lifecycle: BLOCKED")
            print("   Reason: Insufficient funds in sandbox account")
            print("   This is expected with $0 balance")
            print()
            print("💡 Next Steps:")
            print("   Option A: Fund sandbox account and re-test")
            print("   Option B: Proceed to production canary with guards")
        else:
            print("✅ Order Lifecycle: READY")
            print("   Orders can be placed and managed")
        
        return 0  # Exit success - permissions are valid
    else:
        print("❌ PERMISSION CHECK FAILED")
        print()
        if not view_ok:
            print("- VIEW: ❌ Failed")
            print(f"  Reason: {view_msg}")
        if not trade_ok:
            print("- TRADE: ❌ Failed")
            print(f"  Reason: {trade_msg}")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Verify API credentials are correct")
        print("   2. Check API key has required permissions")
        print("   3. Ensure IP is whitelisted")
        print("   4. Confirm using Exchange Sandbox keys (not production)")
        
        return 1  # Exit failure - permissions invalid

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
