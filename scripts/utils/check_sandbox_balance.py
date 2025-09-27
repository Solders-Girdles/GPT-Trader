#!/usr/bin/env python
"""Check Exchange Sandbox account balance"""

import os
import time
import hmac
import hashlib
import base64
import requests

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

def get_accounts():
    """Get all accounts"""
    url = "https://api-public.sandbox.exchange.coinbase.com/accounts"
    headers = get_auth_headers('GET', '/accounts')
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("EXCHANGE SANDBOX ACCOUNT BALANCES")
    print("=" * 60)
    
    accounts = get_accounts()
    if accounts:
        print(f"Found {len(accounts)} accounts:\n")
        
        for account in accounts:
            if float(account.get('balance', 0)) > 0 or account.get('currency') in ['USD', 'BTC']:
                print(f"Currency: {account['currency']}")
                print(f"  Balance: {account.get('balance', '0')}")
                print(f"  Available: {account.get('available', '0')}")
                print(f"  Hold: {account.get('hold', '0')}")
                print()
    else:
        print("Could not retrieve accounts")