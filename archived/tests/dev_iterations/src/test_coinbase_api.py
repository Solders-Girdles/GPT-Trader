#!/usr/bin/env python3
import os
import requests
import base64
import hmac
import hashlib
import time
import json

def test_coinbase_api():
    # Load environment variables
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    base_url = os.getenv('COINBASE_API_BASE', 'https://api.exchange.coinbase.com')
    
    print(f"Testing Coinbase API with:")
    print(f"API Key: {api_key[:20]}...")
    print(f"Base URL: {base_url}")
    print()
    
    # Test a simple endpoint that doesn't require authentication
    try:
        response = requests.get(f"{base_url}/api/v3/brokerage/products")
        print(f"Public products endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data.get('products', []))} products")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error testing public endpoint: {e}")
    
    print()
    print("Note: If the public endpoint works but authenticated calls don't,")
    print("the issue is likely with your API key permissions or authentication.")

if __name__ == "__main__":
    test_coinbase_api()

