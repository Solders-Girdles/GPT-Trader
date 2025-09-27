#!/usr/bin/env python3
"""
Test CDP API Connection

Simple script to test the CDP API connection with the current configuration.
"""

import os
import pytest
import sys
import requests
import json
from pathlib import Path

pytestmark = pytest.mark.integration

def load_env_file(env_file):
    """Load environment variables from .env file."""
    if not Path(env_file).exists():
        return {}
    
    env_vars = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value
    return env_vars

def main():
    print("=" * 60)
    print("CDP API CONNECTION TEST")
    print("=" * 60)
    print()
    
    # Load environment variables
    env_vars = load_env_file('.env')
    
    print("1. CURRENT CONFIGURATION:")
    print("-" * 30)
    
    broker = env_vars.get('BROKER', 'Not set')
    sandbox = env_vars.get('COINBASE_SANDBOX', 'Not set')
    api_mode = env_vars.get('COINBASE_API_MODE', 'Not set')
    auth_type = env_vars.get('COINBASE_AUTH_TYPE', 'Not set')
    cdp_key = env_vars.get('COINBASE_CDP_API_KEY', '')
    cdp_private = env_vars.get('COINBASE_CDP_PRIVATE_KEY', '')
    
    print(f"Broker: {broker}")
    print(f"Sandbox: {sandbox}")
    print(f"API Mode: {api_mode}")
    print(f"Auth Type: {auth_type}")
    print(f"CDP Key: {'Present' if cdp_key else 'Not set'}")
    print(f"CDP Private Key: {'Present' if cdp_private else 'Not set'}")
    
    print()
    print("2. TESTING PUBLIC ENDPOINT:")
    print("-" * 30)
    
    # Test public endpoint (no auth required)
    try:
        url = "https://api.coinbase.com/api/v3/brokerage/products"
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Public endpoint working - Found {len(data.get('products', []))} products")
        else:
            print(f"❌ Public endpoint failed: {response.text}")
    except Exception as e:
        print(f"❌ Error testing public endpoint: {e}")
    
    print()
    print("3. TESTING TIME ENDPOINT (CDP Auth):")
    print("-" * 30)
    
    # Test time endpoint (requires CDP auth)
    try:
        url = "https://api.coinbase.com/api/v3/brokerage/time"
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Time endpoint working - Server time: {data.get('iso', 'N/A')}")
        else:
            print(f"❌ Time endpoint failed: {response.text}")
    except Exception as e:
        print(f"❌ Error testing time endpoint: {e}")
    
    print()
    print("4. TESTING PRODUCTS ENDPOINT (CDP Auth):")
    print("-" * 30)
    
    # Test products endpoint (requires CDP auth)
    try:
        url = "https://api.coinbase.com/api/v3/brokerage/products"
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Products endpoint working - Found {len(data.get('products', []))} products")
        else:
            print(f"❌ Products endpoint failed: {response.text}")
    except Exception as e:
        print(f"❌ Error testing products endpoint: {e}")
    
    print()
    print("5. SUMMARY:")
    print("-" * 30)
    
    if cdp_key and cdp_private:
        print("✅ CDP credentials are configured")
    else:
        print("❌ CDP credentials are missing")
    
    if broker == 'coinbase':
        print("✅ Broker is set to coinbase")
    else:
        print("❌ Broker is not set to coinbase")
    
    if sandbox == '0':
        print("✅ Production mode configured")
    else:
        print("⚠️  Sandbox mode configured")
    
    print()
    print("📋 RECOMMENDATIONS:")
    print("-" * 30)
    
    if not cdp_key or not cdp_private:
        print("• Add CDP credentials to .env file")
    
    print("• Test with the main application:")
    print("  python -m src.bot_v2.simple_cli broker --broker coinbase")
    print()
    print("• For comprehensive testing:")
    print("  python scripts/validate_critical_fixes_v2.py")

if __name__ == "__main__":
    main()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.getenv('COINBASE_CDP_API_KEY') and os.getenv('COINBASE_CDP_PRIVATE_KEY')),
        reason='CDP credentials not configured',
    ),
]

