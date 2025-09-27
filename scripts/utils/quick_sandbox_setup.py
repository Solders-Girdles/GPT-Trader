#!/usr/bin/env python3
"""
Quick Coinbase Sandbox Setup
"""

import os
import sys
from pathlib import Path

# The sandbox secret you provided
SANDBOX_SECRET = "BZN0AzzkvrOyjBz8xs3as+jZcCsZsHI2jVHiyTrFnQpc+YfOqKgZKF1Hu5dfRsO5bVCdgqnasMsYl4rlA0HFUg=="

print("=" * 60)
print("COINBASE SANDBOX SETUP")
print("=" * 60)
print()
print("You provided the Sandbox API Secret:")
print(f"  {SANDBOX_SECRET[:20]}...")
print()
print("Now I need your Sandbox API Key.")
print()
print("To get it:")
print("1. Go to https://public.sandbox.exchange.coinbase.com/")
print("2. Sign in and go to Settings → API")
print("3. Find the API Key that matches this secret")
print("   (or create a new key pair)")
print()

api_key = input("Enter your Sandbox API Key: ").strip()

if not api_key:
    print("❌ API Key cannot be empty")
    sys.exit(1)

print()
print("Configuration to add to your .env file:")
print("-" * 60)

config = f"""# Coinbase Sandbox Configuration
BROKER=coinbase
COINBASE_SANDBOX=1

# Sandbox API Credentials  
COINBASE_API_KEY={api_key}
COINBASE_API_SECRET={SANDBOX_SECRET}
COINBASE_API_PASSPHRASE=

# Sandbox URLs
COINBASE_API_BASE=https://api-public.sandbox.exchange.coinbase.com
COINBASE_WS_URL=wss://ws-feed-public.sandbox.exchange.coinbase.com

# Test Settings
COINBASE_ENABLE_DERIVATIVES=0
COINBASE_RUN_ORDER_TESTS=0
COINBASE_ORDER_SYMBOL=BTC-USD
COINBASE_TEST_LIMIT_PRICE=10
COINBASE_TEST_QTY=0.001"""

print(config)
print("-" * 60)

# Ask if they want to save
save = input("\nSave this configuration? (y/n): ").strip().lower()

if save == 'y':
    # Backup existing .env
    env_file = Path(".env")
    if env_file.exists():
        backup = Path(".env.backup_sandbox")
        import shutil
        shutil.copy(env_file, backup)
        print(f"✅ Backed up existing .env to {backup}")
    
    # Write new config
    with open(".env", "w") as f:
        f.write(config)
    print("✅ Configuration saved to .env")
    
    # Test connection
    print("\nTesting connection...")
    os.environ.update({
        'BROKER': 'coinbase',
        'COINBASE_SANDBOX': '1',
        'COINBASE_API_KEY': api_key,
        'COINBASE_API_SECRET': SANDBOX_SECRET,
        'COINBASE_API_PASSPHRASE': '',
        'COINBASE_API_BASE': 'https://api-public.sandbox.exchange.coinbase.com'
    })
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from bot_v2.orchestration.broker_factory import create_brokerage
        
        broker = create_brokerage()
        connected = broker.connect()
        
        if connected:
            print("✅ SUCCESS! Connected to Coinbase Sandbox")
            print(f"   Account ID: {broker.get_account_id()}")
            
            try:
                products = broker.list_products()
                print(f"   Found {len(products)} products")
                balances = broker.list_balances()
                print(f"   Found {len(balances)} balance entries")
            except Exception as e:
                print(f"   (Could not fetch all data: {e})")
            
            print("\n🎉 Your sandbox is configured and working!")
            print("Run: python scripts/test_coinbase_basic.py")
        else:
            print("❌ Connection failed - check your credentials")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease verify your sandbox API key is correct")
else:
    print("\nConfiguration not saved. Copy the config above to your .env file.")