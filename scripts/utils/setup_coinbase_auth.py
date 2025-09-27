#!/usr/bin/env python3
"""
Interactive Coinbase API Setup Helper

This script will help you properly configure your Coinbase API credentials.
"""

import os
import sys
import base64
from pathlib import Path


def main():
    print("=" * 60)
    print("COINBASE API CONFIGURATION HELPER")
    print("=" * 60)
    print()
    
    # Step 1: Determine environment
    print("Step 1: Choose your environment")
    print("--------------------------------")
    print("1. Sandbox (recommended for testing)")
    print("2. Production (real trading)")
    print()
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            use_sandbox = True
            print("✅ Using SANDBOX environment (safe for testing)")
            break
        elif choice == "2":
            use_sandbox = False
            print("⚠️  Using PRODUCTION environment (real money!)")
            confirm = input("Are you sure? (yes/no): ").strip().lower()
            if confirm == "yes":
                break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print()
    print("Step 2: Enter your API credentials")
    print("-----------------------------------")
    
    if use_sandbox:
        print("Get your sandbox API keys from:")
        print("https://public.sandbox.exchange.coinbase.com/")
        print("Go to Settings → API → New API Key")
    else:
        print("Get your production API keys from:")
        print("https://www.coinbase.com/settings/api")
        print("Choose 'Advanced Trade API' (not legacy Exchange)")
    
    print()
    
    # Get API key
    api_key = input("Enter your API Key: ").strip()
    if not api_key:
        print("❌ API key cannot be empty")
        return 1
    
    # Get API secret
    api_secret = input("Enter your API Secret (base64 string): ").strip()
    if not api_secret:
        print("❌ API secret cannot be empty")
        return 1
    
    # Validate base64
    try:
        base64.b64decode(api_secret, validate=True)
        print("✅ API secret is valid base64")
    except:
        print("⚠️  API secret doesn't appear to be base64 (might still work)")
    
    # Get passphrase (optional for Advanced Trade)
    print()
    print("Note: Advanced Trade API usually doesn't require a passphrase.")
    print("Leave blank if you don't have one.")
    passphrase = input("Enter passphrase (or press Enter to skip): ").strip()
    
    # Step 3: Generate .env configuration
    print()
    print("Step 3: Configuration")
    print("---------------------")
    
    env_content = f"""# Coinbase Configuration
BROKER=coinbase

# Environment
COINBASE_SANDBOX={'1' if use_sandbox else '0'}

# API Credentials
COINBASE_API_KEY={api_key}
COINBASE_API_SECRET={api_secret}
COINBASE_API_PASSPHRASE={passphrase}

# API URLs (auto-detected based on sandbox setting)
"""
    
    if use_sandbox:
        env_content += """COINBASE_API_BASE=https://api-public.sandbox.exchange.coinbase.com
COINBASE_WS_URL=wss://ws-feed-public.sandbox.exchange.coinbase.com
"""
    else:
        env_content += """# Using production URLs (default)
# COINBASE_API_BASE=https://api.coinbase.com
# COINBASE_WS_URL=wss://advanced-trade-ws.coinbase.com
"""
    
    env_content += """
# Trading Configuration
COINBASE_ENABLE_DERIVATIVES=0
COINBASE_RUN_ORDER_TESTS=0
COINBASE_ORDER_SYMBOL=BTC-USD
COINBASE_TEST_LIMIT_PRICE=10
COINBASE_TEST_QTY=0.001
"""
    
    # Ask where to save
    print("Where would you like to save the configuration?")
    print("1. .env (main file)")
    print("2. .env.local (local override, git-ignored)")
    print("3. Show config only (don't save)")
    print()
    
    save_choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if save_choice == "1":
        env_file = ".env"
    elif save_choice == "2":
        env_file = ".env.local"
    else:
        print()
        print("Configuration (copy and paste to your .env file):")
        print("-" * 60)
        print(env_content)
        print("-" * 60)
        return 0
    
    # Backup existing file
    env_path = Path(env_file)
    if env_path.exists():
        backup_path = env_path.with_suffix(f"{env_path.suffix}.backup")
        print(f"⚠️  Backing up existing {env_file} to {backup_path}")
        env_path.rename(backup_path)
    
    # Write new configuration
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Configuration saved to {env_file}")
    
    # Test the configuration
    print()
    print("Step 4: Test Connection")
    print("-----------------------")
    print("Testing your configuration...")
    print()
    
    # Set environment variables for this session
    os.environ['BROKER'] = 'coinbase'
    os.environ['COINBASE_SANDBOX'] = '1' if use_sandbox else '0'
    os.environ['COINBASE_API_KEY'] = api_key
    os.environ['COINBASE_API_SECRET'] = api_secret
    os.environ['COINBASE_API_PASSPHRASE'] = passphrase
    
    if use_sandbox:
        os.environ['COINBASE_API_BASE'] = 'https://api-public.sandbox.exchange.coinbase.com'
    
    # Import and test
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from bot_v2.orchestration.broker_factory import create_brokerage
        
        broker = create_brokerage()
        connected = broker.connect()
        
        if connected:
            print("✅ SUCCESS! Connected to Coinbase API")
            print(f"   Account ID: {broker.get_account_id()}")
            
            # Try to get some data
            try:
                products = broker.list_products()
                print(f"   Found {len(products)} available products")
            except:
                pass
            
            print()
            print("Your Coinbase API is configured correctly!")
            print("You can now run: python scripts/test_coinbase_basic.py")
        else:
            print("❌ Connection failed")
            print("Please check your API credentials and try again")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Troubleshooting tips:")
        print("1. Make sure your API key has 'View' permission")
        print("2. For sandbox, use keys from sandbox.exchange.coinbase.com")
        print("3. For production, use Advanced Trade API keys")
        print("4. Check that your credentials are copied correctly")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled.")
        sys.exit(1)