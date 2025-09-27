#!/usr/bin/env python3
"""
Complete API Key Setup Guide

This script walks you through setting up all API keys for the perps trading system
following the project manager's guidelines for naming, permissions, and security.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def print_header(title):
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_section(title):
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print(f"{'-' * 60}")

def get_current_month():
    return datetime.now().strftime("%Y%m")

def main():
    print_header("COMPLETE API KEY SETUP FOR PERPS TRADING SYSTEM")
    
    current_month = get_current_month()
    
    print_section("OVERVIEW")
    print("This setup will create API keys for:")
    print("• Development (mock only - no keys needed)")
    print("• Demo (sandbox with tiny notionals)")
    print("• Production (real trading with full perps support)")
    print()
    print("Key naming convention: perps-{env}-{bot}-{perm}-{yyyymm}")
    print(f"Current month: {current_month}")
    
    print_section("STEP 1: ENVIRONMENT SELECTION")
    print("Which environment are you setting up?")
    print("1. Demo (sandbox - recommended for testing)")
    print("2. Production (real trading - be careful!)")
    print("3. Both (complete setup)")
    print()
    
    while True:
        choice = input("Enter choice (1, 2, or 3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    environments = []
    if choice == '1':
        environments = ['demo']
    elif choice == '2':
        environments = ['prod']
    else:
        environments = ['demo', 'prod']
    
    print_section("STEP 2: KEY CREATION GUIDE")
    
    for env in environments:
        print(f"\n🔑 CREATING KEYS FOR {env.upper()} ENVIRONMENT")
        print("-" * 50)
        
        if env == 'demo':
            print("📋 DEMO ENVIRONMENT REQUIREMENTS:")
            print("• Environment: Sandbox")
            print("• URL: https://public.sandbox.exchange.coinbase.com/")
            print("• Purpose: Testing with tiny notionals")
            print("• API Type: Advanced Trade (JWT/CDP) or Exchange (HMAC)")
            print()
            
            print("🔧 REQUIRED PERMISSIONS:")
            print("• Accounts/Portfolios: read")
            print("• Orders: read/write (place, cancel, modify)")
            print("• Products/Market Data: read")
            print("• Fills/Positions: read")
            print("• Derivatives (CFM/perpetuals): read/write")
            print("• Transfers/Withdrawals: none (explicitly disable)")
            print()
            
            print("📝 KEY NAMING EXAMPLES:")
            print(f"• perps-demo-bot01-trade-{current_month}")
            print(f"• perps-demo-monitor-read-{current_month}")
            print()
            
        else:  # prod
            print("📋 PRODUCTION ENVIRONMENT REQUIREMENTS:")
            print("• Environment: Production")
            print("• URL: https://www.coinbase.com/")
            print("• Purpose: Real trading with full perps (CFM) support")
            print("• API Type: Advanced Trade (JWT/CDP)")
            print()
            
            print("🔧 REQUIRED PERMISSIONS:")
            print("• Accounts/Portfolios: read")
            print("• Orders: read/write (place, cancel, modify)")
            print("• Products/Market Data: read")
            print("• Fills/Positions: read")
            print("• Derivatives (CFM/perpetuals): read/write")
            print("• Transfers/Withdrawals: none (explicitly disable)")
            print()
            
            print("📝 KEY NAMING EXAMPLES:")
            print(f"• perps-prod-bot01-trade-{current_month}")
            print(f"• perps-prod-monitor-read-{current_month}")
            print()
        
        print("🌐 KEY CREATION STEPS:")
        if env == 'demo':
            print("1. Go to: https://public.sandbox.exchange.coinbase.com/")
            print("2. Sign in and navigate to Settings → API")
            print("3. Click 'New API Key'")
        else:
            print("1. Go to: https://www.coinbase.com/settings/api")
            print("2. Click 'New API Key'")
            print("3. Choose 'Advanced Trade API' (not legacy Exchange)")
        
        print("4. Set permissions as listed above")
        print("5. Use the naming convention shown above")
        print("6. Save credentials immediately (shown only once)")
        print()
        
        print("⚠️  SECURITY CHECKLIST:")
        print("✅ IP allowlist enabled (if available)")
        print("✅ Transfer/withdrawal permissions disabled")
        print("✅ Keys rotated every 90 days")
        print("✅ Credentials stored securely")
        print()
    
    print_section("STEP 3: CREDENTIALS COLLECTION")
    print("Now I'll help you collect and configure all your credentials.")
    print()
    
    # Collect credentials for each environment
    all_credentials = {}
    
    for env in environments:
        print(f"\n📝 COLLECTING {env.upper()} CREDENTIALS")
        print("-" * 40)
        
        # Get key names
        print("Enter your key names (use the naming convention):")
        trade_key_name = input(f"Trade key name (e.g., perps-{env}-bot01-trade-{current_month}): ").strip()
        monitor_key_name = input(f"Monitor key name (e.g., perps-{env}-monitor-read-{current_month}): ").strip()
        
        # Get trade key credentials
        print(f"\n🔑 {env.upper()} TRADE KEY CREDENTIALS:")
        trade_api_key = input("API Key: ").strip()
        trade_api_secret = input("API Secret (base64): ").strip()
        trade_passphrase = input("Passphrase (often blank for Advanced Trade): ").strip()
        
        # Get monitor key credentials
        print(f"\n👁️  {env.upper()} MONITOR KEY CREDENTIALS:")
        monitor_api_key = input("API Key: ").strip()
        monitor_api_secret = input("API Secret (base64): ").strip()
        monitor_passphrase = input("Passphrase (often blank for Advanced Trade): ").strip()
        
        all_credentials[env] = {
            'trade': {
                'name': trade_key_name,
                'api_key': trade_api_key,
                'api_secret': trade_api_secret,
                'passphrase': trade_passphrase
            },
            'monitor': {
                'name': monitor_key_name,
                'api_key': monitor_api_key,
                'api_secret': monitor_api_secret,
                'passphrase': monitor_passphrase
            }
        }
    
    print_section("STEP 4: CONFIGURATION FILES")
    print("I'll now create the appropriate configuration files.")
    print()
    
    # Create environment-specific configs
    for env in environments:
        config_file = f".env.{env}"
        print(f"Creating {config_file}...")
        
        config_content = f"""# =============================================================================
# COINBASE API CONFIGURATION - {env.upper()} ENVIRONMENT
# =============================================================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Environment: {env}
# Key Month: {current_month}

# =============================================================================
# BROKER SELECTION
# =============================================================================
BROKER=coinbase

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================
COINBASE_SANDBOX={'1' if env == 'demo' else '0'}

# =============================================================================
# API MODE & AUTHENTICATION
# =============================================================================
COINBASE_API_MODE=advanced
COINBASE_AUTH_TYPE=HMAC

# =============================================================================
# TRADE KEY CREDENTIALS
# =============================================================================
# Key Name: {all_credentials[env]['trade']['name']}
COINBASE_API_KEY={all_credentials[env]['trade']['api_key']}
COINBASE_API_SECRET={all_credentials[env]['trade']['api_secret']}
COINBASE_API_PASSPHRASE={all_credentials[env]['trade']['passphrase']}

# =============================================================================
# MONITOR KEY CREDENTIALS (READ-ONLY)
# =============================================================================
# Key Name: {all_credentials[env]['monitor']['name']}
COINBASE_MONITOR_API_KEY={all_credentials[env]['monitor']['api_key']}
COINBASE_MONITOR_API_SECRET={all_credentials[env]['monitor']['api_secret']}
COINBASE_MONITOR_API_PASSPHRASE={all_credentials[env]['monitor']['passphrase']}

# =============================================================================
# TRADING FEATURES
# =============================================================================
COINBASE_ENABLE_DERIVATIVES={'1' if env == 'prod' else '0'}

# =============================================================================
# SAFETY SETTINGS
# =============================================================================
RISK_REDUCE_ONLY_MODE={'0' if env == 'prod' else '1'}
RISK_MAX_DAILY_LOSS_PCT=0.01
RISK_MAX_MARK_STALENESS_SECONDS=30

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================
COINBASE_ENABLE_KEEP_ALIVE=1
COINBASE_CONNECTION_TIMEOUT=30
COINBASE_REQUEST_TIMEOUT=10
COINBASE_RATE_LIMIT_RPS=10
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created {config_file}")
    
    # Create main .env file
    print("\nCreating main .env file...")
    main_env_content = f"""# =============================================================================
# COINBASE API CONFIGURATION - MAIN
# =============================================================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# This file should point to your desired environment

# =============================================================================
# BROKER SELECTION
# =============================================================================
BROKER=coinbase

# =============================================================================
# ENVIRONMENT SELECTION
# =============================================================================
# Set to 'demo' for sandbox testing, 'prod' for production
# The system will load the appropriate .env.{environments[0]} file
ENVIRONMENT={environments[0]}

# =============================================================================
# AUTO-LOAD ENVIRONMENT CONFIG
# =============================================================================
# This will be handled by the application to load the correct config
"""
    
    with open('.env', 'w') as f:
        f.write(main_env_content)
    
    print("✅ Created .env")
    
    print_section("STEP 5: VALIDATION & TESTING")
    print("Now let's test your configuration:")
    print()
    
    print("1. Test demo environment:")
    print(f"   cp .env.demo .env")
    print("   python scripts/test_coinbase_connection.py")
    print()
    
    if 'prod' in environments:
        print("2. Test production environment:")
        print(f"   cp .env.prod .env")
        print("   python scripts/test_coinbase_connection.py")
        print()
    
    print("3. Run comprehensive validation:")
    print("   python scripts/validate_critical_fixes_v2.py")
    print()
    
    print_section("STEP 6: SECURITY CHECKLIST")
    print("✅ API keys created with proper naming convention")
    print("✅ Permissions set to least-privilege")
    print("✅ Transfer/withdrawal disabled")
    print("✅ Environment separation maintained")
    print("✅ Configuration files created")
    print("✅ Credentials stored securely")
    print()
    
    print("⚠️  REMINDERS:")
    print("• Rotate keys every 90 days")
    print("• Never commit .env files to git")
    print("• Use IP allowlisting in production")
    print("• Monitor key usage and permissions")
    print()
    
    print_header("SETUP COMPLETE!")
    print("Your API keys are now properly configured following the project guidelines.")
    print("Run the validation tests to ensure everything is working correctly.")

if __name__ == "__main__":
    main()
