#!/usr/bin/env python3
"""
Simple API Key Setup Helper

This script helps you set up your Coinbase API keys by:
1. Copying the .env.template to .env
2. Providing guidance on where to get your API keys
3. Validating the configuration
"""

import os
import shutil
from pathlib import Path


def main():
    print("=" * 60)
    print("COINBASE API KEY SETUP HELPER")
    print("=" * 60)
    print()
    
    # Check if .env already exists
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists!")
        choice = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if choice != 'y':
            print("Setup cancelled.")
            return
    
    # Copy template to .env
    template_file = Path(".env.template")
    if not template_file.exists():
        print("❌ .env.template not found!")
        return
    
    try:
        shutil.copy(template_file, env_file)
        print("✅ Created .env file from template")
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return
    
    print()
    print("=" * 60)
    print("NEXT STEPS TO GET YOUR API KEYS")
    print("=" * 60)
    print()
    
    print("1. CHOOSE YOUR ENVIRONMENT:")
    print("   • SANDBOX (recommended for testing):")
    print("     - Go to: https://public.sandbox.exchange.coinbase.com/")
    print("     - Sign in and go to Settings → API")
    print("     - Click 'New API Key'")
    print("     - Enable: View, Trade")
    print("     - Disable: Transfer (for safety)")
    print()
    print("   • PRODUCTION (real trading):")
    print("     - Go to: https://www.coinbase.com/settings/api")
    print("     - Choose 'Advanced Trade API'")
    print("     - Enable: View, Trade")
    print("     - Disable: Transfer (for safety)")
    print()
    
    print("2. SAVE YOUR CREDENTIALS:")
    print("   • API Key (you'll see this)")
    print("   • API Secret (base64 string - save immediately!)")
    print("   • Passphrase (if shown, often blank for Advanced Trade)")
    print()
    
    print("3. EDIT YOUR .env FILE:")
    print("   • Open .env in your text editor")
    print("   • Replace the placeholder values with your real credentials")
    print("   • Save the file")
    print()
    
    print("4. TEST YOUR CONFIGURATION:")
    print("   • Run: python scripts/test_coinbase_connection.py")
    print("   • Or: python -m bot_v2.simple_cli broker --broker coinbase --sandbox")
    print()
    
    print("⚠️  SECURITY REMINDERS:")
    print("   • NEVER commit .env files to git")
    print("   • Keep your API secret secure")
    print("   • Start with sandbox testing")
    print("   • Use minimal permissions (no transfer)")
    print()
    
    print("✅ Setup complete! Edit .env with your real API credentials.")


if __name__ == "__main__":
    main()
