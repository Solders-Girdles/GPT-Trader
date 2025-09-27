#!/usr/bin/env python3
"""
Add Legacy API Credentials

This script helps you add legacy API credentials to your existing .env file
while preserving your current CDP configuration.
"""

import os
from pathlib import Path

def main():
    print("=" * 60)
    print("ADD LEGACY API CREDENTIALS")
    print("=" * 60)
    print()
    
    print("This will add legacy API credentials to your .env file.")
    print("Your existing CDP configuration will be preserved.")
    print()
    
    # Get legacy credentials
    print("Enter your LEGACY API credentials (from https://www.coinbase.com/settings/api):")
    print()
    
    api_key = input("API Key: ").strip()
    if not api_key:
        print("❌ API key cannot be empty")
        return
    
    api_secret = input("API Secret (base64 string): ").strip()
    if not api_secret:
        print("❌ API secret cannot be empty")
        return
    
    passphrase = input("Passphrase (often blank for Advanced Trade, press Enter if none): ").strip()
    
    # Read current .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return
    
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Find where to insert the legacy credentials
    insert_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("# LEGACY API CREDENTIALS"):
            insert_index = i + 1
            break
    
    if insert_index is None:
        # Add section if it doesn't exist
        insert_index = len(lines)
        lines.append("\n# =============================================================================\n")
        lines.append("# LEGACY API CREDENTIALS (HMAC Authentication)\n")
        lines.append("# =============================================================================\n")
    
    # Insert the credentials
    legacy_creds = [
        f"COINBASE_API_KEY={api_key}\n",
        f"COINBASE_API_SECRET={api_secret}\n",
        f"COINBASE_API_PASSPHRASE={passphrase}\n"
    ]
    
    lines[insert_index:insert_index] = legacy_creds
    
    # Write back to file
    with open(env_file, 'w') as f:
        f.writelines(lines)
    
    print()
    print("✅ Legacy credentials added to .env file")
    print()
    print("📋 NEXT STEPS:")
    print("1. Test your configuration:")
    print("   python scripts/test_coinbase_connection.py")
    print()
    print("2. For sandbox testing, set COINBASE_SANDBOX=1 in .env")
    print()
    print("3. The system will now use legacy credentials for reliable authentication")

if __name__ == "__main__":
    main()
