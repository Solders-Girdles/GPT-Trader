#!/usr/bin/env python3
"""
Helper to set up and test new CDP API credentials.
"""

import os
import sys
from pathlib import Path

def setup_new_key():
    """Guide through setting up new API credentials."""
    
    print("=" * 70)
    print("NEW CDP API KEY SETUP")
    print("=" * 70)
    print("\nYou've created:")
    print("✅ New portfolio: 'Derivatives'")
    print("✅ New API key with all permissions")
    print("✅ No IP whitelist (removes that variable)")
    
    print("\n" + "=" * 70)
    print("ENTER YOUR NEW CREDENTIALS")
    print("=" * 70)
    
    print("\n📋 Step 1: Enter your NEW API Key Name")
    print("Format: organizations/{org-id}/apiKeys/{new-key-id}")
    api_key = input("\nPaste API Key Name: ").strip()
    
    print("\n📋 Step 2: Enter your NEW Private Key")
    print("Include the BEGIN/END headers")
    print("Press Enter twice when done:\n")
    
    private_key_lines = []
    while True:
        line = input()
        if line == "" and private_key_lines and private_key_lines[-1] == "":
            break
        private_key_lines.append(line)
    
    # Remove trailing empty line
    if private_key_lines and private_key_lines[-1] == "":
        private_key_lines.pop()
    
    private_key = "\n".join(private_key_lines).strip()
    
    # Validate
    if not api_key.startswith("organizations/"):
        print("\n⚠️ Warning: API key format looks incorrect")
    
    if not private_key.startswith("-----BEGIN"):
        print("\n⚠️ Warning: Private key should start with -----BEGIN EC PRIVATE KEY-----")
    
    print("\n" + "=" * 70)
    print("BACKING UP OLD CREDENTIALS")
    print("=" * 70)
    
    env_path = Path(".env")
    backup_path = Path(".env.backup_old_key")
    
    if env_path.exists():
        # Backup current .env
        with open(env_path, 'r') as f:
            current_content = f.read()
        
        with open(backup_path, 'w') as f:
            f.write(current_content)
        
        print(f"✅ Backed up current .env to {backup_path}")
    
    print("\n" + "=" * 70)
    print("UPDATING .ENV FILE")
    print("=" * 70)
    
    # Read current .env
    if env_path.exists():
        with open(env_path, 'r') as f:
            lines = f.readlines()
    else:
        # Start from template
        template_path = Path(".env.template")
        if template_path.exists():
            with open(template_path, 'r') as f:
                lines = f.readlines()
        else:
            lines = []
    
    # Update the credentials
    new_lines = []
    skip_next_lines = 0
    
    for i, line in enumerate(lines):
        if skip_next_lines > 0:
            skip_next_lines -= 1
            continue
            
        if line.startswith("COINBASE_CDP_API_KEY="):
            new_lines.append(f'COINBASE_CDP_API_KEY={api_key}\n')
        elif line.startswith("COINBASE_CDP_PRIVATE_KEY="):
            # Handle multi-line private key
            new_lines.append(f'COINBASE_CDP_PRIVATE_KEY="')
            new_lines.append(private_key)
            new_lines.append('"\n')
            # Skip old private key lines
            for j in range(i+1, len(lines)):
                if '-----END EC PRIVATE KEY-----' in lines[j]:
                    skip_next_lines = j - i
                    break
        elif line.startswith("COINBASE_PROD_CDP_API_KEY="):
            new_lines.append(f'COINBASE_PROD_CDP_API_KEY={api_key}\n')
        elif line.startswith("COINBASE_PROD_CDP_PRIVATE_KEY="):
            # Handle multi-line private key
            new_lines.append(f'COINBASE_PROD_CDP_PRIVATE_KEY="')
            new_lines.append(private_key)
            new_lines.append('"\n')
            # Skip old private key lines
            for j in range(i+1, len(lines)):
                if '-----END EC PRIVATE KEY-----' in lines[j]:
                    skip_next_lines = j - i
                    break
        else:
            new_lines.append(line)
    
    # Write updated .env
    with open(env_path, 'w') as f:
        f.write(''.join(new_lines))
    
    print("✅ Updated .env with new credentials")
    
    # Extract key ID for display
    key_id = api_key.split('/')[-1] if '/' in api_key else 'unknown'
    print(f"\n📋 New API Key ID: {key_id}")
    print("📋 Portfolio: Derivatives")
    print("📋 Permissions: All enabled")
    print("📋 IP Whitelist: None (open access)")
    
    print("\n" + "=" * 70)
    print("READY TO TEST")
    print("=" * 70)
    print("\nRun these commands to test:")
    print("\n1. Quick test:")
    print("   poetry run python scripts/test_cdp_comprehensive.py")
    print("\n2. If that works, test the bot:")
    print("   poetry run perps-bot --profile dev --dev-fast")
    print("\n3. For production:")
    print("   poetry run perps-bot --profile canary --dry-run")
    
    return api_key, private_key


if __name__ == "__main__":
    api_key, private_key = setup_new_key()
    
    print("\n" + "=" * 70)
    print("IMMEDIATE TEST")
    print("=" * 70)
    
    choice = input("\nTest the new key now? (y/n): ").strip().lower()
    
    if choice == 'y':
        # Clear any environment variables
        for key in ['COINBASE_PROD_CDP_API_KEY', 'COINBASE_PROD_CDP_PRIVATE_KEY',
                    'COINBASE_CDP_API_KEY', 'COINBASE_CDP_PRIVATE_KEY']:
            if key in os.environ:
                del os.environ[key]
        
        # Test immediately
        os.system("poetry run python scripts/test_cdp_comprehensive.py")