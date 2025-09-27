#!/usr/bin/env python3
"""
Helper script to properly format and add new CDP credentials to .env file.
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("CDP CREDENTIAL SETUP HELPER")
    print("=" * 70)
    
    print("\nThis will help you add your new CDP API credentials to the .env file.")
    print("\n📋 You'll need:")
    print("1. Your API key name (format: organizations/.../apiKeys/...)")
    print("2. Your private key (the full text including BEGIN/END headers)")
    
    print("\n" + "=" * 70)
    print("STEP 1: Enter your API Key Name")
    print("=" * 70)
    print("\nExample format:")
    print("organizations/5184a9ea-2cec-4a66-b00e-7cf6daaf048e/apiKeys/new-key-id-here")
    print("\nPaste your API key name:")
    
    api_key = input("> ").strip()
    
    if not api_key.startswith("organizations/") or "/apiKeys/" not in api_key:
        print("\n⚠️ Warning: API key doesn't match expected format")
        print("Expected: organizations/{org_id}/apiKeys/{key_id}")
    
    print("\n" + "=" * 70)
    print("STEP 2: Enter your Private Key")
    print("=" * 70)
    print("\nPaste your private key (including BEGIN/END lines).")
    print("When done, press Enter twice:")
    
    private_key_lines = []
    print("\n> ", end="")
    while True:
        line = input()
        if line == "" and private_key_lines and private_key_lines[-1] == "":
            break
        private_key_lines.append(line)
    
    # Remove trailing empty line
    if private_key_lines and private_key_lines[-1] == "":
        private_key_lines.pop()
    
    private_key = "\n".join(private_key_lines).strip()
    
    # Validate private key
    if not private_key.startswith("-----BEGIN"):
        print("\n⚠️ Warning: Private key should start with -----BEGIN EC PRIVATE KEY-----")
    if not "-----END" in private_key:
        print("⚠️ Warning: Private key should end with -----END EC PRIVATE KEY-----")
    
    print("\n" + "=" * 70)
    print("STEP 3: Update .env File")
    print("=" * 70)
    
    env_path = Path(".env")
    
    print(f"\n📝 Add these lines to your .env file:\n")
    print("# Production CDP Credentials (with all permissions)")
    print(f'COINBASE_PROD_CDP_API_KEY={api_key}')
    print(f'COINBASE_PROD_CDP_PRIVATE_KEY="""\\')
    print(private_key)
    print('"""')
    
    print("\n" + "=" * 70)
    print("OPTION: Automatically append to .env?")
    print("=" * 70)
    
    if env_path.exists():
        print(f"\n.env file exists at: {env_path.absolute()}")
        print("Do you want to APPEND these credentials to .env? (y/n)")
        choice = input("> ").strip().lower()
        
        if choice == 'y':
            # Read existing content
            with open(env_path, 'r') as f:
                existing = f.read()
            
            # Check if keys already exist
            if "COINBASE_PROD_CDP_API_KEY=" in existing:
                print("\n⚠️ COINBASE_PROD_CDP_API_KEY already exists in .env")
                print("Do you want to UPDATE it? (y/n)")
                update = input("> ").strip().lower()
                
                if update == 'y':
                    # Comment out old values
                    lines = existing.split('\n')
                    new_lines = []
                    in_private_key = False
                    skip_until_quotes = False
                    
                    for line in lines:
                        if line.startswith("COINBASE_PROD_CDP_API_KEY="):
                            new_lines.append(f"# {line}  # Replaced by new key")
                        elif line.startswith("COINBASE_PROD_CDP_PRIVATE_KEY="):
                            new_lines.append(f"# {line}  # Start of replaced key")
                            in_private_key = True
                            if '"""' in line:
                                skip_until_quotes = True
                        elif in_private_key:
                            new_lines.append(f"# {line}")
                            if skip_until_quotes and '"""' in line:
                                in_private_key = False
                                skip_until_quotes = False
                            elif line.endswith('-----'):
                                in_private_key = False
                        else:
                            new_lines.append(line)
                    
                    existing = '\n'.join(new_lines)
            
            # Append new credentials
            with open(env_path, 'w') as f:
                f.write(existing)
                if not existing.endswith('\n'):
                    f.write('\n')
                f.write('\n# Production CDP Credentials (added by setup script)\n')
                f.write(f'COINBASE_PROD_CDP_API_KEY={api_key}\n')
                f.write(f'COINBASE_PROD_CDP_PRIVATE_KEY="""\\\n{private_key}\n"""\n')
            
            print("\n✅ Credentials added to .env file!")
        else:
            print("\n📋 Please manually add the credentials shown above to your .env file")
    else:
        print(f"\n⚠️ No .env file found. Creating one from template...")
        
        template_path = Path(".env.template")
        if template_path.exists():
            with open(template_path, 'r') as f:
                template = f.read()
            
            # Replace the empty credential fields
            lines = template.split('\n')
            new_lines = []
            for line in lines:
                if line.startswith("COINBASE_PROD_CDP_API_KEY="):
                    new_lines.append(f'COINBASE_PROD_CDP_API_KEY={api_key}')
                elif line.startswith("COINBASE_PROD_CDP_PRIVATE_KEY="):
                    new_lines.append(f'COINBASE_PROD_CDP_PRIVATE_KEY="""\\\n{private_key}\n"""')
                else:
                    new_lines.append(line)
            
            with open(env_path, 'w') as f:
                f.write('\n'.join(new_lines))
            
            print(f"✅ Created .env file with your credentials!")
        else:
            print("❌ No .env.template found. Please create .env manually.")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Verify credentials:")
    print("   poetry run python scripts/verify_cdp_permissions.py")
    print("\n2. Test the bot:")
    print("   poetry run perps-bot --profile dev --dev-fast")
    print("\n3. For production:")
    print("   poetry run perps-bot --profile canary --dry-run")


if __name__ == "__main__":
    main()