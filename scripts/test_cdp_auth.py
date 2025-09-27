#!/usr/bin/env python3
"""
Test script to verify CDP authentication format.
Shows the correct way to format the private key in .env file.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_private_key_format():
    """Test and show correct private key format."""
    
    # Example of what your .env should look like
    example_env = '''
# CORRECT FORMAT - Private key with headers on multiple lines in triple quotes:
COINBASE_PROD_CDP_PRIVATE_KEY="""-----BEGIN EC PRIVATE KEY-----
MHcCAQEEINeZ9bzCUz7AxoU1x8mP9TtO0MLsJzfTdsxpHNg1vMfsoAoGCCqGSM49
AwEHoUQDQgAEzmIXCK86/nTz55gjlr8fJh2U2XCTGhlfsrIgNEogihp9/qFLACsr
ZpRqowTeyjNcwIeEY3KztpxH8V1eCnBCHg==
-----END EC PRIVATE KEY-----"""

# ALSO CORRECT - Single line with escaped newlines:
# COINBASE_PROD_CDP_PRIVATE_KEY="-----BEGIN EC PRIVATE KEY-----\\nMHcCAQEEI...\\n-----END EC PRIVATE KEY-----"
'''
    
    print("=" * 70)
    print("CDP AUTHENTICATION FORMAT GUIDE")
    print("=" * 70)
    print(example_env)
    
    # Check current environment
    print("\n" + "=" * 70)
    print("CHECKING YOUR CURRENT CONFIGURATION")
    print("=" * 70)
    
    # Check if keys are set
    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not api_key:
        print("❌ CDP API Key not found")
        print("   Set COINBASE_PROD_CDP_API_KEY in .env")
    else:
        print(f"✅ CDP API Key found: {api_key[:20]}...")
    
    if not private_key:
        print("❌ CDP Private Key not found")
        print("   Set COINBASE_PROD_CDP_PRIVATE_KEY in .env")
    else:
        # Check format
        if private_key.startswith("-----BEGIN"):
            print("✅ Private key has correct header")
            if "-----END" in private_key:
                print("✅ Private key has correct footer")
            else:
                print("❌ Private key missing END footer")
        elif "MHcCAQEEI" in private_key and not "-----BEGIN" in private_key:
            print("❌ Private key is missing headers/footers!")
            print("   The key content is there but needs proper PEM formatting")
            print("\n   FIX: Wrap your key with headers:")
            print('   COINBASE_PROD_CDP_PRIVATE_KEY="""-----BEGIN EC PRIVATE KEY-----')
            print(f'   {private_key[:60]}...')
            print('   -----END EC PRIVATE KEY-----"""')
        else:
            print("❌ Private key format unrecognized")
    
    # Try to load the key
    if private_key:
        print("\n" + "=" * 70)
        print("TESTING KEY LOADING")
        print("=" * 70)
        
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.backends import default_backend
            
            # Normalize the key
            if not private_key.startswith("-----BEGIN"):
                # Try to fix it
                if "MHcCAQEEI" in private_key:
                    # Just the base64 content, add headers
                    private_key = f"-----BEGIN EC PRIVATE KEY-----\n{private_key}\n-----END EC PRIVATE KEY-----"
            
            # Try to load
            key_bytes = private_key.encode() if isinstance(private_key, str) else private_key
            key_obj = serialization.load_pem_private_key(
                key_bytes,
                password=None,
                backend=default_backend()
            )
            print("✅ Private key loaded successfully!")
            print("   Your authentication should work now")
            
        except Exception as e:
            print(f"❌ Failed to load private key: {e}")
            print("\n   This is the exact error you're seeing in the bot")
            print("   Fix the format as shown above")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Edit your .env file")
    print("2. Format the private key with headers/footers as shown above")
    print("3. Run: poetry run perps-bot --profile dev --dev-fast")
    print("4. If that works, try: poetry run perps-bot --profile canary --dry-run")


if __name__ == "__main__":
    test_private_key_format()