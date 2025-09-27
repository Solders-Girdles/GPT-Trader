#!/usr/bin/env python3
"""
Diagnose CDP private key format issues.
"""

import os
import sys
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def diagnose_private_key():
    """Diagnose private key format and provide guidance."""
    print("="*60)
    print("🔍 CDP Private Key Diagnostic")
    print("="*60)
    
    # Check environment variables
    cdp_key = os.getenv("COINBASE_CDP_API_KEY") or os.getenv("COINBASE_PROD_CDP_API_KEY")
    private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY") or os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY")
    
    if not cdp_key:
        print("❌ No CDP API key found in environment")
        print("   Set COINBASE_CDP_API_KEY or COINBASE_PROD_CDP_API_KEY")
        return 1
    
    print(f"✅ CDP API Key found: {cdp_key[:20]}...")
    
    if not private_key:
        print("❌ No CDP private key found in environment")
        print("   Set COINBASE_CDP_PRIVATE_KEY or COINBASE_PROD_CDP_PRIVATE_KEY")
        return 1
    
    print(f"✅ Private key found ({len(private_key)} characters)")
    
    # Analyze key format
    print("\n📋 Key Format Analysis:")
    
    # Check for common formats
    if private_key.startswith("-----BEGIN"):
        print("  ✅ PEM format detected")
        key_format = "PEM"
    elif private_key.startswith("MII"):
        print("  ⚠️  Base64 DER format detected - needs conversion to PEM")
        key_format = "DER_BASE64"
    elif len(private_key) < 100:
        print("  ❌ Key too short - might be a key name/ID instead of actual key")
        key_format = "INVALID"
    else:
        print("  ⚠️  Unknown format - attempting to detect...")
        key_format = "UNKNOWN"
    
    # Try to parse the key
    print("\n🔐 Key Parsing Attempt:")
    
    if key_format == "PEM":
        try:
            # Try to load as PEM
            key_bytes = private_key.encode('utf-8')
            parsed_key = serialization.load_pem_private_key(
                key_bytes,
                password=None,
                backend=default_backend()
            )
            print("  ✅ Successfully parsed as PEM")
            print(f"  Key type: {type(parsed_key).__name__}")
            return 0
        except Exception as e:
            print(f"  ❌ Failed to parse PEM: {e}")
            
            # Check for common issues
            if "EC PRIVATE KEY" in private_key:
                print("\n  💡 Key appears to be EC format. Try converting:")
                print("     openssl ec -in key.pem -out key_pkcs8.pem")
            elif "RSA PRIVATE KEY" in private_key:
                print("\n  💡 Key appears to be RSA format. CDP requires EC keys.")
                print("     You need to generate a new EC key from Coinbase CDP console.")
    
    elif key_format == "DER_BASE64":
        print("  Attempting to convert DER to PEM...")
        try:
            # Decode base64
            der_bytes = base64.b64decode(private_key)
            
            # Try to load as DER
            parsed_key = serialization.load_der_private_key(
                der_bytes,
                password=None,
                backend=default_backend()
            )
            
            # Convert to PEM
            pem_key = parsed_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            print("  ✅ Successfully converted to PEM")
            print("\n  📝 Converted PEM key (set this as COINBASE_CDP_PRIVATE_KEY):")
            print("-" * 40)
            print(pem_key.decode('utf-8'))
            print("-" * 40)
            return 0
            
        except Exception as e:
            print(f"  ❌ Failed to convert: {e}")
    
    # Provide guidance
    print("\n💡 Troubleshooting Guide:")
    print("1. CDP keys must be in EC (Elliptic Curve) format, not RSA")
    print("2. The key should be in PKCS#8 PEM format")
    print("3. It should start with '-----BEGIN PRIVATE KEY-----'")
    print("4. Not '-----BEGIN EC PRIVATE KEY-----' (needs conversion)")
    print("\n📚 To get a proper CDP key:")
    print("1. Go to https://portal.cdp.coinbase.com/")
    print("2. Create a new API key")
    print("3. Download the private key")
    print("4. Set it as COINBASE_CDP_PRIVATE_KEY environment variable")
    print("\n🔧 If you have an EC key in wrong format, convert it:")
    print("   openssl pkcs8 -topk8 -nocrypt -in ec_key.pem -out pkcs8_key.pem")
    
    return 1

if __name__ == "__main__":
    sys.exit(diagnose_private_key())