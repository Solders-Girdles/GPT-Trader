#!/usr/bin/env python3
"""
Test Current API Setup

This script tests your current Coinbase API configuration and provides
recommendations for fixing any issues.
"""

import os
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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
    print("CURRENT API SETUP ANALYSIS")
    print("=" * 60)
    print()
    
    # Load environment variables
    env_vars = load_env_file('.env')
    
    print("1. CURRENT CONFIGURATION:")
    print("-" * 30)
    
    # Check broker selection
    broker = env_vars.get('BROKER', 'Not set')
    print(f"Broker: {broker}")
    
    # Check environment
    sandbox = env_vars.get('COINBASE_SANDBOX', 'Not set')
    print(f"Sandbox Mode: {sandbox}")
    
    # Check API mode
    api_mode = env_vars.get('COINBASE_API_MODE', 'Not set')
    print(f"API Mode: {api_mode}")
    
    # Check auth type
    auth_type = env_vars.get('COINBASE_AUTH_TYPE', 'Not set')
    print(f"Auth Type: {auth_type}")
    
    print()
    print("2. CREDENTIALS ANALYSIS:")
    print("-" * 30)
    
    # Check CDP credentials
    cdp_key = env_vars.get('COINBASE_CDP_API_KEY', '')
    cdp_private = env_vars.get('COINBASE_CDP_PRIVATE_KEY', '')
    
    if cdp_key and cdp_private:
        print("✅ CDP Credentials: Configured")
        print(f"   CDP Key: {cdp_key[:30]}...")
        print(f"   Private Key: {'Present' if 'BEGIN EC PRIVATE KEY' in cdp_private else 'Invalid format'}")
    else:
        print("❌ CDP Credentials: Not configured")
    
    # Check legacy credentials
    api_key = env_vars.get('COINBASE_API_KEY', '')
    api_secret = env_vars.get('COINBASE_API_SECRET', '')
    
    if api_key and api_secret:
        print("✅ Legacy API Credentials: Configured")
        print(f"   API Key: {api_key[:20]}...")
        print(f"   API Secret: {'Present' if len(api_secret) > 10 else 'Too short'}")
    else:
        print("❌ Legacy API Credentials: Not configured")
    
    print()
    print("3. ISSUES IDENTIFIED:")
    print("-" * 30)
    
    issues = []
    
    if cdp_key and cdp_private and not (api_key and api_secret):
        issues.append("⚠️  Using CDP credentials only - these have known issues with Advanced Trade API")
        issues.append("   Recommendation: Add legacy API credentials for reliable operation")
    
    if not api_key and not cdp_key:
        issues.append("❌ No API credentials configured")
        issues.append("   You need to create API keys at Coinbase")
    
    if sandbox == '0' and not issues:
        issues.append("⚠️  Production mode detected - ensure you have proper credentials")
    
    if not issues:
        print("✅ No major issues detected")
    else:
        for issue in issues:
            print(issue)
    
    print()
    print("4. RECOMMENDATIONS:")
    print("-" * 30)
    
    if cdp_key and not api_key:
        print("🔧 IMMEDIATE ACTION NEEDED:")
        print("   1. Create legacy API keys at https://www.coinbase.com/settings/api")
        print("   2. Choose 'Advanced Trade API' (not CDP)")
        print("   3. Enable: View, Trade")
        print("   4. Disable: Transfer (for safety)")
        print("   5. Add COINBASE_API_KEY and COINBASE_API_SECRET to .env")
        print()
        print("   This will provide reliable authentication for the Advanced Trade API.")
    
    print("📋 NEXT STEPS:")
    print("   1. Test current setup: python scripts/test_coinbase_connection.py")
    print("   2. If tests fail, follow the recommendations above")
    print("   3. For sandbox testing, set COINBASE_SANDBOX=1")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
