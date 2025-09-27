#!/usr/bin/env python3
"""
Setup Legacy HMAC Authentication

This script helps set up legacy HMAC authentication which is more reliable
than CDP JWT for the Advanced Trade API.
"""

import os
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 60)
    print("SETUP LEGACY HMAC AUTHENTICATION")
    print("=" * 60)
    print()
    
    print("Based on the project documentation, CDP JWT authentication has")
    print("known issues with the Advanced Trade API. The recommended solution")
    print("is to use legacy HMAC authentication.")
    print()
    
    print("📋 LEGACY HMAC KEY CREATION GUIDE:")
    print("-" * 40)
    print()
    print("1. Go to: https://www.coinbase.com/settings/api")
    print("2. Click 'New API Key' (NOT 'Create CDP Key')")
    print("3. Choose 'Advanced Trade API' (not legacy Exchange)")
    print("4. Set permissions:")
    print("   ✅ View (required)")
    print("   ✅ Trade (for trade key)")
    print("   ❌ Transfer (disable for safety)")
    print("5. Use naming convention: perps-prod-bot01-trade-202508")
    print("6. Save credentials immediately (shown only once)")
    print()
    
    print("🔑 CREDENTIALS NEEDED:")
    print("-" * 20)
    print("• API Key (alphanumeric string)")
    print("• API Secret (base64 encoded string)")
    print("• Passphrase (often blank for Advanced Trade)")
    print()
    
    # Get credentials
    print("Enter your LEGACY HMAC credentials:")
    print()
    
    api_key = input("API Key: ").strip()
    if not api_key:
        print("❌ API key cannot be empty")
        return
    
    api_secret = input("API Secret (base64): ").strip()
    if not api_secret:
        print("❌ API secret cannot be empty")
        return
    
    passphrase = input("Passphrase (often blank, press Enter if none): ").strip()
    
    print()
    print("🔧 CREATING LEGACY HMAC CONFIGURATION")
    print("-" * 40)
    
    # Create legacy HMAC config
    config_content = f"""# =============================================================================
# COINBASE API CONFIGURATION - PRODUCTION ENVIRONMENT (LEGACY HMAC)
# =============================================================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Environment: prod
# Key: perps-prod-bot01-trade-202508 (Legacy HMAC)
# Authentication: HMAC (more reliable than CDP JWT)

# =============================================================================
# BROKER SELECTION
# =============================================================================
BROKER=coinbase

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================
COINBASE_SANDBOX=0

# =============================================================================
# API MODE & AUTHENTICATION
# =============================================================================
COINBASE_API_MODE=advanced
COINBASE_AUTH_TYPE=HMAC

# =============================================================================
# LEGACY API CREDENTIALS (HMAC Authentication)
# =============================================================================
# Key Name: perps-prod-bot01-trade-202508
COINBASE_API_KEY={api_key}
COINBASE_API_SECRET={api_secret}
COINBASE_API_PASSPHRASE={passphrase}

# =============================================================================
# TRADING FEATURES
# =============================================================================
COINBASE_ENABLE_DERIVATIVES=1

# =============================================================================
# SAFETY SETTINGS
# =============================================================================
RISK_REDUCE_ONLY_MODE=0
RISK_MAX_DAILY_LOSS_PCT=0.01
RISK_MAX_MARK_STALENESS_SECONDS=30

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================
COINBASE_ENABLE_KEEP_ALIVE=1
COINBASE_CONNECTION_TIMEOUT=30
COINBASE_REQUEST_TIMEOUT=10
COINBASE_RATE_LIMIT_RPS=10

# =============================================================================
# API ENDPOINTS
# =============================================================================
COINBASE_API_BASE=https://api.coinbase.com
COINBASE_WS_URL=wss://advanced-trade-ws.coinbase.com
COINBASE_API_VERSION=2024-10-24
"""
    
    # Save legacy HMAC config
    with open('.env.prod.hmac', 'w') as f:
        f.write(config_content)
    
    print("✅ Created .env.prod.hmac")
    
    # Update main .env file
    main_env_content = f"""# =============================================================================
# COINBASE API CONFIGURATION - MAIN (LEGACY HMAC)
# =============================================================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# This file uses legacy HMAC authentication (more reliable)

# =============================================================================
# BROKER SELECTION
# =============================================================================
BROKER=coinbase

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================
COINBASE_SANDBOX=0

# =============================================================================
# API MODE & AUTHENTICATION
# =============================================================================
COINBASE_API_MODE=advanced
COINBASE_AUTH_TYPE=HMAC

# =============================================================================
# LEGACY API CREDENTIALS (HMAC Authentication)
# =============================================================================
# Key Name: perps-prod-bot01-trade-202508
COINBASE_API_KEY={api_key}
COINBASE_API_SECRET={api_secret}
COINBASE_API_PASSPHRASE={passphrase}

# =============================================================================
# TRADING FEATURES
# =============================================================================
COINBASE_ENABLE_DERIVATIVES=1

# =============================================================================
# SAFETY SETTINGS
# =============================================================================
RISK_REDUCE_ONLY_MODE=0
RISK_MAX_DAILY_LOSS_PCT=0.01
RISK_MAX_MARK_STALENESS_SECONDS=30

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================
COINBASE_ENABLE_KEEP_ALIVE=1
COINBASE_CONNECTION_TIMEOUT=30
COINBASE_REQUEST_TIMEOUT=10
COINBASE_RATE_LIMIT_RPS=10

# =============================================================================
# API ENDPOINTS
# =============================================================================
COINBASE_API_BASE=https://api.coinbase.com
COINBASE_WS_URL=wss://advanced-trade-ws.coinbase.com
COINBASE_API_VERSION=2024-10-24
"""
    
    with open('.env', 'w') as f:
        f.write(main_env_content)
    
    print("✅ Updated .env with legacy HMAC credentials")
    
    print()
    print("📋 NEXT STEPS:")
    print("1. Test the legacy HMAC configuration:")
    print("   python test_legacy_hmac.py")
    print()
    print("2. Create a monitor key (read-only) for production")
    print("3. Create demo/sandbox keys for testing")
    print()
    print("⚠️  SECURITY REMINDERS:")
    print("• Keep your API credentials secure")
    print("• Never commit .env files to git")
    print("• Use IP allowlisting in production")
    print()
    print("✅ Legacy HMAC configuration created successfully!")

if __name__ == "__main__":
    main()
