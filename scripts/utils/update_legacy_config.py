#!/usr/bin/env python3
"""
Update Legacy API Configuration

This script updates the production configuration with the new legacy API credentials.
"""

import json
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 60)
    print("UPDATING LEGACY API CONFIGURATION")
    print("=" * 60)
    print()
    
    # Read the new credentials file
    creds_file = Path("/Users/rj/Downloads/perps-prod-bot01-trade-202508-legacy-api.json")
    
    if not creds_file.exists():
        print("❌ Credentials file not found!")
        print(f"Expected: {creds_file}")
        return
    
    try:
        with open(creds_file, 'r') as f:
            creds = json.load(f)
    except Exception as e:
        print(f"❌ Error reading credentials file: {e}")
        return
    
    print("✅ New credentials loaded successfully")
    print(f"Key Name: {creds['name']}")
    print(f"Private Key: {'Present' if 'BEGIN EC PRIVATE KEY' in creds['privateKey'] else 'Invalid format'}")
    print()
    
    # Update production config
    config_content = f"""# =============================================================================
# COINBASE API CONFIGURATION - PRODUCTION ENVIRONMENT
# =============================================================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Environment: prod
# Key: perps-prod-bot01-trade-202508-legacy-api
# Key ID: e497c5d9-7694-43c6-973f-f6dd2e15e60a

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
COINBASE_AUTH_TYPE=JWT

# =============================================================================
# CDP API CREDENTIALS (JWT Authentication)
# =============================================================================
# Key Name: perps-prod-bot01-trade-202508-legacy-api
COINBASE_CDP_API_KEY={creds['name']}
COINBASE_CDP_PRIVATE_KEY={creds['privateKey']}

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
    
    # Save updated production config
    with open('.env.prod', 'w') as f:
        f.write(config_content)
    
    print("✅ Updated .env.prod with new credentials")
    
    # Update main .env file
    main_env_content = f"""# =============================================================================
# COINBASE API CONFIGURATION - MAIN
# =============================================================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# This file points to production environment

# =============================================================================
# BROKER SELECTION
# =============================================================================
BROKER=coinbase

# =============================================================================
# ENVIRONMENT SELECTION
# =============================================================================
# Set to 'demo' for sandbox testing, 'prod' for production
ENVIRONMENT=prod

# =============================================================================
# AUTO-LOAD ENVIRONMENT CONFIG
# =============================================================================
# This will be handled by the application to load the correct config
"""
    
    with open('.env', 'w') as f:
        f.write(main_env_content)
    
    print("✅ Updated .env")
    
    print()
    print("📋 NEXT STEPS:")
    print("1. Test the new configuration:")
    print("   pytest -q -m integration tests/integration/test_cdp_comprehensive.py")
    print()
    print("2. Create a monitor key (read-only) for production")
    print("3. Create demo/sandbox keys for testing")
    print()
    print("⚠️  SECURITY REMINDERS:")
    print("• Keep the credentials file secure")
    print("• Never commit .env files to git")
    print("• Consider deleting the downloaded JSON file after setup")
    print()
    print("✅ Configuration updated with new legacy API credentials!")

if __name__ == "__main__":
    main()
