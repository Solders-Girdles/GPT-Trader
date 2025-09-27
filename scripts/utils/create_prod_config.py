#!/usr/bin/env python3
"""
Create Production Configuration

This script helps set up the production configuration with the newly created API credentials.
"""

import json
import base64
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 60)
    print("SETTING UP PRODUCTION CONFIGURATION")
    print("=" * 60)
    print()
    
    # Read the credentials file
    creds_file = Path("/Users/rj/Downloads/perps-prod-bot01-trade-202508.json")
    
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
    
    print("✅ Credentials loaded successfully")
    print(f"Key ID: {creds['id']}")
    print(f"Private Key: {creds['privateKey'][:20]}...")
    print()
    
    # Get additional information
    print("📝 ADDITIONAL INFORMATION NEEDED:")
    print()
    
    # For CDP keys, we need the organization and key path
    org_id = input("Organization ID (from CDP dashboard): ").strip()
    if not org_id:
        print("❌ Organization ID is required")
        return
    
    # Create the CDP API key path
    cdp_api_key = f"organizations/{org_id}/apiKeys/{creds['id']}"
    
    print()
    print("🔧 CREATING PRODUCTION CONFIGURATION")
    print("-" * 40)
    
    # Create production config
    config_content = f"""# =============================================================================
# COINBASE API CONFIGURATION - PRODUCTION ENVIRONMENT
# =============================================================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Environment: prod
# Key: perps-prod-bot01-trade-202508
# Key ID: {creds['id']}

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
# Key Name: perps-prod-bot01-trade-202508
COINBASE_CDP_API_KEY={cdp_api_key}
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
    
    # Save production config
    with open('.env.prod', 'w') as f:
        f.write(config_content)
    
    print("✅ Created .env.prod")
    
    # Create main .env file
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
    
    print("✅ Created .env")
    
    print()
    print("📋 NEXT STEPS:")
    print("1. Create a monitor key (read-only) for production")
    print("2. Create demo/sandbox keys for testing")
    print("3. Test the configuration:")
    print("   python scripts/test_coinbase_connection.py")
    print()
    print("⚠️  SECURITY REMINDERS:")
    print("• Keep the credentials file secure")
    print("• Never commit .env files to git")
    print("• Consider deleting the downloaded JSON file after setup")
    print()
    print("✅ Production configuration created successfully!")

if __name__ == "__main__":
    main()
