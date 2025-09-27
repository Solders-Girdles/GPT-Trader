#!/usr/bin/env python3
"""
Debug broker creation to see what credentials are being used.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check environment
print("Environment Variables:")
print("=" * 60)
print(f"BROKER: {os.getenv('BROKER')}")
print(f"COINBASE_SANDBOX: {os.getenv('COINBASE_SANDBOX')}")
print(f"COINBASE_API_BASE: {os.getenv('COINBASE_API_BASE')}")
print(f"COINBASE_CDP_API_KEY: ...{os.getenv('COINBASE_CDP_API_KEY', '')[-30:]}")
print(f"COINBASE_CDP_PRIVATE_KEY: {'Present' if os.getenv('COINBASE_CDP_PRIVATE_KEY') else 'Missing'}")
print(f"COINBASE_AUTH_TYPE: {os.getenv('COINBASE_AUTH_TYPE')}")
print()

# Create broker and inspect config
from src.bot_v2.orchestration.broker_factory import create_brokerage

broker = create_brokerage()
print("Broker Configuration:")
print("=" * 60)
print(f"Type: {type(broker).__name__}")
print(f"Base URL: {broker.config.base_url}")
print(f"Sandbox: {broker.config.sandbox}")
print(f"Auth Type: {broker.config.auth_type}")
print(f"CDP API Key: ...{broker.config.cdp_api_key[-30:] if broker.config.cdp_api_key else 'None'}")
print(f"CDP Private Key: {'Present' if broker.config.cdp_private_key else 'Missing'}")
print()

# Check auth object
print("Auth Object:")
print("=" * 60)
print(f"Auth Type: {type(broker._auth).__name__}")

if hasattr(broker._auth, 'api_key_name'):
    print(f"CDP API Key Name: ...{broker._auth.api_key_name[-30:]}")
    print(f"CDP Private Key: {'Present' if broker._auth.private_key_pem else 'Missing'}")
    
    # Test JWT generation
    try:
        jwt = broker._auth.generate_jwt("GET", "/api/v3/brokerage/accounts")
        print(f"JWT Generation: ✅ Success ({len(jwt)} chars)")
    except Exception as e:
        print(f"JWT Generation: ❌ Failed - {e}")
else:
    print("Not using CDP auth")