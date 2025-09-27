#!/usr/bin/env python3
"""
Test environment loading from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
print(f"Loading environment from: {env_path}")
load_dotenv(env_path, override=True)

print("\nEnvironment Variables After Loading:")
print("=" * 60)
print(f"BROKER: {os.getenv('BROKER')}")
print(f"COINBASE_SANDBOX: {os.getenv('COINBASE_SANDBOX')}")
print(f"COINBASE_API_BASE: {os.getenv('COINBASE_API_BASE')}")
print(f"COINBASE_CDP_API_KEY: ...{os.getenv('COINBASE_CDP_API_KEY', '')[-30:]}")
print(f"COINBASE_CDP_PRIVATE_KEY: {'Present' if os.getenv('COINBASE_CDP_PRIVATE_KEY') else 'Missing'}")
print(f"COINBASE_AUTH_TYPE: {os.getenv('COINBASE_AUTH_TYPE')}")

# Now test broker creation
print("\nBroker Creation Test:")
print("=" * 60)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bot_v2.orchestration.broker_factory import create_brokerage

broker = create_brokerage()
print(f"Broker Type: {type(broker).__name__}")
print(f"Base URL: {broker.config.base_url}")
print(f"Auth Type: {broker.config.auth_type}")
print(f"Using CDP: {broker.config.cdp_api_key is not None}")

if broker.config.cdp_api_key:
    print("\n✅ CDP credentials detected!")
    print("Testing CDP authentication...")
    
    # Test connection
    if broker.connect():
        print("✅ Successfully connected with CDP!")
    else:
        print("❌ CDP authentication failed")
else:
    print("\n❌ CDP credentials not detected")