#!/usr/bin/env python3
"""
Manually load and test environment variables.
"""

import os
import sys
from pathlib import Path

def load_env_file(file_path):
    """Manually parse .env file."""
    with open(file_path) as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # Parse key=value
        if '=' in line:
            key, value = line.split('=', 1)
            
            # Handle multi-line values (like private keys)
            if value.startswith('"') and not value.endswith('"'):
                # Multi-line value
                value_lines = [value[1:]]  # Remove opening quote
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    if next_line.endswith('"'):
                        value_lines.append(next_line[:-1])  # Remove closing quote
                        value = '\n'.join(value_lines)
                        break
                    value_lines.append(next_line)
                    i += 1
            else:
                # Single line value - remove quotes if present
                value = value.strip('"')
            
            os.environ[key] = value
        
        i += 1

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
print(f"Loading environment from: {env_path}")
load_env_file(env_path)

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
        account_id = broker.get_account_id()
        print(f"Account ID: {account_id}")
    else:
        print("❌ CDP authentication failed (401 - check API key permissions)")
else:
    print("\n❌ CDP credentials not detected")
    print(f"API Key: {broker.config.api_key[:20] if broker.config.api_key else 'None'}...")
    print(f"Has Passphrase: {broker.config.passphrase is not None}")