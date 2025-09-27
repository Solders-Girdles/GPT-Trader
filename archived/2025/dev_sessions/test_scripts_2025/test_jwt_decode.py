#!/usr/bin/env python3
"""
Decode and verify JWT structure.
"""

import os
import sys
import json
import base64
from pathlib import Path

# Load environment
def load_production_env():
    env_file = Path(__file__).parent.parent / ".env.production"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Handle multi-line values (like private keys)
                        if value.startswith('"') and not value.endswith('"'):
                            lines = [value[1:]]  # Remove opening quote
                            for next_line in f:
                                next_line = next_line.strip()
                                if next_line.endswith('"'):
                                    lines.append(next_line[:-1])  # Remove closing quote
                                    value = '\n'.join(lines)
                                    break
                                lines.append(next_line)
                        else:
                            value = value.strip('"')
                        os.environ[key] = value

load_production_env()

# Import after env is loaded
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bot_v2.features.brokerages.coinbase.cdp_auth import create_cdp_auth
import jwt

# Get credentials
cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")

print("JWT Structure Verification")
print("=" * 60)

# Create auth and generate JWT
auth = create_cdp_auth(cdp_api_key, cdp_private_key)
method = "GET"
path = "/api/v3/brokerage/accounts"

# Generate JWT
token = auth.generate_jwt(method, path)
print(f"Token generated: {len(token)} chars")
print(f"Token: {token[:50]}...")

# Decode without verification to inspect structure
try:
    # Decode header and payload
    parts = token.split('.')
    
    # Decode header
    header_b64 = parts[0]
    header_b64 += '=' * (4 - len(header_b64) % 4)  # Add padding
    header = json.loads(base64.urlsafe_b64decode(header_b64))
    
    # Decode payload
    payload_b64 = parts[1]
    payload_b64 += '=' * (4 - len(payload_b64) % 4)  # Add padding
    payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    
    print("\nJWT Header:")
    print(json.dumps(header, indent=2))
    
    print("\nJWT Payload:")
    print(json.dumps(payload, indent=2))
    
    # Verify structure matches Coinbase requirements
    print("\n✓ Structure Check:")
    print(f"  Header alg: {'✅' if header.get('alg') == 'ES256' else '❌'} {header.get('alg')}")
    print(f"  Header kid: {'✅' if header.get('kid') == cdp_api_key else '❌'} Present")
    print(f"  Header typ: {'✅' if header.get('typ') == 'JWT' else '❌'} {header.get('typ')}")
    print(f"  Header nonce: {'✅' if 'nonce' in header else '❌'} {'Present' if 'nonce' in header else 'Missing'}")
    
    print(f"\n  Payload sub: {'✅' if payload.get('sub') == cdp_api_key else '❌'} Matches API key")
    print(f"  Payload iss: {'✅' if payload.get('iss') == 'coinbase-cloud' else '❌'} {payload.get('iss')}")
    print(f"  Payload aud: {'✅' if 'retail_rest_api_proxy' in payload.get('aud', []) else '❌'} {payload.get('aud')}")
    print(f"  Payload uri: {'✅' if payload.get('uri') == f'{method} {path}' else '❌'} {payload.get('uri')}")
    
    # Check timing
    import time
    current_time = int(time.time())
    nbf = payload.get('nbf', 0)
    exp = payload.get('exp', 0)
    print(f"\n  Time valid: {'✅' if nbf <= current_time <= exp else '❌'} Current: {current_time}, NBF: {nbf}, EXP: {exp}")
    
except Exception as e:
    print(f"Error decoding JWT: {e}")
    import traceback
    traceback.print_exc()