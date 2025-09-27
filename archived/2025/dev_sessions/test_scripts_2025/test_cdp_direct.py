#!/usr/bin/env python3
"""
Direct test of CDP authentication without the broker layer.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bot_v2.features.brokerages.coinbase.cdp_auth import create_cdp_auth
from src.bot_v2.features.brokerages.coinbase.client import CoinbaseClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_production_env():
    """Load production environment variables."""
    env_file = Path(__file__).parent.parent / ".env.production"
    if env_file.exists():
        logger.info(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Handle multi-line values (like private keys)
                        if value.startswith('"') and not value.endswith('"'):
                            # Multi-line value
                            lines = [value[1:]]  # Remove opening quote
                            for next_line in f:
                                next_line = next_line.strip()
                                if next_line.endswith('"'):
                                    lines.append(next_line[:-1])  # Remove closing quote
                                    value = '\n'.join(lines)
                                    break
                                lines.append(next_line)
                        else:
                            # Single line value - remove quotes if present
                            value = value.strip('"')
                        os.environ[key] = value


def test_cdp_auth():
    """Test CDP authentication directly."""
    logger.info("Testing CDP Authentication Directly")
    logger.info("=" * 60)
    
    # Load environment
    load_production_env()
    
    # Get CDP credentials
    cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
    cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not cdp_api_key or not cdp_private_key:
        logger.error("CDP credentials not found")
        return
    
    logger.info(f"CDP API Key: ...{cdp_api_key[-30:]}")
    logger.info(f"Private Key Length: {len(cdp_private_key)} chars")
    
    # Create CDP auth
    try:
        auth = create_cdp_auth(cdp_api_key, cdp_private_key)
        logger.info("✅ CDP auth created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create CDP auth: {e}")
        return
    
    # Test JWT generation
    try:
        method = "GET"
        path = "/api/v3/brokerage/accounts"
        jwt_token = auth.generate_jwt(method, path)
        logger.info(f"✅ JWT generated successfully")
        logger.info(f"   Token length: {len(jwt_token)} chars")
        logger.info(f"   Token preview: {jwt_token[:50]}...")
    except Exception as e:
        logger.error(f"❌ Failed to generate JWT: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create client and test request
    try:
        client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=auth
        )
        logger.info("✅ Client created with CDP auth")
        
        # Make test request
        logger.info("\nMaking test request to /api/v3/brokerage/accounts...")
        data = client.get_accounts()
        
        if data:
            logger.info("✅ Successfully authenticated and got accounts!")
            accounts = data.get("accounts") or data.get("data") or []
            logger.info(f"   Found {len(accounts)} accounts")
            for acc in accounts[:3]:
                logger.info(f"   - {acc.get('currency', 'Unknown')}: {acc.get('balance', 0)}")
        else:
            logger.error("❌ No data returned from accounts endpoint")
            
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_cdp_auth()