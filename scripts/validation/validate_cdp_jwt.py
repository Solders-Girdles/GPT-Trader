#!/usr/bin/env python3
"""
Validate CDP JWT generation for Coinbase Advanced Trade.

Prints a short-lived JWT and its expiry to confirm credentials are valid.

Env vars:
  COINBASE_PROD_CDP_API_KEY (or COINBASE_CDP_API_KEY)
  COINBASE_PROD_CDP_PRIVATE_KEY (or COINBASE_CDP_PRIVATE_KEY)
"""

from __future__ import annotations

import os
import sys
from datetime import datetime


def main() -> int:
    api_key = os.getenv('COINBASE_PROD_CDP_API_KEY') or os.getenv('COINBASE_CDP_API_KEY')
    private_key = os.getenv('COINBASE_PROD_CDP_PRIVATE_KEY') or os.getenv('COINBASE_CDP_PRIVATE_KEY')

    if not (api_key and private_key):
        print("ERROR: Missing CDP credentials. Set COINBASE_PROD_CDP_API_KEY and COINBASE_PROD_CDP_PRIVATE_KEY.")
        return 2

    try:
        from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import create_cdp_auth_v2
        auth = create_cdp_auth_v2(api_key_name=api_key, private_key_pem=private_key)
        token = auth.generate_jwt(method="GET", path="/api/v3/brokerage/accounts")
        claims = auth.validate_jwt(token)
        exp = claims.get('exp')
        exp_dt = datetime.utcfromtimestamp(exp) if isinstance(exp, (int, float)) else exp
        print("JWT generation OK")
        print(f"  key: {api_key}")
        print(f"  kid: {claims.get('kid')}")
        print(f"  exp: {exp_dt}")
        print(f"  sample: {token[:20]}...{token[-20:]}")
        return 0
    except Exception as e:
        print(f"ERROR: JWT generation failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

