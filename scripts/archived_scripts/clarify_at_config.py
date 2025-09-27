#!/usr/bin/env python3
"""
Clarify Advanced Trade vs Exchange API configuration.
Explains endpoint differences and proper setup.
"""

import os
import sys
import json
from pathlib import Path

def clarify_configuration():
    """Clarify AT vs Exchange API configuration."""
    print("🔍 ADVANCED TRADE vs EXCHANGE API CLARIFICATION")
    print("="*70)
    
    print("\n📚 KEY DIFFERENCES:")
    print("-"*70)
    
    print("\n1️⃣  Advanced Trade (AT) - CDP/JWT Auth:")
    print("   • Endpoint: api.coinbase.com")
    print("   • WebSocket: advanced-trade-ws.coinbase.com")
    print("   • Auth: JWT with CDP API key")
    print("   • Sandbox: NO PUBLIC SANDBOX (use production with tiny sizes)")
    print("   • Products: Spot, Perpetuals, Futures")
    
    print("\n2️⃣  Exchange API - Legacy Auth:")
    print("   • Endpoint: api.exchange.coinbase.com")
    print("   • Sandbox: api-public.sandbox.exchange.coinbase.com")
    print("   • WebSocket: ws-direct.exchange.coinbase.com")
    print("   • Auth: HMAC with API key/secret/passphrase")
    print("   • Products: Spot only (no derivatives)")
    
    print("\n" + "="*70)
    print("⚙️  CURRENT CONFIGURATION:")
    print("-"*70)
    
    # Check current settings
    api_mode = os.getenv('COINBASE_API_MODE', 'not_set')
    is_sandbox = os.getenv('COINBASE_SANDBOX', '0') == '1'
    auth_type = os.getenv('COINBASE_AUTH_TYPE', 'not_set')
    has_cdp_key = bool(os.getenv('COINBASE_CDP_API_KEY'))
    has_derivatives = os.getenv('COINBASE_ENABLE_DERIVATIVES') == '1'
    
    print(f"\nAPI Mode: {api_mode}")
    print(f"Auth Type: {auth_type}")
    print(f"Sandbox Flag: {'Yes' if is_sandbox else 'No'}")
    print(f"CDP Key: {'Configured' if has_cdp_key else 'Not configured'}")
    print(f"Derivatives: {'Enabled' if has_derivatives else 'Disabled'}")
    
    # Determine what's configured
    if api_mode == 'advanced' and auth_type == 'JWT' and has_cdp_key:
        print("\n✅ Configured for: ADVANCED TRADE (CDP)")
        
        if is_sandbox:
            print("\n⚠️  IMPORTANT CLARIFICATION:")
            print("   Advanced Trade has NO public sandbox!")
            print("   Even with COINBASE_SANDBOX=1, we use production endpoints")
            print("   Safety: Using minimal position sizes (0.0001 BTC)")
            print("\n   This is CORRECT behavior for AT demo trading")
    
    elif api_mode == 'legacy' or not has_cdp_key:
        print("\n📍 Configured for: EXCHANGE API (Legacy)")
        
        if is_sandbox:
            print("   Using sandbox endpoints (safe for testing)")
        else:
            print("   Using production endpoints")
    
    else:
        print("\n❌ MISCONFIGURED - Mixed settings detected")
    
    print("\n" + "="*70)
    print("📋 CORRECT CONFIGURATIONS:")
    print("-"*70)
    
    print("\n🎯 For DEMO Trading with Advanced Trade (Recommended):")
    print("""
# Demo with Advanced Trade (production endpoints, tiny sizes)
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_CDP_API_KEY="organizations/xxx/apiKeys/yyy"
export COINBASE_CDP_PRIVATE_KEY="/path/to/cdp-private-key.pem"
export COINBASE_ENABLE_DERIVATIVES=1
export COINBASE_SANDBOX=1  # Flag for demo mode (tiny sizes)
export COINBASE_MAX_POSITION_SIZE=0.0001  # Minimal size for safety

# Note: AT always uses production endpoints (api.coinbase.com)
# The SANDBOX flag just enables safety limits, not different endpoints
""")
    
    print("\n🎯 For PRODUCTION Trading with Advanced Trade:")
    print("""
# Production with Advanced Trade
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_CDP_API_KEY="organizations/xxx/apiKeys/yyy"
export COINBASE_CDP_PRIVATE_KEY="/path/to/cdp-private-key.pem"
export COINBASE_ENABLE_DERIVATIVES=1
export COINBASE_SANDBOX=0  # Production mode
export COINBASE_MAX_POSITION_SIZE=0.01  # Normal position size
export COINBASE_DAILY_LOSS_LIMIT=100  # $100 daily loss limit
""")
    
    print("\n" + "="*70)
    print("🔧 ENVIRONMENT FILES:")
    print("-"*70)
    
    # Create corrected environment files
    demo_env = """#!/bin/bash
# Advanced Trade Demo Configuration
# Uses production endpoints with safety limits

# API Configuration
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1

# Demo mode flag (for safety limits, not endpoints)
export COINBASE_SANDBOX=1

# CDP Authentication (file-based)
export COINBASE_CDP_API_KEY="$(cat ~/.coinbase/cdp-api-key.txt 2>/dev/null)"
export COINBASE_CDP_PRIVATE_KEY="~/.coinbase/cdp-private-key.pem"

# Safety Parameters (conservative for demo)
export COINBASE_MAX_POSITION_SIZE=0.0001  # Minimal BTC
export COINBASE_DAILY_LOSS_LIMIT=10      # $10 max loss
export COINBASE_MAX_IMPACT_BPS=10        # 10 bps max impact
export COINBASE_KILL_SWITCH=enabled

echo "✅ Advanced Trade DEMO environment loaded"
echo "   Using production endpoints with minimal sizes"
echo "   Max position: 0.0001 BTC"
"""
    
    prod_env = """#!/bin/bash
# Advanced Trade Production Configuration

# API Configuration
export COINBASE_API_MODE=advanced
export COINBASE_AUTH_TYPE=JWT
export COINBASE_ENABLE_DERIVATIVES=1

# Production mode
export COINBASE_SANDBOX=0

# CDP Authentication (file-based)
export COINBASE_CDP_API_KEY="$(cat ~/.coinbase/cdp-api-key.txt 2>/dev/null)"
export COINBASE_CDP_PRIVATE_KEY="~/.coinbase/cdp-private-key.pem"

# Production Parameters
export COINBASE_MAX_POSITION_SIZE=0.01   # 0.01 BTC
export COINBASE_DAILY_LOSS_LIMIT=100     # $100 max loss
export COINBASE_MAX_IMPACT_BPS=15        # 15 bps max impact
export COINBASE_KILL_SWITCH=enabled

echo "✅ Advanced Trade PRODUCTION environment loaded"
echo "   ⚠️  LIVE TRADING ENABLED"
echo "   Max position: 0.01 BTC"
echo "   Daily loss limit: $100"
"""
    
    # Save corrected files
    with open('set_env.at_demo.sh', 'w') as f:
        f.write(demo_env)
    os.chmod('set_env.at_demo.sh', 0o755)
    
    with open('set_env.at_prod.sh', 'w') as f:
        f.write(prod_env)
    os.chmod('set_env.at_prod.sh', 0o755)
    
    print("\n✅ Created clarified environment files:")
    print("   • set_env.at_demo.sh - Demo trading (AT with safety)")
    print("   • set_env.at_prod.sh - Production trading")
    
    print("\n" + "="*70)
    print("💡 KEY TAKEAWAYS:")
    print("-"*70)
    print("""
1. Advanced Trade has NO public sandbox environment
2. Demo mode uses production endpoints with tiny position sizes
3. COINBASE_SANDBOX=1 is a safety flag, not an endpoint switch
4. Always use api.coinbase.com for Advanced Trade
5. CDP/JWT authentication is required for derivatives
""")
    
    # Create clarification document
    doc = {
        'advanced_trade': {
            'endpoints': {
                'api': 'api.coinbase.com',
                'websocket': 'advanced-trade-ws.coinbase.com'
            },
            'sandbox_available': False,
            'demo_approach': 'Use production with minimal sizes',
            'auth': 'JWT with CDP keys',
            'products': ['Spot', 'Perpetuals', 'Futures']
        },
        'exchange_api': {
            'endpoints': {
                'production': {
                    'api': 'api.exchange.coinbase.com',
                    'websocket': 'ws-direct.exchange.coinbase.com'
                },
                'sandbox': {
                    'api': 'api-public.sandbox.exchange.coinbase.com',
                    'websocket': 'ws-direct.sandbox.exchange.coinbase.com'
                }
            },
            'sandbox_available': True,
            'auth': 'HMAC with API key/secret/passphrase',
            'products': ['Spot only']
        },
        'recommendation': 'Use Advanced Trade for derivatives/perpetuals trading'
    }
    
    doc_path = Path('docs/ops/preflight/api_clarification.json')
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(doc_path, 'w') as f:
        json.dump(doc, f, indent=2)
    
    print(f"\n📄 Documentation saved to: {doc_path}")


def main():
    """Run clarification."""
    clarify_configuration()
    
    print("\n" + "="*70)
    print("✅ CONFIGURATION CLARIFIED")
    print("="*70)
    print("\nNext Steps:")
    print("1. Use the clarified environment file:")
    print("   source set_env.at_demo.sh  # For demo")
    print("\n2. Run enhanced preflight:")
    print("   python scripts/preflight_check_enhanced.py")
    print("\n3. Test with correct endpoints:")
    print("   python scripts/demo_run_validator.py")


if __name__ == "__main__":
    main()