#!/usr/bin/env python3
"""
Test the fixed balance retrieval functionality.
"""

import os
import sys
from pathlib import Path
from decimal import Decimal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
env_file = Path(__file__).parent.parent / '.env.production'
if not env_file.exists():
    env_file = Path(__file__).parent.parent / '.env'

if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.strip().strip('"')
                if key == 'COINBASE_CDP_PRIVATE_KEY':
                    private_key_lines = [value] if value else []
                    for next_line in f:
                        next_line = next_line.strip()
                        private_key_lines.append(next_line)
                        if 'END EC PRIVATE KEY' in next_line:
                            break
                    value = '\n'.join(private_key_lines)
                os.environ[key] = value

from src.bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from src.bot_v2.features.brokerages.coinbase.models import APIConfig

def test_balance_retrieval():
    """Test the fixed balance retrieval."""
    
    print("=" * 80)
    print("TESTING FIXED BALANCE RETRIEVAL")
    print("=" * 80)
    
    # Create configuration
    config = APIConfig(
        api_key="",
        api_secret="",
        passphrase=None,
        base_url=os.getenv('COINBASE_API_BASE', 'https://api.coinbase.com'),
        sandbox=False,
        ws_url=os.getenv('COINBASE_WS_URL', 'wss://advanced-trade-ws.coinbase.com'),
        cdp_api_key=os.getenv('COINBASE_CDP_API_KEY'),
        cdp_private_key=os.getenv('COINBASE_CDP_PRIVATE_KEY'),
        api_version=os.getenv('COINBASE_API_VERSION', '2024-10-24')
    )
    
    # Create brokerage adapter
    broker = CoinbaseBrokerage(config)
    
    # Test connection
    print("\n1. Testing Connection...")
    if broker.connect():
        print("✅ Connected successfully")
        account_id = broker.get_account_id()
        print(f"   Account ID: {account_id}")
    else:
        print("❌ Connection failed")
        return
    
    # Test regular balance retrieval (crypto wallets)
    print("\n2. Testing Regular Balance Retrieval (Crypto Wallets)...")
    try:
        balances = broker.list_balances()
        if balances:
            print(f"✅ Retrieved {len(balances)} balances")
            
            # Show non-zero balances
            funded = [b for b in balances if b.total > 0]
            if funded:
                print("\n   Funded Wallets:")
                for bal in funded[:10]:  # First 10
                    print(f"   - {bal.asset}: {bal.total:.8f} (available: {bal.available:.8f}, hold: {bal.hold:.8f})")
            
            # Check for USD
            usd_balances = [b for b in balances if b.asset == 'USD']
            if usd_balances:
                usd = usd_balances[0]
                print(f"\n   💵 USD Balance: ${usd.total:.2f}")
        else:
            print("⚠️  No balances returned")
    except Exception as e:
        print(f"❌ Balance retrieval failed: {e}")
    
    # Test portfolio balance retrieval (includes USD)
    print("\n3. Testing Portfolio Balance Retrieval (Complete Portfolio)...")
    try:
        portfolio_balances = broker.get_portfolio_balances()
        if portfolio_balances:
            print(f"✅ Retrieved {len(portfolio_balances)} portfolio balances")
            
            # Show all balances
            total_usd_value = Decimal('0')
            
            # Separate USD and crypto
            usd_balances = [b for b in portfolio_balances if b.asset == 'USD']
            crypto_balances = [b for b in portfolio_balances if b.asset != 'USD' and b.total > 0]
            
            if usd_balances:
                for usd in usd_balances:
                    print(f"\n   💵 USD Balance: ${usd.total:.2f}")
                    print(f"      Available: ${usd.available:.2f}")
                    print(f"      Hold: ${usd.hold:.2f}")
                    total_usd_value += usd.total
            
            if crypto_balances:
                print(f"\n   🪙 Crypto Holdings ({len(crypto_balances)} assets):")
                # Sort by amount
                crypto_balances.sort(key=lambda x: -x.total)
                for bal in crypto_balances[:10]:  # Top 10
                    print(f"   - {bal.asset}: {bal.total:.8f}")
            
            print(f"\n   📊 Total USD Value: ${total_usd_value:.2f}")
        else:
            print("⚠️  No portfolio balances returned")
    except Exception as e:
        print(f"⚠️  Portfolio balance retrieval failed (expected if using fallback): {e}")
    
    # Test getting a quote
    print("\n4. Testing Market Data...")
    try:
        quote = broker.get_quote("BTC-USD")
        if quote:
            print(f"✅ BTC-USD Quote:")
            print(f"   Bid: ${quote.bid:.2f}")
            print(f"   Ask: ${quote.ask:.2f}")
            print(f"   Last: ${quote.last:.2f}")
            print(f"   Spread: ${(quote.ask - quote.bid):.2f}")
    except Exception as e:
        print(f"❌ Quote retrieval failed: {e}")
    
    # Disconnect
    broker.disconnect()
    print("\n✅ Test complete - disconnected")
    
    print("\n" + "=" * 80)
    print("BALANCE RETRIEVAL TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_balance_retrieval()