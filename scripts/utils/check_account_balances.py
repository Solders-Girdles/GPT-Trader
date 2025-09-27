#!/usr/bin/env python3
"""
Check Coinbase Account Balances
Verifies the integration is working by retrieving and displaying all account balances.
"""

import os
import sys
from pathlib import Path
from decimal import Decimal, InvalidOperation
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load production environment
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

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import create_cdp_auth_v2

def get_account_balances():
    """Get and display all account balances."""
    
    print("=" * 80)
    print("COINBASE ACCOUNT BALANCE CHECK")
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
    
    print(f"\n📋 Configuration:")
    print(f"   API Version: {config.api_version}")
    print(f"   CDP Key: {config.cdp_api_key[:50] if config.cdp_api_key else 'Not set'}...")
    
    # Method 1: Direct API Client
    print("\n" + "=" * 60)
    print("METHOD 1: DIRECT API CLIENT")
    print("=" * 60)
    
    auth = create_cdp_auth_v2(
        api_key_name=config.cdp_api_key,
        private_key_pem=config.cdp_private_key
    )
    
    client = CoinbaseClient(
        base_url=config.base_url,
        auth=auth,
        api_version=config.api_version
    )
    
    try:
        # Get accounts
        result = client.get_accounts()
        
        if result and 'accounts' in result:
            accounts = result['accounts']
            print(f"\n✅ Found {len(accounts)} accounts")
            
            # Calculate total balances
            total_usd = Decimal('0')
            funded_accounts = []
            
            for account in accounts:
                try:
                    currency = account.get('currency', 'Unknown')
                    
                    # Try different balance field names
                    balance_str = None
                    if 'available_balance' in account:
                        balance_data = account['available_balance']
                        if isinstance(balance_data, dict) and 'value' in balance_data:
                            balance_str = balance_data['value']
                        elif isinstance(balance_data, (str, int, float)):
                            balance_str = str(balance_data)
                    elif 'balance' in account:
                        balance_data = account['balance']
                        if isinstance(balance_data, dict) and 'value' in balance_data:
                            balance_str = balance_data['value']
                        elif isinstance(balance_data, (str, int, float)):
                            balance_str = str(balance_data)
                    
                    if balance_str:
                        # Clean up balance string
                        balance_str = balance_str.strip()
                        if balance_str and balance_str != '0':
                            try:
                                balance = Decimal(balance_str)
                                if balance > 0:
                                    funded_accounts.append({
                                        'currency': currency,
                                        'balance': balance,
                                        'name': account.get('name', currency),
                                        'uuid': account.get('uuid', 'N/A')
                                    })
                            except (InvalidOperation, ValueError) as e:
                                # Skip accounts with unparseable balances
                                pass
                                
                except Exception as e:
                    # Skip problematic accounts
                    continue
            
            # Sort by currency name
            funded_accounts.sort(key=lambda x: x['currency'])
            
            # Display funded accounts
            if funded_accounts:
                print(f"\n💰 FUNDED ACCOUNTS ({len(funded_accounts)} total):")
                print("-" * 60)
                
                for acc in funded_accounts:
                    print(f"  {acc['currency']:8s} : {acc['balance']:>20.8f}")
                    if acc['currency'] in ['USD', 'USDC', 'USDT']:
                        total_usd += acc['balance']
                
                print("-" * 60)
                print(f"  {'USD Total':8s} : ${total_usd:>19.2f}")
                
            else:
                print("\n⚠️  No funded accounts found")
                
            # Show account summary
            print(f"\n📊 ACCOUNT SUMMARY:")
            print(f"   Total Accounts: {len(accounts)}")
            print(f"   Funded Accounts: {len(funded_accounts)}")
            print(f"   USD Balance: ${total_usd:.2f}")
            
        else:
            print("❌ Failed to retrieve accounts")
            if result:
                print(f"   Response: {json.dumps(result, indent=2)[:500]}")
                
    except Exception as e:
        print(f"❌ Error getting accounts: {e}")
    
    # Method 2: Using Adapter
    print("\n" + "=" * 60)
    print("METHOD 2: BROKERAGE ADAPTER")
    print("=" * 60)
    
    try:
        broker = CoinbaseBrokerage(config)
        
        if broker.connect():
            print("✅ Connected to Coinbase")
            
            # Get account ID
            account_id = broker.get_account_id()
            print(f"   Account ID: {account_id}")
            
            # Try to get balances through adapter
            print("\n📊 Getting balances through adapter...")
            try:
                balances = broker.list_balances()
                if balances:
                    print(f"✅ Retrieved {len(balances)} balances")
                    
                    # Show non-zero balances
                    funded = [b for b in balances if b.total > 0]
                    if funded:
                        print("\n💰 FUNDED BALANCES:")
                        for bal in funded[:20]:  # Show first 20
                            print(f"   {bal.asset}: {bal.total}")
                else:
                    print("⚠️  No balances returned")
                    
            except Exception as e:
                print(f"⚠️  Adapter balance retrieval issue: {e}")
                print("   (This is the decimal parsing issue we identified)")
            
            broker.disconnect()
            
        else:
            print("❌ Failed to connect through adapter")
            
    except Exception as e:
        print(f"❌ Adapter error: {e}")
    
    # Get some market prices for context
    print("\n" + "=" * 60)
    print("MARKET PRICES (for reference)")
    print("=" * 60)
    
    try:
        symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
        for symbol in symbols:
            ticker = client.get_ticker(symbol)
            if ticker and 'price' in ticker:
                price = float(ticker['price'])
                print(f"   {symbol}: ${price:,.2f}")
    except Exception as e:
        print(f"⚠️  Could not get market prices: {e}")
    
    print("\n" + "=" * 80)
    print("BALANCE CHECK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    get_account_balances()