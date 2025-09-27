#!/usr/bin/env python3
"""
Get Portfolio Balances - Fetch complete portfolio including cash and all assets.
"""

import os
import sys
from pathlib import Path
import json
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

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import create_cdp_auth_v2

def get_complete_portfolio():
    """Get complete portfolio including cash balances."""
    
    print("=" * 80)
    print("FETCHING COMPLETE COINBASE PORTFOLIO")
    print("=" * 80)
    
    # Create auth and client
    auth = create_cdp_auth_v2(
        api_key_name=os.getenv('COINBASE_CDP_API_KEY'),
        private_key_pem=os.getenv('COINBASE_CDP_PRIVATE_KEY')
    )
    
    client = CoinbaseClient(
        base_url='https://api.coinbase.com',
        auth=auth,
        api_version='2024-10-24'
    )
    
    # Method 1: Try portfolios endpoint
    print("\n1. CHECKING PORTFOLIOS ENDPOINT")
    print("=" * 60)
    
    try:
        portfolios = client.list_portfolios()
        if portfolios:
            print(f"✅ Found portfolios")
            
            # Save raw response
            with open('coinbase_portfolios_raw.json', 'w') as f:
                json.dump(portfolios, f, indent=2)
            print("📁 Saved to coinbase_portfolios_raw.json")
            
            # Parse portfolios
            if 'portfolios' in portfolios:
                for portfolio in portfolios['portfolios']:
                    print(f"\nPortfolio: {portfolio.get('name', 'Unnamed')}")
                    print(f"  UUID: {portfolio.get('uuid', 'N/A')[:20]}...")
                    print(f"  Type: {portfolio.get('type', 'N/A')}")
                    
                    # Check for balances in portfolio
                    if 'balances' in portfolio:
                        print("  Balances found in portfolio!")
                        for currency, balance in portfolio['balances'].items():
                            print(f"    {currency}: {balance}")
            else:
                print(f"Response: {json.dumps(portfolios, indent=2)[:500]}")
                
    except Exception as e:
        print(f"❌ Portfolios endpoint failed: {e}")
    
    # Method 2: Try different account endpoint paths
    print("\n2. TRYING ALTERNATE ACCOUNT ENDPOINTS")
    print("=" * 60)
    
    # Try v2 accounts (might have different data)
    try:
        print("\nTrying /v2/accounts...")
        response = client._request("GET", "/v2/accounts")
        if response:
            print(f"✅ Got response from v2 accounts")
            
            with open('coinbase_v2_accounts.json', 'w') as f:
                json.dump(response, f, indent=2)
            print("📁 Saved to coinbase_v2_accounts.json")
            
            if 'data' in response:
                accounts = response['data']
                print(f"Found {len(accounts)} accounts")
                
                # Show accounts with balances
                for acc in accounts[:10]:  # First 10
                    currency = acc.get('currency', {}).get('code', 'Unknown')
                    balance = acc.get('balance', {})
                    if balance:
                        amount = balance.get('amount', '0')
                        if amount != '0' and amount != '0.00000000':
                            print(f"  {currency}: {amount}")
                            
    except Exception as e:
        print(f"❌ V2 accounts failed: {e}")
    
    # Method 3: Try payment methods (might show USD balance)
    print("\n3. CHECKING PAYMENT METHODS")
    print("=" * 60)
    
    try:
        payment_methods = client.list_payment_methods()
        if payment_methods:
            print("✅ Found payment methods")
            
            with open('coinbase_payment_methods.json', 'w') as f:
                json.dump(payment_methods, f, indent=2)
            print("📁 Saved to coinbase_payment_methods.json")
            
            if 'payment_methods' in payment_methods:
                for pm in payment_methods['payment_methods']:
                    pm_type = pm.get('type', 'Unknown')
                    print(f"\nPayment Method: {pm_type}")
                    if 'fiat_account' in pm:
                        fiat = pm['fiat_account']
                        print(f"  Balance: {fiat.get('balance', 'N/A')}")
                        
    except Exception as e:
        print(f"❌ Payment methods failed: {e}")
    
    # Method 4: Get specific account details
    print("\n4. GETTING SPECIFIC ACCOUNT DETAILS")
    print("=" * 60)
    
    # First get the main portfolio ID
    accounts_response = client.get_accounts()
    if accounts_response and 'accounts' in accounts_response:
        # Get the retail portfolio ID from first account
        first_account = accounts_response['accounts'][0]
        portfolio_id = first_account.get('retail_portfolio_id')
        
        if portfolio_id:
            print(f"\nRetail Portfolio ID: {portfolio_id}")
            
            # Try to get portfolio details
            try:
                portfolio_detail = client.get_portfolio(portfolio_id)
                if portfolio_detail:
                    print("✅ Got portfolio details")
                    
                    with open('coinbase_portfolio_detail.json', 'w') as f:
                        json.dump(portfolio_detail, f, indent=2)
                    print("📁 Saved to coinbase_portfolio_detail.json")
                    
                    # Parse portfolio
                    if 'balances' in portfolio_detail:
                        print("\n💰 PORTFOLIO BALANCES:")
                        print("-" * 60)
                        
                        balances = portfolio_detail['balances']
                        if isinstance(balances, dict):
                            for currency, balance_info in balances.items():
                                if isinstance(balance_info, dict):
                                    amount = balance_info.get('amount', balance_info.get('value', '0'))
                                else:
                                    amount = str(balance_info)
                                
                                if amount != '0' and amount != '0.00000000':
                                    print(f"  {currency}: {amount}")
                        
                        # Look specifically for USD
                        if 'USD' in balances:
                            print(f"\n💵 USD BALANCE: {balances['USD']}")
                            
            except Exception as e:
                print(f"⚠️  Portfolio detail failed: {e}")
    
    # Method 5: Try to get all balances including USD through a different approach
    print("\n5. CHECKING ACCOUNT BALANCES WITH FILTERS")
    print("=" * 60)
    
    # Get accounts but look for ALL types
    try:
        # Make raw request to see full response
        print("\nMaking raw request to accounts endpoint...")
        raw_response = client._request("GET", "/api/v3/brokerage/accounts?include_deleted=false")
        
        if raw_response:
            with open('coinbase_accounts_full.json', 'w') as f:
                json.dump(raw_response, f, indent=2)
            print("📁 Saved full response to coinbase_accounts_full.json")
            
            if 'accounts' in raw_response:
                accounts = raw_response['accounts']
                
                # Group by currency type
                crypto_accounts = []
                fiat_accounts = []
                
                for acc in accounts:
                    currency = acc.get('currency', 'Unknown')
                    balance_obj = acc.get('available_balance', {})
                    
                    if isinstance(balance_obj, dict):
                        balance = balance_obj.get('value', '0')
                    else:
                        balance = str(balance_obj)
                    
                    try:
                        bal_decimal = Decimal(balance)
                        if bal_decimal > 0:
                            if currency in ['USD', 'EUR', 'GBP', 'CAD', 'AUD']:
                                fiat_accounts.append((currency, bal_decimal))
                            else:
                                crypto_accounts.append((currency, bal_decimal))
                    except:
                        pass
                
                if fiat_accounts:
                    print("\n💵 FIAT BALANCES:")
                    for currency, balance in fiat_accounts:
                        print(f"  {currency}: {balance:.2f}")
                
                if crypto_accounts:
                    print(f"\n🪙 CRYPTO BALANCES ({len(crypto_accounts)} assets):")
                    # Sort by balance
                    crypto_accounts.sort(key=lambda x: -x[1])
                    for currency, balance in crypto_accounts[:20]:  # Top 20
                        print(f"  {currency}: {balance:.8f}")
                        
    except Exception as e:
        print(f"⚠️  Full accounts request failed: {e}")
    
    print("\n" + "=" * 80)
    print("PORTFOLIO FETCH COMPLETE")
    print("Check the saved JSON files for complete raw data")
    print("=" * 80)

if __name__ == "__main__":
    get_complete_portfolio()