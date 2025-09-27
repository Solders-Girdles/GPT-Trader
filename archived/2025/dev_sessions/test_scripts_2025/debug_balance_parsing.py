#!/usr/bin/env python3
"""
Debug Balance Parsing - Find the correct fields for all balances including cash.
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

from src.bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from src.bot_v2.features.brokerages.coinbase.cdp_auth_v2 import create_cdp_auth_v2

def debug_balance_fields():
    """Debug all balance fields to find correct parsing."""
    
    print("=" * 80)
    print("DEBUGGING COINBASE BALANCE FIELDS")
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
    
    print("\nFetching accounts...")
    result = client.get_accounts()
    
    if result and 'accounts' in result:
        accounts = result['accounts']
        print(f"✅ Found {len(accounts)} accounts\n")
        
        # Save raw response for analysis
        with open('coinbase_accounts_raw.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("📁 Saved raw response to coinbase_accounts_raw.json\n")
        
        # Analyze different account types
        print("ANALYZING ACCOUNT STRUCTURES:")
        print("=" * 60)
        
        # Group accounts by structure
        structures = {}
        all_fields = set()
        
        for i, account in enumerate(accounts):
            # Collect all field names
            all_fields.update(account.keys())
            
            # Create structure signature
            structure = tuple(sorted(account.keys()))
            if structure not in structures:
                structures[structure] = []
            structures[structure].append(account)
        
        print(f"\nFound {len(structures)} different account structures")
        print(f"All fields seen: {sorted(all_fields)}\n")
        
        # Show samples of each structure type
        for struct_num, (structure, accts) in enumerate(structures.items(), 1):
            print(f"\n{'='*60}")
            print(f"STRUCTURE #{struct_num} ({len(accts)} accounts)")
            print(f"{'='*60}")
            
            # Show first account of this type
            sample = accts[0]
            currency = sample.get('currency', 'Unknown')
            print(f"\nSample Account: {currency}")
            print(f"Name: {sample.get('name', 'N/A')}")
            
            # Show all balance-related fields
            print("\nBalance Fields:")
            for key in sorted(sample.keys()):
                if 'balance' in key.lower() or 'amount' in key.lower() or key in ['value', 'hold']:
                    value = sample[key]
                    print(f"  {key}: {value}")
            
            # Try to extract balance
            print("\nAttempting to parse balance:")
            
            # Method 1: available_balance
            if 'available_balance' in sample:
                bal = sample['available_balance']
                print(f"  available_balance: {bal}")
                if isinstance(bal, dict):
                    print(f"    - value: {bal.get('value')}")
                    print(f"    - currency: {bal.get('currency')}")
            
            # Method 2: balance field
            if 'balance' in sample:
                bal = sample['balance']
                print(f"  balance: {bal}")
                if isinstance(bal, dict):
                    print(f"    - value: {bal.get('value')}")
                    print(f"    - amount: {bal.get('amount')}")
            
            # Method 3: Direct value field
            if 'value' in sample:
                print(f"  value: {sample['value']}")
            
            # Method 4: hold field
            if 'hold' in sample:
                print(f"  hold: {sample['hold']}")
                if isinstance(sample['hold'], dict):
                    print(f"    - value: {sample['hold'].get('value')}")
        
        # Now try to get ALL balances correctly
        print("\n" + "="*80)
        print("EXTRACTING ALL BALANCES")
        print("="*80)
        
        all_balances = []
        
        for account in accounts:
            currency = account.get('currency', 'Unknown')
            name = account.get('name', currency)
            
            balance = None
            hold = None
            
            # Try all possible balance extraction methods
            
            # Method 1: available_balance.value
            if 'available_balance' in account:
                ab = account['available_balance']
                if isinstance(ab, dict) and 'value' in ab:
                    try:
                        balance = Decimal(str(ab['value']))
                    except:
                        pass
                elif isinstance(ab, (str, int, float)):
                    try:
                        balance = Decimal(str(ab))
                    except:
                        pass
            
            # Method 2: balance.value or balance.amount
            if balance is None and 'balance' in account:
                b = account['balance']
                if isinstance(b, dict):
                    if 'value' in b:
                        try:
                            balance = Decimal(str(b['value']))
                        except:
                            pass
                    elif 'amount' in b:
                        try:
                            balance = Decimal(str(b['amount']))
                        except:
                            pass
                elif isinstance(b, (str, int, float)):
                    try:
                        balance = Decimal(str(b))
                    except:
                        pass
            
            # Method 3: Direct value field
            if balance is None and 'value' in account:
                try:
                    balance = Decimal(str(account['value']))
                except:
                    pass
            
            # Get hold amount
            if 'hold' in account:
                h = account['hold']
                if isinstance(h, dict) and 'value' in h:
                    try:
                        hold = Decimal(str(h['value']))
                    except:
                        pass
                elif isinstance(h, (str, int, float)):
                    try:
                        hold = Decimal(str(h))
                    except:
                        pass
            
            # Store balance info
            if balance is not None:
                all_balances.append({
                    'currency': currency,
                    'name': name,
                    'balance': balance,
                    'hold': hold or Decimal('0'),
                    'uuid': account.get('uuid', 'N/A')[:12]
                })
        
        # Sort and display
        all_balances.sort(key=lambda x: (-x['balance'], x['currency']))
        
        print(f"\n✅ Successfully parsed {len(all_balances)} balances")
        
        # Show all non-zero balances
        funded = [b for b in all_balances if b['balance'] > 0]
        
        if funded:
            print(f"\n💰 ALL FUNDED ACCOUNTS ({len(funded)} total):")
            print("-" * 80)
            print(f"{'Currency':<10} {'Balance':>20} {'Hold':>20} {'Name':<20}")
            print("-" * 80)
            
            total_usd = Decimal('0')
            
            for acc in funded:
                print(f"{acc['currency']:<10} {acc['balance']:>20.8f} {acc['hold']:>20.8f} {acc['name'][:20]:<20}")
                
                # Count USD equivalents
                if acc['currency'] in ['USD', 'USDC', 'USDT', 'DAI', 'BUSD']:
                    total_usd += acc['balance']
            
            print("-" * 80)
            print(f"{'USD/Stable':<10} ${total_usd:>19.2f}")
            
            # Look specifically for USD
            usd_accounts = [b for b in all_balances if 'USD' in b['currency']]
            if usd_accounts:
                print(f"\n💵 USD-RELATED ACCOUNTS:")
                for acc in usd_accounts:
                    print(f"  {acc['currency']}: ${acc['balance']:.2f}")
        
        # Check for zero balances that might have holds
        with_holds = [b for b in all_balances if b['hold'] > 0]
        if with_holds:
            print(f"\n🔒 ACCOUNTS WITH HOLDS:")
            for acc in with_holds:
                print(f"  {acc['currency']}: Hold = {acc['hold']}")
        
        print("\n" + "="*80)
        
        # Show first few raw accounts for manual inspection
        print("\n📋 RAW ACCOUNT SAMPLES (first 3):")
        print("="*60)
        for i, account in enumerate(accounts[:3]):
            print(f"\nAccount #{i+1}: {account.get('currency', 'Unknown')}")
            print(json.dumps(account, indent=2))
            
    else:
        print("❌ Failed to retrieve accounts")

if __name__ == "__main__":
    debug_balance_fields()