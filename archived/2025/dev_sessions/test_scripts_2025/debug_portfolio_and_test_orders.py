#!/usr/bin/env python3
"""
Debug portfolio breakdown and test order placement functionality.
"""

import os
import sys
from pathlib import Path
from decimal import Decimal
import json

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
from src.bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce

def debug_portfolio_issue():
    """Debug why portfolio breakdown returns 404."""
    
    print("=" * 80)
    print("DEBUGGING PORTFOLIO BREAKDOWN ISSUE")
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
    
    # Connect
    if not broker.connect():
        print("❌ Connection failed")
        return None
    
    print("✅ Connected successfully")
    
    # Get raw accounts data to find portfolio ID
    print("\n1. Getting accounts data to find portfolio ID...")
    try:
        raw_accounts = broker._client.get_accounts()
        print(f"   Raw accounts response keys: {list(raw_accounts.keys())}")
        
        accounts = raw_accounts.get("accounts", [])
        if accounts:
            first_account = accounts[0]
            print(f"\n   First account keys: {list(first_account.keys())}")
            
            # Look for portfolio ID
            portfolio_id = first_account.get("retail_portfolio_id")
            print(f"   Portfolio ID: {portfolio_id}")
            
            if portfolio_id:
                # Try direct API call
                print(f"\n2. Attempting direct portfolio breakdown for ID: {portfolio_id}")
                try:
                    breakdown = broker._client.get_portfolio_breakdown(portfolio_id)
                    print("   ✅ Direct API call succeeded!")
                    
                    # Show what we got
                    if breakdown:
                        breakdown_data = breakdown.get("breakdown", {})
                        print(f"   Breakdown keys: {list(breakdown_data.keys())}")
                        
                        # Check for spot positions
                        spot_positions = breakdown_data.get("spot_positions", [])
                        print(f"   Found {len(spot_positions)} spot positions")
                        
                        # Find USD
                        for pos in spot_positions:
                            if pos.get("asset") == "USD":
                                balance = pos.get("total_balance_crypto", {})
                                if isinstance(balance, dict):
                                    amount = balance.get("value", "0")
                                else:
                                    amount = balance
                                print(f"   💵 USD Balance: ${amount}")
                                break
                                
                except Exception as e:
                    print(f"   ❌ Direct API call failed: {e}")
                    
                # Try through adapter method
                print("\n3. Attempting through adapter method...")
                try:
                    portfolio_balances = broker.get_portfolio_balances()
                    print(f"   ✅ Adapter method returned {len(portfolio_balances)} balances")
                    
                    # Find USD
                    for bal in portfolio_balances:
                        if bal.asset == "USD":
                            print(f"   💵 USD Balance: ${bal.total:.2f}")
                            break
                            
                except Exception as e:
                    print(f"   ❌ Adapter method failed: {e}")
                    
        else:
            print("   No accounts found")
            
    except Exception as e:
        print(f"❌ Error getting accounts: {e}")
    
    broker.disconnect()
    return broker

def test_order_placement():
    """Test placing orders (in sandbox/paper mode)."""
    
    print("\n" + "=" * 80)
    print("TESTING ORDER PLACEMENT")
    print("=" * 80)
    
    # Create configuration
    config = APIConfig(
        api_key="",
        api_secret="",
        passphrase=None,
        base_url=os.getenv('COINBASE_API_BASE', 'https://api.coinbase.com'),
        sandbox=False,  # We'll use small amounts for testing
        ws_url=os.getenv('COINBASE_WS_URL', 'wss://advanced-trade-ws.coinbase.com'),
        cdp_api_key=os.getenv('COINBASE_CDP_API_KEY'),
        cdp_private_key=os.getenv('COINBASE_CDP_PRIVATE_KEY'),
        api_version=os.getenv('COINBASE_API_VERSION', '2024-10-24')
    )
    
    # Create brokerage adapter
    broker = CoinbaseBrokerage(config)
    
    # Connect
    if not broker.connect():
        print("❌ Connection failed")
        return
    
    print("✅ Connected successfully")
    
    # Get current market price for BTC
    print("\n1. Getting current BTC price...")
    try:
        quote = broker.get_quote("BTC-USD")
        current_price = quote.last
        print(f"   BTC-USD Current Price: ${current_price:,.2f}")
        print(f"   Bid: ${quote.bid:,.2f}, Ask: ${quote.ask:,.2f}")
        print(f"   Spread: ${(quote.ask - quote.bid):,.2f}")
    except Exception as e:
        print(f"❌ Failed to get quote: {e}")
        broker.disconnect()
        return
    
    # Test 1: Small limit buy order (well below market)
    print("\n2. Testing LIMIT BUY order (below market)...")
    try:
        # Place a limit buy at 10% below current price
        limit_price = current_price * Decimal("0.90")
        qty = Decimal("0.0001")  # Very small amount (~$10 at $100k BTC)
        
        print(f"   Placing limit buy: {qty} BTC at ${limit_price:,.2f}")
        
        order = broker.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=qty,
            price=limit_price,
            tif=TimeInForce.GTC
        )
        
        print(f"   ✅ Order placed successfully!")
        print(f"   Order ID: {order.id}")
        print(f"   Status: {order.status}")
        print(f"   Type: {order.type}")
        print(f"   Side: {order.side}")
        print(f"   Quantity: {order.qty}")
        print(f"   Price: ${order.price:,.2f}" if order.price else "Market")
        
        # Cancel the order immediately
        print("\n3. Cancelling test order...")
        if broker.cancel_order(order.id):
            print("   ✅ Order cancelled successfully")
        else:
            print("   ⚠️  Could not confirm cancellation")
            
    except Exception as e:
        print(f"   ❌ Order placement failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, '__dict__'):
            print(f"   Error details: {e.__dict__}")
    
    # Test 2: Check if we can place stop-loss orders
    print("\n4. Testing STOP-LOSS order capability...")
    try:
        # Try to place a stop-loss sell (if we had BTC)
        stop_price = current_price * Decimal("0.95")  # 5% below current
        qty = Decimal("0.0001")
        
        print(f"   Would place stop-loss: Sell {qty} BTC if price drops to ${stop_price:,.2f}")
        
        # Note: We won't actually place this unless we have BTC
        # Just demonstrating the API call structure
        print("   ⚠️  Not placing actual stop-loss (would require BTC holdings)")
        
    except Exception as e:
        print(f"   ❌ Stop-loss test failed: {e}")
    
    # Test 3: List any open orders
    print("\n5. Checking for open orders...")
    try:
        open_orders = broker.list_orders()
        if open_orders:
            print(f"   Found {len(open_orders)} open orders:")
            for order in open_orders[:5]:  # Show first 5
                print(f"   - {order.symbol}: {order.side} {order.qty} @ ${order.price:,.2f} if order.price else 'Market'")
        else:
            print("   No open orders")
    except Exception as e:
        print(f"   ❌ Failed to list orders: {e}")
    
    broker.disconnect()
    print("\n✅ Order placement test complete")

def prepare_paper_trading():
    """Prepare configuration for paper trading."""
    
    print("\n" + "=" * 80)
    print("PREPARING PAPER TRADING CONFIGURATION")
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
    
    # Connect
    if not broker.connect():
        print("❌ Connection failed")
        return
    
    print("✅ Connected successfully")
    
    # Get account balances
    print("\n1. Account Status:")
    try:
        # Try portfolio balances first
        try:
            balances = broker.get_portfolio_balances()
        except:
            # Fallback to regular balances
            balances = broker.list_balances()
        
        # Find USD balance
        usd_balance = Decimal("0")
        for bal in balances:
            if bal.asset == "USD":
                usd_balance = bal.available
                break
        
        print(f"   💵 Available USD: ${usd_balance:.2f}")
        
        # Calculate safe trading amounts
        if usd_balance > 0:
            # Conservative risk management
            max_position_size = usd_balance * Decimal("0.02")  # 2% per trade
            max_portfolio_risk = usd_balance * Decimal("0.10")  # 10% total risk
            
            print(f"\n2. Recommended Risk Limits:")
            print(f"   Max position size: ${max_position_size:.2f} (2% of capital)")
            print(f"   Max portfolio risk: ${max_portfolio_risk:.2f} (10% of capital)")
            print(f"   Suggested symbols: BTC-USD, ETH-USD (high liquidity)")
            
            print(f"\n3. Paper Trading Configuration:")
            paper_config = {
                "mode": "paper",
                "initial_capital": float(usd_balance),
                "max_position_size": float(max_position_size),
                "max_portfolio_risk": float(max_portfolio_risk),
                "symbols": ["BTC-USD", "ETH-USD"],
                "strategies": ["momentum", "mean_reversion"],
                "risk_management": {
                    "stop_loss": 0.02,  # 2% stop loss
                    "take_profit": 0.05,  # 5% take profit
                    "max_positions": 3,
                    "position_sizing": "kelly_criterion"
                }
            }
            
            print(json.dumps(paper_config, indent=2))
            
            # Save configuration
            config_path = Path(__file__).parent.parent / "config" / "paper_trading_config.json"
            config_path.parent.mkdir(exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(paper_config, f, indent=2)
            
            print(f"\n   ✅ Configuration saved to: {config_path}")
            
        else:
            print("   ⚠️  No USD balance found - using simulation mode")
            
    except Exception as e:
        print(f"❌ Error preparing paper trading: {e}")
    
    broker.disconnect()

if __name__ == "__main__":
    # Run all tests
    print("COINBASE INTEGRATION TEST SUITE")
    print("=" * 80)
    
    # Debug portfolio issue
    debug_portfolio_issue()
    
    # Test order placement
    test_order_placement()
    
    # Prepare paper trading
    prepare_paper_trading()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)