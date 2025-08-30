"""Alpaca Paper Trading Integration Demo.

This demo showcases the complete Alpaca paper trading integration including:
- Client authentication and connection
- Real-time market data streaming
- Order execution and management
- Integration with existing orchestrator
"""

import asyncio
import os
import time
from datetime import datetime, timedelta

from bot.brokers.alpaca import (
    AlpacaClient,
    AlpacaConfig, 
    AlpacaExecutor,
    AlpacaDataFeed,
    PaperTradingBridge,
    MarketDataConfig,
    PaperTradingConfig
)
from bot.logging import get_logger

logger = get_logger("alpaca_demo")


async def demo_basic_client():
    """Demo basic Alpaca client functionality."""
    print("\n=== Alpaca Client Demo ===")
    
    try:
        # Load configuration from environment
        config = AlpacaConfig.from_env()
        print(f"✓ Configuration loaded (paper={config.paper_trading})")
        
        # Initialize client
        with AlpacaClient(config) as client:
            print("✓ Client initialized and connected")
            
            # Get account information
            account = client.get_account()
            print(f"✓ Account: {account['id']}")
            print(f"  - Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"  - Buying Power: ${account['buying_power']:,.2f}")
            print(f"  - Cash: ${account['cash']:,.2f}")
            
            # Get positions
            positions = client.get_positions()
            print(f"✓ Positions: {len(positions)} total")
            for pos in positions[:5]:  # Show first 5
                print(f"  - {pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']:.2f}")
            
            # Get recent orders
            orders = client.get_orders(limit=5)
            print(f"✓ Recent orders: {len(orders)} found")
            for order in orders:
                print(f"  - {order['symbol']} {order['side']} {order['qty']} ({order['status']})")
            
            # Get historical data
            symbols = ["AAPL", "MSFT"]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            bars = client.get_historical_bars(symbols, start_date, end_date, "1Day")
            print(f"✓ Historical data retrieved for {len(symbols)} symbols")
            
            # Get latest quotes
            quotes = client.get_latest_quotes(symbols)
            print(f"✓ Latest quotes retrieved")
            
    except Exception as e:
        print(f"✗ Client demo failed: {e}")
        return False
    
    return True


async def demo_order_execution():
    """Demo order execution functionality."""
    print("\n=== Order Execution Demo ===")
    
    try:
        config = AlpacaConfig.from_env()
        
        with AlpacaClient(config) as client:
            executor = AlpacaExecutor(client)
            
            # Demo symbol (use a low-priced stock for testing)
            symbol = "SIRI"  # Sirius XM - typically under $10
            qty = 1
            
            print(f"✓ Testing orders for {symbol}")
            
            # Submit a limit buy order (well below market price)
            result = executor.submit_limit_order(
                symbol=symbol,
                side="buy",
                qty=qty,
                limit_price=1.00,  # Very low price, won't execute
                time_in_force="day"
            )
            
            if result.success:
                order_id = result.order_id
                print(f"✓ Limit order submitted: {order_id}")
                
                # Check order status
                await asyncio.sleep(1)
                status = executor.get_order_status(order_id)
                if status:
                    print(f"  - Status: {status['status']}")
                    print(f"  - Filled: {status['filled_qty']}/{status['qty']}")
                
                # Cancel the order
                cancel_result = executor.cancel_order(order_id)
                if cancel_result.success:
                    print("✓ Order canceled successfully")
                else:
                    print(f"✗ Cancel failed: {cancel_result.error}")
            else:
                print(f"✗ Order submission failed: {result.error}")
            
            # Get open orders
            open_orders = executor.get_open_orders()
            print(f"✓ Open orders: {len(open_orders)} found")
            
    except Exception as e:
        print(f"✗ Execution demo failed: {e}")
        return False
    
    return True


async def demo_data_feed():
    """Demo real-time data feed."""
    print("\n=== Real-time Data Feed Demo ===")
    
    try:
        config = AlpacaConfig.from_env()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Data received counters
        quote_count = 0
        trade_count = 0
        
        def handle_quote(data_point):
            nonlocal quote_count
            quote_count += 1
            if quote_count <= 3:  # Only print first few
                print(f"  📊 Quote: {data_point.symbol} "
                      f"${data_point.data['bid_price']:.2f} x ${data_point.data['ask_price']:.2f}")
        
        def handle_trade(data_point):
            nonlocal trade_count
            trade_count += 1
            if trade_count <= 3:  # Only print first few
                print(f"  💰 Trade: {data_point.symbol} "
                      f"${data_point.data['price']:.2f} ({data_point.data['size']} shares)")
        
        with AlpacaClient(config) as client:
            data_config = MarketDataConfig(
                symbols=symbols,
                subscribe_quotes=True,
                subscribe_trades=True
            )
            
            async with AlpacaDataFeed(client, data_config) as feed:
                print(f"✓ Data feed connected for {len(symbols)} symbols")
                
                # Add handlers
                feed.add_quote_handler(handle_quote)
                feed.add_trade_handler(handle_trade)
                
                # Collect data for a short time
                print("  Collecting data for 10 seconds...")
                await asyncio.sleep(10)
                
                print(f"✓ Received {quote_count} quotes and {trade_count} trades")
                
                # Show latest data
                for symbol in symbols:
                    latest_quote = feed.get_latest_data(symbol, "quote")
                    if latest_quote:
                        data = latest_quote.data
                        print(f"  - {symbol}: ${data['bid_price']:.2f} x ${data['ask_price']:.2f}")
    
    except Exception as e:
        print(f"✗ Data feed demo failed: {e}")
        return False
    
    return True


async def demo_paper_trading_bridge():
    """Demo the complete paper trading bridge."""
    print("\n=== Paper Trading Bridge Demo ===")
    
    try:
        # Configuration
        alpaca_config = AlpacaConfig.from_env()
        config = PaperTradingConfig(
            alpaca_config=alpaca_config,
            enable_real_time_data=True,
            data_symbols=["AAPL", "MSFT"],
            simulate_execution_delay=True,
            max_order_value=1000.0,  # Low limit for demo
            log_all_orders=True
        )
        
        async with PaperTradingBridge(config) as bridge:
            print("✓ Paper trading bridge started")
            
            # Get account info
            account = bridge.get_account()
            print(f"  - Account: {account.id}")
            print(f"  - Portfolio Value: ${account.portfolio_value:,.2f}")
            
            # Get positions
            positions = bridge.get_positions()
            print(f"  - Positions: {len(positions)}")
            
            # Demo order submission
            symbol = "SIRI"
            
            # Try a small market order (will fail due to low order value limit)
            result = bridge.submit_order(
                symbol=symbol,
                side="buy",
                qty=100,
                order_type="market"
            )
            
            if result.success:
                print(f"✓ Market order submitted: {result.order_id}")
            else:
                print(f"  ⚠ Market order blocked: {result.error}")
            
            # Try a limit order with lower value
            result = bridge.submit_order(
                symbol=symbol,
                side="buy", 
                qty=10,
                order_type="limit",
                limit_price=1.00
            )
            
            if result.success:
                print(f"✓ Limit order submitted: {result.order_id}")
                
                # Check status
                await asyncio.sleep(1)
                status = bridge.get_order_status(result.order_id)
                if status:
                    print(f"  - Status: {status['status']}")
                
                # Cancel it
                cancel_result = bridge.cancel_order(result.order_id)
                if cancel_result.success:
                    print("✓ Order canceled")
            else:
                print(f"  ⚠ Limit order failed: {result.error}")
            
            # Show execution metrics
            metrics = bridge.get_execution_metrics()
            print(f"✓ Execution metrics:")
            print(f"  - Total orders: {metrics.total_orders}")
            print(f"  - Success rate: {metrics.success_rate:.1%}")
            print(f"  - Avg execution time: {metrics.avg_execution_time_ms:.1f}ms")
    
    except Exception as e:
        print(f"✗ Bridge demo failed: {e}")
        return False
    
    return True


async def main():
    """Run all demos."""
    print("🚀 Alpaca Paper Trading Integration Demo")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"✗ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        print("export ALPACA_API_KEY='your_api_key'")
        print("export ALPACA_SECRET_KEY='your_secret_key'")
        print("export ALPACA_PAPER='true'  # Optional, defaults to true")
        return
    
    print("✓ Environment variables configured")
    
    # Run demos
    demos = [
        ("Basic Client", demo_basic_client),
        ("Order Execution", demo_order_execution),
        ("Data Feed", demo_data_feed),
        ("Paper Trading Bridge", demo_paper_trading_bridge),
    ]
    
    results = []
    for name, demo_func in demos:
        print(f"\n🔄 Running {name} demo...")
        try:
            success = await demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ {name} demo failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Demo Results Summary")
    print("=" * 50)
    
    passed = 0
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} - {name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} demos passed")
    
    if passed == len(results):
        print("🎉 All demos completed successfully!")
        print("\nNext steps:")
        print("1. Configure your Alpaca API keys")
        print("2. Test with small position sizes first")
        print("3. Integrate with your existing strategies")
        print("4. Monitor execution metrics and logs")
    else:
        print("⚠️  Some demos failed. Check logs for details.")


if __name__ == "__main__":
    asyncio.run(main())