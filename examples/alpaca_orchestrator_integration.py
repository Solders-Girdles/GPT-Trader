"""Integration Example: Alpaca Paper Trading with Orchestrator.

This example demonstrates how to integrate the new Alpaca paper trading system
with the existing GPT-Trader orchestrator for end-to-end strategy execution.
"""

import asyncio
import os
from datetime import datetime, timedelta

from bot.brokers.alpaca import (
    AlpacaConfig,
    PaperTradingBridge,
    PaperTradingConfig,
)
from bot.integration.orchestrator import IntegratedOrchestrator, BacktestConfig
from bot.strategy.demo_ma import DemoMovingAverageStrategy
from bot.logging import get_logger

logger = get_logger("alpaca_integration_example")


class AlpacaIntegratedOrchestrator(IntegratedOrchestrator):
    """Extended orchestrator with Alpaca paper trading capabilities."""
    
    def __init__(self, alpaca_bridge: PaperTradingBridge):
        """Initialize with Alpaca bridge."""
        super().__init__()
        self.alpaca_bridge = alpaca_bridge
        self._live_mode = False
    
    async def start_live_trading(self):
        """Start live paper trading mode."""
        try:
            await self.alpaca_bridge.start()
            self._live_mode = True
            logger.info("Live trading mode activated with Alpaca paper trading")
            return True
        except Exception as e:
            logger.error(f"Failed to start live trading: {e}")
            return False
    
    async def stop_live_trading(self):
        """Stop live trading mode."""
        if self._live_mode:
            await self.alpaca_bridge.stop()
            self._live_mode = False
            logger.info("Live trading mode deactivated")
    
    def execute_trade(self, symbol: str, side: str, qty: int, price: float = None) -> bool:
        """Execute trade through Alpaca if in live mode, otherwise use simulator."""
        if self._live_mode:
            # Use Alpaca for actual execution
            order_type = "limit" if price else "market"
            result = self.alpaca_bridge.submit_order(
                symbol=symbol,
                side=side,
                qty=qty,
                order_type=order_type,
                limit_price=price
            )
            
            if result.success:
                logger.info(f"Alpaca order executed: {symbol} {side} {qty} (ID: {result.order_id})")
                return True
            else:
                logger.error(f"Alpaca order failed: {result.error}")
                return False
        else:
            # Use simulator for backtesting
            return super().execute_trade(symbol, side, qty, price)
    
    def get_live_positions(self) -> dict:
        """Get current live positions from Alpaca."""
        if not self._live_mode:
            return {}
        
        positions = self.alpaca_bridge.get_positions()
        return {pos.symbol: pos for pos in positions}
    
    def get_live_account_info(self) -> dict:
        """Get live account information."""
        if not self._live_mode:
            return {}
        
        account = self.alpaca_bridge.get_account()
        return {
            "account_id": account.id,
            "portfolio_value": account.portfolio_value,
            "buying_power": account.buying_power,
            "cash": account.cash,
            "equity": account.equity,
        }
    
    def get_execution_metrics(self) -> dict:
        """Get execution performance metrics."""
        if not self._live_mode:
            return {}
        
        metrics = self.alpaca_bridge.get_execution_metrics()
        return {
            "total_orders": metrics.total_orders,
            "success_rate": metrics.success_rate,
            "avg_execution_time_ms": metrics.avg_execution_time_ms,
            "total_volume": metrics.total_volume,
            "total_notional": metrics.total_notional,
        }


async def run_integrated_example():
    """Run complete integration example."""
    print("🚀 Alpaca-Orchestrator Integration Example")
    print("=" * 50)
    
    # Check environment
    required_vars = ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nSet them with:")
        print("export ALPACA_API_KEY='your_key'")
        print("export ALPACA_SECRET_KEY='your_secret'")
        return
    
    # Configuration
    alpaca_config = AlpacaConfig.from_env()
    paper_config = PaperTradingConfig(
        alpaca_config=alpaca_config,
        enable_real_time_data=False,  # Disable for this example
        data_symbols=[],
        max_order_value=500.0,  # Small orders for demo
        max_daily_trades=10,
        log_all_orders=True,
    )
    
    # Initialize components
    async with PaperTradingBridge(paper_config) as alpaca_bridge:
        orchestrator = AlpacaIntegratedOrchestrator(alpaca_bridge)
        
        # Show account info
        account_info = orchestrator.get_live_account_info()
        print(f"✅ Connected to account: {account_info.get('account_id', 'Unknown')}")
        print(f"   Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
        print(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
        
        # Demo 1: Run a backtest first
        print("\n📊 Running backtest for strategy validation...")
        
        strategy = DemoMovingAverageStrategy()
        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now() - timedelta(days=1),
            initial_capital=10000.0,
            show_progress=False,
            quiet_mode=True,
        )
        
        try:
            results = orchestrator.run_backtest(
                strategy=strategy,
                symbols=["AAPL"],
                config=config
            )
            
            print(f"   Total Return: {results.total_return:.2%}")
            print(f"   Sharpe Ratio: {results.sharpe_ratio:.2f}")
            print(f"   Max Drawdown: {results.max_drawdown:.2%}")
            print(f"   Total Trades: {results.total_trades}")
            
        except Exception as e:
            print(f"   Backtest failed: {e}")
        
        # Demo 2: Switch to live mode
        print("\n🔴 Switching to live paper trading mode...")
        
        success = await orchestrator.start_live_trading()
        if not success:
            print("❌ Failed to start live trading")
            return
        
        print("✅ Live trading mode activated")
        
        # Demo 3: Execute a small test order
        print("\n📈 Executing test order...")
        
        # Use a penny stock for testing
        test_symbol = "SIRI"
        test_qty = 1
        
        # Get latest quote first (if available)
        try:
            quote = alpaca_bridge.get_latest_quote(test_symbol)
            if quote:
                current_price = (quote["bid_price"] + quote["ask_price"]) / 2
                print(f"   Current price for {test_symbol}: ${current_price:.2f}")
            else:
                print(f"   Could not get quote for {test_symbol}")
        except Exception as e:
            print(f"   Quote error: {e}")
        
        # Submit a conservative limit order
        success = orchestrator.execute_trade(
            symbol=test_symbol,
            side="buy",
            qty=test_qty,
            price=1.00  # Very low limit price, likely won't fill
        )
        
        if success:
            print("✅ Test order submitted successfully")
        else:
            print("❌ Test order failed")
        
        # Demo 4: Check order status
        print("\n📋 Checking open orders...")
        
        open_orders = alpaca_bridge.get_open_orders()
        print(f"   Open orders: {len(open_orders)}")
        
        for order in open_orders[:3]:  # Show first 3
            print(f"   - {order['symbol']} {order['side']} {order['qty']} @ ${order.get('limit_price', 'market')} ({order['status']})")
        
        # Demo 5: Show execution metrics
        print("\n📊 Execution metrics:")
        
        metrics = orchestrator.get_execution_metrics()
        print(f"   Total Orders: {metrics.get('total_orders', 0)}")
        print(f"   Success Rate: {metrics.get('success_rate', 0):.1%}")
        print(f"   Avg Execution Time: {metrics.get('avg_execution_time_ms', 0):.1f}ms")
        
        # Demo 6: Cancel any open orders
        if open_orders:
            print("\n🚫 Canceling test orders...")
            
            for order in open_orders:
                cancel_result = alpaca_bridge.cancel_order(order['id'])
                if cancel_result.success:
                    print(f"   ✅ Canceled order {order['id']}")
                else:
                    print(f"   ❌ Failed to cancel {order['id']}: {cancel_result.error}")
        
        # Demo 7: Show final positions
        print("\n💼 Current positions:")
        
        positions = orchestrator.get_live_positions()
        if positions:
            for symbol, position in positions.items():
                print(f"   - {symbol}: {position.qty} shares @ ${position.current_price:.2f}")
                print(f"     Market Value: ${position.market_value:.2f}")
                print(f"     P&L: ${position.unrealized_pl:.2f} ({position.unrealized_plpc:.2%})")
        else:
            print("   No positions")
        
        # Stop live trading
        await orchestrator.stop_live_trading()
        print("\n✅ Live trading stopped")
    
    print("\n🎉 Integration example completed!")
    print("\nNext steps:")
    print("1. Customize risk limits in PaperTradingConfig")
    print("2. Add your own strategies")
    print("3. Implement real-time data feeds")
    print("4. Set up monitoring and alerts")


def demo_configuration_examples():
    """Show different configuration examples."""
    print("\n⚙️ Configuration Examples")
    print("=" * 30)
    
    # Conservative config for testing
    print("1. Conservative Testing Config:")
    conservative_config = PaperTradingConfig(
        alpaca_config=AlpacaConfig.from_env(),
        enable_real_time_data=False,
        simulate_execution_delay=True,
        max_order_value=100.0,
        max_daily_trades=5,
        log_all_orders=True,
    )
    print(f"   Max order value: ${conservative_config.max_order_value}")
    print(f"   Max daily trades: {conservative_config.max_daily_trades}")
    
    # Aggressive config for active strategies
    print("\n2. Active Trading Config:")
    active_config = PaperTradingConfig(
        alpaca_config=AlpacaConfig.from_env(),
        enable_real_time_data=True,
        data_symbols=["AAPL", "MSFT", "GOOGL", "SPY", "QQQ"],
        simulate_execution_delay=False,
        max_order_value=10000.0,
        max_daily_trades=100,
        log_all_orders=False,  # Reduce noise
    )
    print(f"   Max order value: ${active_config.max_order_value}")
    print(f"   Max daily trades: {active_config.max_daily_trades}")
    print(f"   Real-time symbols: {len(active_config.data_symbols)}")


if __name__ == "__main__":
    # Show configuration examples
    try:
        demo_configuration_examples()
    except ValueError as e:
        print(f"Configuration demo skipped: {e}")
    
    # Run main example
    asyncio.run(run_integrated_example())