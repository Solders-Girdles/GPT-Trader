#!/usr/bin/env python3
"""
Orchestrator-Execution Integration Demo

Demonstrates how the execution simulator integrates with the existing
orchestrator system to provide realistic backtesting capabilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import logging
from datetime import datetime, timedelta
from decimal import Decimal

from bot.execution.orchestrator_bridge import ExecutionBridge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def simulate_strategy_trades(execution_bridge: ExecutionBridge):
    """Simulate trades from multiple strategies."""
    print("\n🎯 Simulating Strategy-Generated Trades")
    print("-" * 50)
    
    # Simulate trades from different strategies over time
    base_time = datetime.now()
    
    trades = [
        # Day 1: Momentum strategy entries
        {
            "timestamp": base_time,
            "symbol": "AAPL",
            "quantity": 200,
            "price": Decimal("175.50"),
            "strategy_id": "momentum_breakout",
            "reason": "Bullish breakout above resistance"
        },
        {
            "timestamp": base_time + timedelta(hours=1),
            "symbol": "MSFT", 
            "quantity": 150,
            "price": Decimal("380.25"),
            "strategy_id": "momentum_breakout",
            "reason": "Following strong earnings"
        },
        
        # Day 2: Mean reversion entries
        {
            "timestamp": base_time + timedelta(days=1),
            "symbol": "GOOGL",
            "quantity": 100,
            "price": Decimal("142.85"),
            "strategy_id": "mean_reversion",
            "reason": "Oversold bounce opportunity"
        },
        {
            "timestamp": base_time + timedelta(days=1, hours=2),
            "symbol": "TSLA",
            "quantity": 75,
            "price": Decimal("185.75"),
            "strategy_id": "mean_reversion", 
            "reason": "Support level hold"
        },
        
        # Day 3: Risk management exits
        {
            "timestamp": base_time + timedelta(days=2),
            "symbol": "AAPL",
            "quantity": -100,  # Partial exit
            "price": Decimal("180.25"),
            "strategy_id": "risk_management",
            "reason": "Taking profits at target"
        },
        {
            "timestamp": base_time + timedelta(days=2, hours=1),
            "symbol": "MSFT",
            "quantity": -50,  # Partial exit
            "price": Decimal("385.50"),
            "strategy_id": "risk_management",
            "reason": "Position sizing adjustment"
        },
        
        # Day 4: Trend following entries
        {
            "timestamp": base_time + timedelta(days=3),
            "symbol": "NVDA",
            "quantity": 80,
            "price": Decimal("875.50"),
            "strategy_id": "trend_following",
            "reason": "AI sector momentum"
        },
    ]
    
    # Execute trades
    execution_results = []
    for trade in trades:
        print(f"📈 {trade['strategy_id']}: {trade['symbol']} "
              f"{'BUY' if trade['quantity'] > 0 else 'SELL'} {abs(trade['quantity'])} "
              f"@ ${trade['price']:.2f}")
        
        result = execution_bridge.execute_trade(
            symbol=trade["symbol"],
            quantity=trade["quantity"],
            price=trade["price"],
            timestamp=trade["timestamp"],
            strategy_id=trade["strategy_id"],
            reason=trade["reason"]
        )
        
        execution_results.append(result)
        
        if result["success"]:
            fill_rate = result["executed_quantity"] / result["requested_quantity"]
            print(f"   ✅ Executed: {result['executed_quantity']}/{result['requested_quantity']} "
                  f"({fill_rate:.1%}) @ ${result['execution_price']:.2f}")
        else:
            print(f"   ❌ Failed: {result.get('reason', 'Unknown error')}")
    
    return execution_results


def simulate_market_updates(execution_bridge: ExecutionBridge):
    """Simulate market price updates."""
    print("\n📊 Simulating Market Price Updates")
    print("-" * 50)
    
    # Simulate market movements over time
    price_scenarios = [
        {
            "day": 1,
            "description": "Market opens positive",
            "prices": {
                "AAPL": 177.25,
                "MSFT": 382.50,
                "GOOGL": 144.20,
                "TSLA": 188.30,
                "NVDA": 880.75
            }
        },
        {
            "day": 2,
            "description": "Mid-day volatility",
            "prices": {
                "AAPL": 179.50,
                "MSFT": 385.75,
                "GOOGL": 141.95,
                "TSLA": 182.40,
                "NVDA": 895.25
            }
        },
        {
            "day": 3,
            "description": "Market correction",
            "prices": {
                "AAPL": 174.80,
                "MSFT": 378.25,
                "GOOGL": 138.60,
                "TSLA": 179.20,
                "NVDA": 865.40
            }
        },
        {
            "day": 4,
            "description": "Recovery rally",
            "prices": {
                "AAPL": 181.30,
                "MSFT": 390.80,
                "GOOGL": 145.75,
                "TSLA": 192.50,
                "NVDA": 920.15
            }
        }
    ]
    
    for scenario in price_scenarios:
        print(f"📅 Day {scenario['day']}: {scenario['description']}")
        
        execution_bridge.update_market_prices(
            scenario["prices"],
            datetime.now() + timedelta(days=scenario["day"])
        )
        
        # Show key prices
        for symbol, price in scenario["prices"].items():
            print(f"   {symbol}: ${price:.2f}")
        
        # Show portfolio impact
        portfolio_value = execution_bridge.get_portfolio_value()
        print(f"   Portfolio Value: ${portfolio_value:,.2f}")


def analyze_performance(execution_bridge: ExecutionBridge):
    """Analyze comprehensive performance metrics."""
    print("\n📈 Performance Analysis")
    print("-" * 50)
    
    # Get comprehensive metrics
    metrics = execution_bridge.get_performance_metrics()
    positions = execution_bridge.get_positions()
    trade_history = execution_bridge.get_trade_history()
    
    # Portfolio overview
    print("💼 Portfolio Overview:")
    print(f"   Value: ${metrics['portfolio_value']:,.2f}")
    print(f"   Cash: ${metrics['cash']:,.2f}")
    print(f"   Total Return: {metrics['total_return']:+.2%}")
    print(f"   Total P&L: ${metrics['total_pnl']:+,.2f}")
    
    # Performance metrics
    print("\n📊 Performance Metrics:")
    print(f"   Annualized Return: {metrics['annualized_return']:+.2%}")
    print(f"   Volatility: {metrics['volatility']:.2%}")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:+.2f}")
    print(f"   Max Drawdown: {metrics['max_drawdown']:+.2%}")
    print(f"   Calmar Ratio: {metrics['calmar_ratio']:+.2f}")
    
    # Trading statistics
    print("\n📋 Trading Statistics:")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.1%}")
    print(f"   Total Orders: {metrics['total_orders']}")
    print(f"   Avg Fill Rate: {metrics['avg_fill_rate']:.1%}")
    print(f"   Total Commission: ${metrics['total_commission']:.2f}")
    
    # Risk metrics
    print("\n⚠️  Risk Metrics:")
    print(f"   Position Count: {metrics['position_count']}")
    print(f"   Gross Exposure: ${metrics['gross_exposure']:,.2f}")
    print(f"   Net Exposure: ${metrics['net_exposure']:,.2f}")
    
    # Current positions
    print("\n📍 Current Positions:")
    if positions:
        for symbol, pos_data in positions.items():
            side_emoji = "📈" if pos_data["quantity"] > 0 else "📉"
            print(f"   {side_emoji} {symbol}: {pos_data['quantity']} shares @ ${pos_data['average_price']:.2f}")
            print(f"      Market Value: ${pos_data['market_value']:,.2f}")
            print(f"      Unrealized P&L: ${pos_data['unrealized_pnl']:+,.2f}")
    else:
        print("   No active positions")
    
    # Strategy attribution
    print("\n🎯 Strategy Attribution:")
    strategy_performance = {}
    for trade in trade_history:
        if trade["success"]:
            strategy = trade.get("strategy_id", "unknown")
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {"trades": 0, "volume": 0}
            strategy_performance[strategy]["trades"] += 1
            strategy_performance[strategy]["volume"] += trade["executed_quantity"] * trade["execution_price"]
    
    for strategy, stats in strategy_performance.items():
        print(f"   {strategy}: {stats['trades']} trades, ${stats['volume']:,.0f} volume")


def main():
    """Run the orchestrator-execution integration demo."""
    print("🚀 Orchestrator-Execution Integration Demo")
    print("=" * 60)
    print("Demonstrating realistic execution simulation within the orchestrator framework")
    
    try:
        # Create execution bridge
        execution_bridge = ExecutionBridge(
            initial_capital=Decimal("1000000"),
            commission_per_share=Decimal("0.005"),
            slippage_enabled=True
        )
        
        print(f"\n💰 Starting Portfolio: ${execution_bridge.get_portfolio_value():,.2f}")
        
        # Run simulation
        execution_results = simulate_strategy_trades(execution_bridge)
        simulate_market_updates(execution_bridge)
        analyze_performance(execution_bridge)
        
        print("\n" + "=" * 60)
        print("✅ Integration Demo Complete!")
        print("=" * 60)
        print("🎯 Key Integration Benefits:")
        print("• Realistic execution simulation with slippage & commissions")
        print("• Comprehensive position & P&L tracking")
        print("• Strategy performance attribution")
        print("• Risk metrics and drawdown analysis")
        print("• Seamless integration with existing orchestrator")
        print("\n🚀 Ready for:")
        print("• Enhanced backtesting realism")
        print("• Paper trading preparation")
        print("• Live trading simulation")
        print("• Strategy validation & optimization")
        
        # Show summary
        final_metrics = execution_bridge.get_performance_metrics()
        print(f"\n📊 Final Results:")
        print(f"   Portfolio: ${final_metrics['portfolio_value']:,.2f}")
        print(f"   Return: {final_metrics['total_return']:+.2%}")
        print(f"   Trades: {final_metrics['total_trades']}")
        print(f"   Fill Rate: {final_metrics['avg_fill_rate']:.1%}")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()