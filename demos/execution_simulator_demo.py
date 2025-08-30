#!/usr/bin/env python3
"""
Execution Simulator Demo

Demonstrates the realistic execution simulation capabilities including:
- Order lifecycle management with realistic fills
- Position tracking with P&L calculation
- Portfolio state management
- Risk metrics and performance attribution

This simulation prepares the system for paper trading integration.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict

from bot.execution import (
    OrderManager, 
    Order, 
    OrderType, 
    OrderSide,
    PositionTracker,
    PortfolioState
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_market_data() -> Dict[str, Decimal]:
    """Create sample market data for simulation."""
    return {
        "AAPL": Decimal("175.50"),
        "MSFT": Decimal("380.25"),
        "GOOGL": Decimal("142.85"),
        "TSLA": Decimal("185.75"),
        "NVDA": Decimal("875.50"),
    }


def demo_order_execution():
    """Demonstrate realistic order execution simulation."""
    print("\n" + "="*60)
    print("🚀 Order Execution Simulation Demo")
    print("="*60)
    
    # Create order manager
    order_manager = OrderManager(
        commission_per_share=Decimal("0.005"),
        min_commission=Decimal("1.0"),
        partial_fill_prob=0.2,  # 20% chance of partial fills
    )
    
    # Demo orders
    orders = [
        Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100,
            strategy_id="demo_strategy_1"
        ),
        Order(
            symbol="MSFT", 
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=50,
            limit_price=Decimal("380.00"),
            strategy_id="demo_strategy_2"
        ),
        Order(
            symbol="GOOGL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=75,
            strategy_id="demo_strategy_1"
        )
    ]
    
    # Submit orders
    order_ids = []
    for order in orders:
        print(f"\n📋 Submitting order: {order.symbol} {order.side.value} {order.quantity} @ {order.order_type.value}")
        order_id = order_manager.submit_order(order)
        order_ids.append(order_id)
        time.sleep(0.1)  # Brief delay between orders
    
    # Check order status
    print(f"\n📊 Order Execution Results:")
    print("-" * 50)
    
    for order_id in order_ids:
        order = order_manager.get_order(order_id)
        if order:
            print(f"Order {order_id[-8:]}:")
            print(f"  Status: {order.status.value}")
            print(f"  Fill Rate: {order.fill_rate:.1%}")
            if order.filled_quantity > 0:
                print(f"  Avg Fill Price: ${order.average_fill_price:.2f}")
                print(f"  Commission: ${order.total_commission:.2f}")
            print()
    
    # Show execution statistics
    stats = order_manager.get_execution_stats()
    print(f"📈 Execution Statistics:")
    print(f"  Total Orders: {stats['total_orders']}")
    print(f"  Total Fills: {stats['total_fills']}")
    print(f"  Average Fill Rate: {stats['avg_fill_rate']:.1%}")
    print(f"  Total Commission: ${stats['total_commission']:.2f}")


def demo_position_tracking():
    """Demonstrate position tracking with P&L calculation."""
    print("\n" + "="*60)
    print("📊 Position Tracking Demo")
    print("="*60)
    
    # Create position tracker
    tracker = PositionTracker(initial_cash=Decimal("100000"))
    
    # Simulate some trades
    from bot.execution.position_tracker import PositionUpdate
    
    trades = [
        # Buy AAPL
        PositionUpdate(
            symbol="AAPL",
            quantity=100,
            price=Decimal("175.00"),
            timestamp=datetime.now(),
            trade_id="trade_001",
            commission=Decimal("1.00")
        ),
        # Add to AAPL position
        PositionUpdate(
            symbol="AAPL", 
            quantity=50,
            price=Decimal("176.50"),
            timestamp=datetime.now() + timedelta(minutes=30),
            trade_id="trade_002",
            commission=Decimal("1.00")
        ),
        # Buy MSFT
        PositionUpdate(
            symbol="MSFT",
            quantity=50,
            price=Decimal("380.00"),
            timestamp=datetime.now() + timedelta(hours=1),
            trade_id="trade_003", 
            commission=Decimal("1.00")
        ),
        # Partial sell of AAPL
        PositionUpdate(
            symbol="AAPL",
            quantity=-75,  # Sell 75 shares
            price=Decimal("178.25"),
            timestamp=datetime.now() + timedelta(hours=2),
            trade_id="trade_004",
            commission=Decimal("1.00")
        )
    ]
    
    # Execute trades
    print("💰 Executing trades...")
    for trade in trades:
        action = "BUY" if trade.quantity > 0 else "SELL"
        print(f"  {action} {abs(trade.quantity)} {trade.symbol} @ ${trade.price:.2f}")
        tracker.update_position(trade)
    
    # Update with market prices for unrealized P&L
    market_prices = {
        "AAPL": Decimal("179.50"),
        "MSFT": Decimal("385.25"),
    }
    
    print(f"\n📈 Updating market prices...")
    for symbol, price in market_prices.items():
        print(f"  {symbol}: ${price:.2f}")
    
    tracker.update_market_prices(market_prices)
    
    # Show position summary
    summary = tracker.get_position_summary()
    print(f"\n📋 Position Summary:")
    print("-" * 40)
    print(f"Cash: ${summary['cash']:,.2f}")
    print(f"Portfolio Value: ${summary['portfolio_value']:,.2f}")
    print(f"Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"Realized P&L: ${summary['realized_pnl']:,.2f}")
    print(f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
    print(f"Portfolio Return: {summary['portfolio_return_pct']:+.2f}%")
    print(f"Win Rate: {summary['win_rate']:.1%}")
    
    print(f"\n📍 Active Positions:")
    for symbol, pos_data in summary['positions'].items():
        print(f"  {symbol}: {pos_data['quantity']} shares @ ${pos_data['average_price']:.2f}")
        print(f"    Market Value: ${pos_data['market_value']:,.2f}")
        print(f"    Unrealized P&L: ${pos_data['unrealized_pnl']:+,.2f}")


def demo_portfolio_state():
    """Demonstrate comprehensive portfolio state management."""
    print("\n" + "="*60)
    print("🎯 Portfolio State Management Demo")
    print("="*60)
    
    # Create portfolio state manager
    portfolio = PortfolioState(
        initial_cash=Decimal("500000"),
        snapshot_frequency=timedelta(hours=1)
    )
    
    # Simulate trading over several days
    print("🔄 Simulating multi-day trading...")
    
    base_time = datetime.now()
    market_prices = create_sample_market_data()
    
    # Day 1: Initial trades
    print("\n📅 Day 1: Opening positions")
    day1_trades = [
        ("AAPL", 200, Decimal("175.25"), "momentum_strategy"),
        ("MSFT", 100, Decimal("380.50"), "value_strategy"), 
        ("GOOGL", 150, Decimal("142.75"), "momentum_strategy"),
    ]
    
    for i, (symbol, qty, price, strategy) in enumerate(day1_trades):
        trade_time = base_time + timedelta(hours=i)
        portfolio.execute_trade(
            symbol=symbol,
            quantity=qty,
            price=price,
            timestamp=trade_time,
            strategy_id=strategy,
            commission=Decimal("1.00")
        )
        print(f"  {symbol}: BUY {qty} @ ${price:.2f}")
    
    # Update market prices
    portfolio.update_market_prices(market_prices, base_time + timedelta(hours=4))
    
    # Day 2: Some adjustments
    print("\n📅 Day 2: Position adjustments")
    day2_time = base_time + timedelta(days=1)
    
    # Market moved up
    updated_prices = {k: v * Decimal("1.02") for k, v in market_prices.items()}
    portfolio.update_market_prices(updated_prices, day2_time)
    
    # Take some profits
    portfolio.execute_trade(
        symbol="AAPL",
        quantity=-100,  # Sell half position
        price=updated_prices["AAPL"],
        timestamp=day2_time + timedelta(hours=2),
        strategy_id="momentum_strategy",
        commission=Decimal("1.00")
    )
    print(f"  AAPL: SELL 100 @ ${updated_prices['AAPL']:.2f} (taking profits)")
    
    # Day 3: Market correction
    print("\n📅 Day 3: Market correction")
    day3_time = base_time + timedelta(days=2)
    
    # Market dropped
    corrected_prices = {k: v * Decimal("0.95") for k, v in updated_prices.items()}
    portfolio.update_market_prices(corrected_prices, day3_time)
    
    # Force a snapshot
    snapshot = portfolio.take_snapshot(day3_time)
    
    # Show comprehensive results
    print(f"\n📊 Portfolio Performance Summary:")
    print("-" * 50)
    
    current_state = portfolio.get_current_state()
    print(f"Portfolio Value: ${current_state['portfolio_value']:,.2f}")
    print(f"Cash: ${current_state['cash']:,.2f}")
    print(f"Total P&L: ${current_state['total_pnl']:,.2f}")
    print(f"Portfolio Return: {current_state['portfolio_return']:+.2%}")
    print(f"Position Count: {current_state['position_count']}")
    print(f"Total Trades: {current_state['total_trades']}")
    print(f"Win Rate: {current_state['win_rate']:.1%}")
    
    # Show risk metrics
    risk_metrics = current_state['risk_metrics']
    print(f"\n⚠️  Risk Metrics:")
    print(f"Gross Exposure: ${current_state['gross_exposure']:,.2f}")
    print(f"Net Exposure: ${current_state['net_exposure']:,.2f}")
    print(f"Max Drawdown: {risk_metrics['max_drawdown']:+.2%}")
    print(f"Current Drawdown: {risk_metrics['current_drawdown']:+.2%}")
    
    # Performance summary
    perf_summary = portfolio.get_performance_summary()
    print(f"\n📈 Performance Analysis:")
    print(f"Annualized Return: {perf_summary['annualized_return']:+.2%}")
    print(f"Volatility: {perf_summary['volatility']:+.2%}")
    print(f"Sharpe Ratio: {perf_summary['sharpe_ratio']:+.2f}")
    print(f"Calmar Ratio: {perf_summary['calmar_ratio']:+.2f}")


def main():
    """Run all execution simulator demos."""
    print("🎮 GPT-Trader Execution Simulator Demo")
    print("======================================")
    print("Demonstrating realistic trade execution simulation for paper trading preparation")
    
    try:
        # Run all demos
        demo_order_execution()
        demo_position_tracking() 
        demo_portfolio_state()
        
        print("\n" + "="*60)
        print("✅ Execution Simulator Demo Complete!")
        print("="*60)
        print("The execution layer provides:")
        print("• Realistic order execution with slippage & partial fills")
        print("• Comprehensive position tracking with P&L calculation")
        print("• Portfolio state management with risk metrics")
        print("• Performance attribution and drawdown analysis")
        print("\nThis foundation enables realistic backtesting and")
        print("prepares the system for paper trading integration.")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()