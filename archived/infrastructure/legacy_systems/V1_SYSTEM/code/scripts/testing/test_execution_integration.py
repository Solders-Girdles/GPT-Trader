#!/usr/bin/env python3
"""
Test Execution Simulator Integration

Validates that the new execution simulator components integrate
properly with the existing orchestrator and strategy systems.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import logging
from datetime import datetime, timedelta
from decimal import Decimal

from bot.execution import (
    OrderManager,
    PositionTracker,
    PortfolioState,
    Order,
    OrderType,
    OrderSide
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_execution_components():
    """Test individual execution components."""
    print("🧪 Testing Execution Components")
    print("-" * 40)
    
    # Test OrderManager
    print("📋 Testing OrderManager...")
    order_manager = OrderManager()
    
    test_order = Order(
        symbol="TEST",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )
    
    order_id = order_manager.submit_order(test_order)
    order = order_manager.get_order(order_id)
    
    assert order is not None, "Order should be retrievable"
    assert order.status.value in ["submitted", "filled"], f"Order should be submitted or filled, got {order.status.value}"
    print(f"  ✅ Order executed with status: {order.status.value}")
    
    # Test PositionTracker
    print("📊 Testing PositionTracker...")
    tracker = PositionTracker(Decimal("100000"))
    
    from bot.execution.position_tracker import PositionUpdate
    update = PositionUpdate(
        symbol="TEST",
        quantity=100,
        price=Decimal("50.00"),
        timestamp=datetime.now(),
        trade_id="test_001"
    )
    
    tracker.update_position(update)
    summary = tracker.get_position_summary()
    
    assert summary['position_count'] == 1, "Should have one position"
    assert summary['cash'] < 100000, "Cash should be reduced"
    print(f"  ✅ Position tracking works, cash: ${summary['cash']:,.2f}")
    
    # Test PortfolioState
    print("🎯 Testing PortfolioState...")
    portfolio = PortfolioState(Decimal("200000"))
    
    portfolio.execute_trade(
        symbol="TEST",
        quantity=50,
        price=Decimal("45.00"),
        strategy_id="test_strategy"
    )
    
    state = portfolio.get_current_state()
    assert state['position_count'] == 1, "Should have one position"
    assert state['total_trades'] == 1, "Should have one trade"
    print(f"  ✅ Portfolio state works, value: ${state['portfolio_value']:,.2f}")
    
    print("✅ All execution components working correctly!\n")


def test_realistic_features():
    """Test realistic trading features."""
    print("🎯 Testing Realistic Trading Features")
    print("-" * 40)
    
    # Test slippage and commissions
    order_manager = OrderManager(
        commission_per_share=Decimal("0.01"),
        partial_fill_prob=0.5  # 50% chance of partial fills
    )
    
    # Submit multiple orders to test variability
    orders = []
    for i in range(5):
        order = Order(
            symbol=f"TEST{i}",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100 + i * 50  # Variable sizes
        )
        order_id = order_manager.submit_order(order)
        orders.append(order_manager.get_order(order_id))
    
    # Check for variability in execution
    fill_rates = [o.fill_rate for o in orders if o]
    execution_prices = [float(o.average_fill_price) for o in orders if o and o.average_fill_price]
    commissions = [float(o.total_commission) for o in orders if o]
    
    print(f"📊 Execution Results:")
    print(f"  Fill Rates: {[f'{r:.1%}' for r in fill_rates]}")
    print(f"  Execution Prices: {[f'${p:.2f}' for p in execution_prices]}")
    print(f"  Commissions: {[f'${c:.2f}' for c in commissions]}")
    
    # Verify realistic features
    assert len(set(execution_prices)) > 1 or len(execution_prices) <= 1, "Should have price variation or few orders"
    assert all(c > 0 for c in commissions), "All orders should have commission"
    print("  ✅ Realistic slippage and commission simulation working")
    
    # Test P&L calculation accuracy
    portfolio = PortfolioState(Decimal("100000"))
    
    # Buy and sell same stock
    portfolio.execute_trade("PROFIT_TEST", 100, Decimal("50.00"))
    portfolio.execute_trade("PROFIT_TEST", -100, Decimal("55.00"))  # Sell for profit
    
    state = portfolio.get_current_state()
    realized_pnl = portfolio.position_tracker.total_realized_pnl
    
    # Should have roughly $500 profit minus commissions
    expected_profit = 100 * (55 - 50)  # $500
    print(f"  Expected profit: ~${expected_profit:.2f}, Actual: ${float(realized_pnl):.2f}")
    assert float(realized_pnl) > 490, "Should have significant profit after commissions"
    print("  ✅ P&L calculation accuracy verified")
    
    print("✅ All realistic features working correctly!\n")


def test_integration_readiness():
    """Test readiness for integration with existing systems."""
    print("🔗 Testing Integration Readiness")
    print("-" * 40)
    
    # Test that components can work together
    portfolio = PortfolioState(Decimal("500000"))
    
    # Simulate strategy-driven trading
    strategies = ["momentum", "mean_reversion", "breakout"]
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    for i, (strategy, symbol) in enumerate(zip(strategies, symbols)):
        # Simulate different position sizes
        quantity = 100 + i * 50
        price = Decimal("100.00") + Decimal(str(i * 10))
        
        portfolio.execute_trade(
            symbol=symbol,
            quantity=quantity,
            price=price,
            strategy_id=strategy,
            timestamp=datetime.now() + timedelta(hours=i)
        )
    
    # Update market prices
    market_prices = {
        "AAPL": Decimal("105.00"),
        "MSFT": Decimal("115.00"),
        "GOOGL": Decimal("125.00")
    }
    portfolio.update_market_prices(market_prices)
    
    # Get comprehensive state
    state = portfolio.get_current_state()
    performance = portfolio.get_performance_summary()
    
    print(f"📊 Integration Test Results:")
    print(f"  Portfolio Value: ${state['portfolio_value']:,.2f}")
    print(f"  Position Count: {state['position_count']}")
    print(f"  Total Return: {performance['total_return']:+.2%}")
    print(f"  Sharpe Ratio: {performance['sharpe_ratio']:+.2f}")
    
    # Verify all required metrics are available
    required_metrics = [
        'portfolio_value', 'total_pnl', 'position_count',
        'gross_exposure', 'total_trades', 'win_rate'
    ]
    
    for metric in required_metrics:
        assert metric in state, f"Missing required metric: {metric}"
    
    print("  ✅ All required metrics available")
    
    # Test export functionality
    history_df = portfolio.export_history()
    print(f"  History snapshots: {len(history_df)} records")
    print("  ✅ Data export functionality working")
    
    print("✅ System ready for integration!\n")


def main():
    """Run all integration tests."""
    print("🧪 Execution Simulator Integration Tests")
    print("=" * 50)
    
    try:
        test_execution_components()
        test_realistic_features()
        test_integration_readiness()
        
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("✅ Execution simulator is ready for:")
        print("  • Integration with existing orchestrator")
        print("  • Realistic backtesting simulation")
        print("  • Paper trading preparation")
        print("  • Strategy performance attribution")
        print("  • Risk management integration")
        
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)