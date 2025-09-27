"""Direct test of tracking modules without importing bot package."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add src to path and import modules directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import tracking modules directly
try:
    from bot.tracking.position_manager import PositionManager, PositionSnapshot
    from bot.tracking.pnl_calculator import PnLCalculator
    from bot.tracking.trade_ledger import TradeLedger, TradeEntry, TradeType, OrderType, TradeStatus
    from bot.execution.position_tracker import PositionUpdate
    
    print("✓ Successfully imported tracking modules")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    # Try direct file imports
    try:
        # Import logging module first
        sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "bot"))
        
        import logging
        
        # Create logger function
        def get_logger(name):
            return logging.getLogger(name)
        
        # Mock the logging import
        import sys
        logging_module = type(sys)('logging')
        logging_module.get_logger = get_logger
        sys.modules['bot.logging'] = logging_module
        
        # Import specific modules
        from tracking.position_manager import PositionManager, PositionSnapshot
        from tracking.pnl_calculator import PnLCalculator
        from tracking.trade_ledger import TradeLedger, TradeEntry, TradeType, OrderType, TradeStatus
        from execution.position_tracker import PositionUpdate
        
        print("✓ Successfully imported tracking modules (direct)")
        
    except ImportError as e2:
        print(f"❌ Direct import also failed: {e2}")
        sys.exit(1)

def test_tracking_system():
    """Test the tracking system components."""
    
    print("\n=== Testing GPT-Trader Position Tracking System ===")
    
    # Test 1: Position Manager
    print("\n1. Testing Position Manager...")
    try:
        position_manager = PositionManager()
        
        # Create portfolio
        success = position_manager.create_portfolio(
            portfolio_id="TEST_PORTFOLIO",
            initial_cash=Decimal("100000"),
            strategy_name="Test Strategy"
        )
        print(f"  ✓ Created portfolio: {success}")
        
        # Add trades
        trades = [
            ("AAPL", 100, Decimal("150.00")),
            ("GOOGL", 50, Decimal("2500.00")),
            ("AAPL", -25, Decimal("155.00"))
        ]
        
        for i, (symbol, quantity, price) in enumerate(trades):
            update = PositionUpdate(
                symbol=symbol,
                quantity=quantity,
                price=price,
                timestamp=datetime.now() - timedelta(minutes=len(trades) - i),
                trade_id=f"TEST_{i+1}",
                commission=Decimal("2.50")
            )
            
            success = position_manager.add_trade("TEST_PORTFOLIO", update)
            action = "BUY" if quantity > 0 else "SELL"
            print(f"  ✓ {action} {abs(quantity)} {symbol} @ ${price}")
        
        # Update market prices
        market_prices = {
            "AAPL": Decimal("152.50"),
            "GOOGL": Decimal("2520.00")
        }
        position_manager.update_market_prices(market_prices)
        
        # Get snapshot
        snapshot = position_manager.get_portfolio_snapshot("TEST_PORTFOLIO")
        if snapshot:
            print(f"  ✓ Portfolio Value: ${snapshot.total_value:,.2f}")
            print(f"  ✓ Total P&L: ${snapshot.total_pnl:,.2f}")
            print(f"  ✓ Positions: {snapshot.position_count}")
        
        print("  ✓ Position Manager test passed")
        
    except Exception as e:
        print(f"  ❌ Position Manager test failed: {e}")
        return False
    
    # Test 2: P&L Calculator
    print("\n2. Testing P&L Calculator...")
    try:
        pnl_calculator = PnLCalculator()
        
        # Add some portfolio values
        for i in range(5):
            timestamp = datetime.now() - timedelta(hours=5-i)
            value = Decimal("100000") + Decimal(str(i * 500))
            pnl_calculator.add_portfolio_value("TEST_PORTFOLIO", timestamp, value)
        
        # Calculate metrics
        metrics = pnl_calculator.calculate_pnl_metrics("TEST_PORTFOLIO")
        print(f"  ✓ Total P&L: ${metrics.total_pnl:.2f}")
        print(f"  ✓ Total Return: {metrics.total_return_pct:.2f}%")
        print(f"  ✓ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        
        print("  ✓ P&L Calculator test passed")
        
    except Exception as e:
        print(f"  ❌ P&L Calculator test failed: {e}")
        return False
    
    # Test 3: Trade Ledger
    print("\n3. Testing Trade Ledger...")
    try:
        ledger_path = Path("test_ledger.db")
        trade_ledger = TradeLedger(ledger_path=ledger_path, use_database=True)
        
        # Add trades
        for i, (symbol, quantity, price) in enumerate(trades):
            trade_entry = TradeEntry(
                trade_id=f"LEDGER_TEST_{i+1}",
                portfolio_id="TEST_PORTFOLIO",
                strategy_name="Test Strategy",
                symbol=symbol,
                trade_type=TradeType.BUY if quantity > 0 else TradeType.SELL,
                quantity=abs(quantity),
                price=price,
                timestamp=datetime.now() - timedelta(minutes=len(trades) - i),
                order_type=OrderType.MARKET,
                status=TradeStatus.FILLED,
                filled_quantity=abs(quantity),
                avg_fill_price=price,
                commission=Decimal("2.50"),
                realized_pnl=Decimal("125.00") if quantity < 0 else Decimal("0"),
                source="test"
            )
            
            success = trade_ledger.add_trade(trade_entry)
            print(f"  ✓ Added trade: {trade_entry.trade_id}")
        
        # Get summary
        summary = trade_ledger.get_ledger_summary()
        print(f"  ✓ Total Trades: {summary.total_trades}")
        print(f"  ✓ Total Volume: ${summary.total_volume:,.2f}")
        print(f"  ✓ Win Rate: {summary.win_rate:.1f}%")
        
        # Export trades
        export_path = Path("test_trades.csv")
        success = trade_ledger.export_trades(export_path, format="csv")
        print(f"  ✓ Exported trades: {success}")
        
        print("  ✓ Trade Ledger test passed")
        
        # Cleanup
        if ledger_path.exists():
            ledger_path.unlink()
        if export_path.exists():
            export_path.unlink()
        
    except Exception as e:
        print(f"  ❌ Trade Ledger test failed: {e}")
        return False
    
    # Test 4: Data Structures
    print("\n4. Testing Data Structures...")
    try:
        # Test trade entry serialization
        sample_trade = TradeEntry(
            trade_id="SERIALIZATION_TEST",
            portfolio_id="TEST_PORTFOLIO",
            strategy_name="Test Strategy",
            symbol="AAPL",
            trade_type=TradeType.BUY,
            quantity=100,
            price=Decimal("150.00"),
            timestamp=datetime.now()
        )
        
        # Convert to dict and back
        trade_dict = sample_trade.to_dict()
        reconstructed = TradeEntry.from_dict(trade_dict)
        
        if reconstructed.trade_id == sample_trade.trade_id:
            print("  ✓ Trade serialization works")
        else:
            print("  ❌ Trade serialization failed")
            return False
        
        print("  ✓ Data structures test passed")
        
    except Exception as e:
        print(f"  ❌ Data structures test failed: {e}")
        return False
    
    print("\n=== All Tests Passed! ===")
    print("Position tracking system is working correctly!")
    return True

if __name__ == "__main__":
    success = test_tracking_system()
    if success:
        print("\n🎉 Position tracking system test completed successfully!")
    else:
        print("\n💥 Position tracking system test failed!")
        sys.exit(1)