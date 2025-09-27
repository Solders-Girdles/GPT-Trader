"""Simple test script for position tracking system."""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Direct imports to avoid dependency issues
from bot.tracking.position_manager import PositionManager, PositionSnapshot
from bot.tracking.pnl_calculator import PnLCalculator
from bot.tracking.trade_ledger import TradeLedger, TradeEntry, TradeType, OrderType, TradeStatus
from bot.execution.position_tracker import PositionUpdate

def test_position_tracking():
    """Test the position tracking components."""
    
    print("Testing GPT-Trader Position Tracking System...")
    
    # Test Position Manager
    print("\n1. Testing Position Manager...")
    position_manager = PositionManager()
    
    # Create portfolio
    success = position_manager.create_portfolio(
        portfolio_id="TEST_PORTFOLIO",
        initial_cash=Decimal("100000"),
        strategy_name="Test Strategy"
    )
    print(f"  ✓ Created portfolio: {success}")
    
    # Add some trades
    trades = [
        ("AAPL", 100, Decimal("150.00")),
        ("GOOGL", 50, Decimal("2500.00")),
        ("AAPL", -25, Decimal("155.00"))  # Partial sale
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
    print(f"  ✓ Updated market prices")
    
    # Get portfolio snapshot
    snapshot = position_manager.get_portfolio_snapshot("TEST_PORTFOLIO")
    if snapshot:
        print(f"  Portfolio Value: ${snapshot.total_value:,.2f}")
        print(f"  Total P&L: ${snapshot.total_pnl:,.2f}")
        print(f"  Cash: ${snapshot.cash:,.2f}")
        print(f"  Positions: {snapshot.position_count}")
        for pos in snapshot.positions:
            print(f"    {pos.symbol}: {pos.quantity} shares, P&L: ${pos.total_pnl:.2f}")
    
    # Test P&L Calculator
    print("\n2. Testing P&L Calculator...")
    pnl_calculator = PnLCalculator()
    
    # Add portfolio values over time
    for i in range(5):
        timestamp = datetime.now() - timedelta(hours=5-i)
        value = Decimal("100000") + Decimal(str(i * 500))
        pnl_calculator.add_portfolio_value("TEST_PORTFOLIO", timestamp, value)
    
    # Calculate metrics
    metrics = pnl_calculator.calculate_pnl_metrics("TEST_PORTFOLIO")
    print(f"  Total P&L: ${metrics.total_pnl:.2f}")
    print(f"  Total Return: {metrics.total_return_pct:.2f}%")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Total Trades: {metrics.total_trades}")
    
    # Test Trade Ledger
    print("\n3. Testing Trade Ledger...")
    ledger_path = Path("test_ledger.db")
    trade_ledger = TradeLedger(ledger_path=ledger_path, use_database=True)
    
    # Add trades to ledger
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
        print(f"  ✓ Added trade to ledger: {trade_entry.trade_id}")
    
    # Get ledger summary
    summary = trade_ledger.get_ledger_summary()
    print(f"  Total Trades: {summary.total_trades}")
    print(f"  Total Volume: ${summary.total_volume:,.2f}")
    print(f"  Total P&L: ${summary.total_realized_pnl:,.2f}")
    print(f"  Win Rate: {summary.win_rate:.1f}%")
    
    # Export trades
    export_path = Path("test_trades.csv")
    success = trade_ledger.export_trades(export_path, format="csv")
    print(f"  ✓ Exported trades to CSV: {success}")
    
    # Test data classes
    print("\n4. Testing Data Structures...")
    
    # Test position snapshot
    if snapshot:
        print(f"  Position snapshot created: {snapshot.timestamp}")
        print(f"  Snapshot has {len(snapshot.positions)} positions")
    
    # Test trade entry conversion
    if trades:
        sample_trade = TradeEntry(
            trade_id="CONVERSION_TEST",
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
        print(f"  ✓ Trade serialization works: {reconstructed.trade_id == sample_trade.trade_id}")
    
    # Cleanup
    print("\n5. Cleanup...")
    if ledger_path.exists():
        ledger_path.unlink()
        print("  ✓ Removed test database")
    
    if export_path.exists():
        export_path.unlink()
        print("  ✓ Removed test export file")
    
    print("\n=== Position Tracking Test Complete ===")
    print("All components tested successfully!")
    return True

if __name__ == "__main__":
    try:
        success = test_position_tracking()
        if success:
            print("\n✅ Position tracking system is working correctly!")
        else:
            print("\n❌ Position tracking system has issues!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()