"""Demo script for the real-time position tracking system.

This script demonstrates the comprehensive position tracking system with:
- Real-time position management
- P&L calculations and performance metrics
- Trade ledger and audit trail
- Position reconciliation with broker
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bot.tracking import (
    PositionManager, PnLCalculator, TradeLedger, Reconciliation,
    TradeEntry, TradeType, OrderType, TradeStatus
)
from bot.execution.position_tracker import PositionUpdate
from bot.brokers.alpaca.alpaca_client import AlpacaClient
from bot.brokers.alpaca.alpaca_data import AlpacaDataFeed, MarketDataConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("position_tracking_demo")


async def demonstrate_position_tracking():
    """Demonstrate the complete position tracking system."""
    
    logger.info("=== GPT-Trader Position Tracking System Demo ===")
    
    # Initialize components
    logger.info("\n1. Initializing Position Tracking Components...")
    
    # Position Manager
    position_manager = PositionManager(
        data_feed=None,  # Would normally connect to Alpaca data feed
        update_interval=1.0,
        snapshot_interval=60.0
    )
    
    # P&L Calculator
    pnl_calculator = PnLCalculator(risk_free_rate=0.02)
    
    # Trade Ledger
    ledger_path = Path("demo_trade_ledger.db")
    trade_ledger = TradeLedger(
        ledger_path=ledger_path,
        use_database=True,
        max_memory_trades=1000
    )
    
    # Create demo portfolios
    logger.info("\n2. Creating Demo Portfolios...")
    
    portfolios = ["PORTFOLIO_1", "PORTFOLIO_2", "STRATEGY_A"]
    for portfolio_id in portfolios:
        position_manager.create_portfolio(
            portfolio_id=portfolio_id,
            initial_cash=Decimal("100000"),
            strategy_name=f"Demo Strategy {portfolio_id}"
        )
        logger.info(f"Created portfolio: {portfolio_id}")
    
    # Simulate some trades
    logger.info("\n3. Simulating Trading Activity...")
    
    demo_trades = [
        # Portfolio 1 trades
        {
            'portfolio_id': 'PORTFOLIO_1',
            'symbol': 'AAPL',
            'trade_type': TradeType.BUY,
            'quantity': 100,
            'price': Decimal('150.00'),
            'strategy': 'Demo Strategy PORTFOLIO_1'
        },
        {
            'portfolio_id': 'PORTFOLIO_1',
            'symbol': 'GOOGL',
            'trade_type': TradeType.BUY,
            'quantity': 50,
            'price': Decimal('2500.00'),
            'strategy': 'Demo Strategy PORTFOLIO_1'
        },
        {
            'portfolio_id': 'PORTFOLIO_1',
            'symbol': 'AAPL',
            'trade_type': TradeType.SELL,
            'quantity': 25,
            'price': Decimal('155.00'),
            'strategy': 'Demo Strategy PORTFOLIO_1'
        },
        
        # Portfolio 2 trades
        {
            'portfolio_id': 'PORTFOLIO_2',
            'symbol': 'TSLA',
            'trade_type': TradeType.BUY,
            'quantity': 75,
            'price': Decimal('800.00'),
            'strategy': 'Demo Strategy PORTFOLIO_2'
        },
        {
            'portfolio_id': 'PORTFOLIO_2',
            'symbol': 'MSFT',
            'trade_type': TradeType.BUY,
            'quantity': 150,
            'price': Decimal('300.00'),
            'strategy': 'Demo Strategy PORTFOLIO_2'
        },
        
        # Strategy A trades
        {
            'portfolio_id': 'STRATEGY_A',
            'symbol': 'SPY',
            'trade_type': TradeType.BUY,
            'quantity': 200,
            'price': Decimal('420.00'),
            'strategy': 'Demo Strategy STRATEGY_A'
        }
    ]
    
    trade_counter = 1
    for trade_data in demo_trades:
        # Create position update
        timestamp = datetime.now() - timedelta(minutes=len(demo_trades) - trade_counter + 1)
        
        position_update = PositionUpdate(
            symbol=trade_data['symbol'],
            quantity=trade_data['quantity'] if trade_data['trade_type'] == TradeType.BUY else -trade_data['quantity'],
            price=trade_data['price'],
            timestamp=timestamp,
            trade_id=f"TRADE_{trade_counter:03d}",
            commission=Decimal("2.50")
        )
        
        # Add to position manager
        success = position_manager.add_trade(trade_data['portfolio_id'], position_update)
        
        if success:
            logger.info(f"  ✓ {trade_data['trade_type'].value.upper()} {trade_data['quantity']} {trade_data['symbol']} @ ${trade_data['price']}")
        
        # Create trade ledger entry
        trade_entry = TradeEntry(
            trade_id=f"TRADE_{trade_counter:03d}",
            portfolio_id=trade_data['portfolio_id'],
            strategy_name=trade_data['strategy'],
            symbol=trade_data['symbol'],
            trade_type=trade_data['trade_type'],
            quantity=trade_data['quantity'],
            price=trade_data['price'],
            timestamp=timestamp,
            order_type=OrderType.MARKET,
            status=TradeStatus.FILLED,
            filled_quantity=trade_data['quantity'],
            avg_fill_price=trade_data['price'],
            commission=Decimal("2.50"),
            realized_pnl=Decimal("125.00") if trade_data['trade_type'] == TradeType.SELL else Decimal("0"),
            source="demo"
        )
        
        # Add to trade ledger
        trade_ledger.add_trade(trade_entry)
        
        # Add data to P&L calculator
        portfolio_value = Decimal("100000") + Decimal(str(trade_counter * 1000))  # Simulate growth
        pnl_calculator.add_portfolio_value(trade_data['portfolio_id'], timestamp, portfolio_value)
        
        trade_counter += 1
        
        # Small delay to simulate realistic timing
        await asyncio.sleep(0.1)
    
    # Simulate market price updates
    logger.info("\n4. Simulating Market Price Updates...")
    
    market_prices = {
        'AAPL': Decimal('152.50'),
        'GOOGL': Decimal('2520.00'),
        'TSLA': Decimal('785.00'),
        'MSFT': Decimal('310.00'),
        'SPY': Decimal('425.00')
    }
    
    position_manager.update_market_prices(market_prices)
    logger.info(f"Updated market prices for {len(market_prices)} symbols")
    
    # Generate portfolio snapshots
    logger.info("\n5. Generating Portfolio Snapshots...")
    
    all_snapshots = position_manager.get_all_portfolio_snapshots()
    
    for portfolio_id, snapshot in all_snapshots.items():
        logger.info(f"\n--- Portfolio: {portfolio_id} ---")
        logger.info(f"  Total Value: ${snapshot.total_value:,.2f}")
        logger.info(f"  Cash: ${snapshot.cash:,.2f}")
        logger.info(f"  Total P&L: ${snapshot.total_pnl:,.2f}")
        logger.info(f"  Total Return: {snapshot.total_return_pct:.2f}%")
        logger.info(f"  Positions: {snapshot.position_count}")
        logger.info(f"  Gross Exposure: ${snapshot.gross_exposure:,.2f}")
        logger.info(f"  Win Rate: {snapshot.win_rate:.1f}%")
        
        # Show individual positions
        if snapshot.positions:
            logger.info("  Individual Positions:")
            for pos in snapshot.positions:
                logger.info(f"    {pos.symbol}: {pos.quantity} shares @ ${pos.average_price:.2f} "
                          f"(P&L: ${pos.total_pnl:.2f})")
    
    # Calculate comprehensive P&L metrics
    logger.info("\n6. Calculating P&L Metrics...")
    
    for portfolio_id in portfolios:
        logger.info(f"\n--- P&L Analysis: {portfolio_id} ---")
        
        # Calculate metrics
        metrics = pnl_calculator.calculate_pnl_metrics(
            identifier=portfolio_id,
            start_date=datetime.now() - timedelta(hours=1)
        )
        
        logger.info(f"  Total P&L: ${metrics.total_pnl:,.2f}")
        logger.info(f"  Realized P&L: ${metrics.realized_pnl:,.2f}")
        logger.info(f"  Unrealized P&L: ${metrics.unrealized_pnl:,.2f}")
        logger.info(f"  Total Return: {metrics.total_return_pct:.2f}%")
        logger.info(f"  Annualized Return: {metrics.annualized_return_pct:.2f}%")
        logger.info(f"  Volatility: {metrics.volatility_pct:.2f}%")
        logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        logger.info(f"  Total Trades: {metrics.total_trades}")
        logger.info(f"  Win Rate: {metrics.win_rate_pct:.1f}%")
        
        # Calculate performance metrics
        perf_metrics = pnl_calculator.calculate_performance_metrics(
            identifier=portfolio_id,
            start_date=datetime.now() - timedelta(hours=1)
        )
        
        logger.info(f"  vs Benchmark: {perf_metrics.excess_return_pct:.2f}% excess return")
        logger.info(f"  Beta: {perf_metrics.beta:.2f}")
        logger.info(f"  Tracking Error: {perf_metrics.tracking_error_pct:.2f}%")
    
    # Generate trade ledger summary
    logger.info("\n7. Trade Ledger Analysis...")
    
    ledger_summary = trade_ledger.get_ledger_summary()
    
    logger.info(f"\n--- Trade Ledger Summary ---")
    logger.info(f"  Total Trades: {ledger_summary.total_trades}")
    logger.info(f"  Total Volume: ${ledger_summary.total_volume:,.2f}")
    logger.info(f"  Total Commission: ${ledger_summary.total_commission:,.2f}")
    logger.info(f"  Total Realized P&L: ${ledger_summary.total_realized_pnl:,.2f}")
    logger.info(f"  Win Rate: {ledger_summary.win_rate:.1f}%")
    logger.info(f"  Average Trade Size: ${ledger_summary.avg_trade_size:,.2f}")
    logger.info(f"  Largest Win: ${ledger_summary.largest_win:,.2f}")
    logger.info(f"  Largest Loss: ${ledger_summary.largest_loss:,.2f}")
    logger.info(f"  Fill Rate: {ledger_summary.fill_rate:.1f}%")
    
    # Show strategy performance breakdown
    if ledger_summary.strategy_performance:
        logger.info("\n  Performance by Strategy:")
        for strategy, stats in ledger_summary.strategy_performance.items():
            logger.info(f"    {strategy}: {stats['trades']} trades, "
                      f"${stats['total_pnl']} P&L, {stats['win_rate']:.1f}% win rate")
    
    # Show symbol performance breakdown
    if ledger_summary.symbol_performance:
        logger.info("\n  Performance by Symbol:")
        for symbol, stats in ledger_summary.symbol_performance.items():
            logger.info(f"    {symbol}: {stats['trades']} trades, "
                      f"${stats['total_pnl']} P&L, {stats['win_rate']:.1f}% win rate")
    
    # Export trade data
    logger.info("\n8. Exporting Trade Data...")
    
    export_path = Path("demo_trades_export.csv")
    success = trade_ledger.export_trades(
        file_path=export_path,
        format="csv"
    )
    
    if success:
        logger.info(f"  ✓ Exported trades to {export_path}")
    
    # Demonstrate rolling metrics
    logger.info("\n9. Calculating Rolling Metrics...")
    
    for portfolio_id in portfolios:
        rolling_returns = pnl_calculator.calculate_rolling_metrics(
            identifier=portfolio_id,
            window_days=7,  # 7-day rolling window
            metric_type="return"
        )
        
        if rolling_returns.returns:
            logger.info(f"\n--- Rolling Returns: {portfolio_id} ---")
            logger.info(f"  Average 7-day Return: {rolling_returns.avg_return:.2f}%")
            logger.info(f"  Min 7-day Return: {rolling_returns.min_return:.2f}%")
            logger.info(f"  Max 7-day Return: {rolling_returns.max_return:.2f}%")
            logger.info(f"  Return Skewness: {rolling_returns.return_skewness:.2f}")
    
    # Demonstrate position history
    logger.info("\n10. Position History Analysis...")
    
    for portfolio_id in portfolios:
        snapshot = position_manager.get_portfolio_snapshot(portfolio_id)
        if snapshot and snapshot.positions:
            for pos in snapshot.positions[:2]:  # Show first 2 positions
                position_history = position_manager.get_position_history(
                    portfolio_id=portfolio_id,
                    symbol=pos.symbol
                )
                
                if position_history:
                    logger.info(f"\n--- Position History: {portfolio_id} {pos.symbol} ---")
                    logger.info(f"  Number of snapshots: {len(position_history)}")
                    logger.info(f"  First snapshot: {position_history[0].timestamp}")
                    logger.info(f"  Latest P&L: ${position_history[-1].total_pnl:.2f}")
                    logger.info(f"  Position age: {position_history[-1].position_age_seconds/3600:.2f} hours")
    
    # Clean up
    logger.info("\n11. Cleanup...")
    
    # Remove demo database
    if ledger_path.exists():
        ledger_path.unlink()
        logger.info("  ✓ Cleaned up demo database")
    
    if export_path.exists():
        export_path.unlink()
        logger.info("  ✓ Cleaned up export file")
    
    logger.info("\n=== Demo Complete ===")
    logger.info("The position tracking system provides:")
    logger.info("  ✓ Real-time position management with WebSocket integration")
    logger.info("  ✓ Comprehensive P&L calculations and performance metrics")
    logger.info("  ✓ Complete trade ledger with audit trail")
    logger.info("  ✓ Position reconciliation with broker")
    logger.info("  ✓ Multi-portfolio/strategy support")
    logger.info("  ✓ Historical analysis and rolling metrics")
    logger.info("  ✓ Export/import capabilities")
    logger.info("  ✓ Real-time event handling and notifications")


async def demonstrate_reconciliation():
    """Demonstrate the reconciliation system (simplified without real Alpaca connection)."""
    
    logger.info("\n=== Position Reconciliation Demo ===")
    
    # This would normally use real Alpaca client
    # For demo purposes, we'll show the reconciliation structure
    
    position_manager = PositionManager()
    trade_ledger = TradeLedger(ledger_path=Path("demo_reconciliation.db"))
    
    # Create a mock Alpaca client (normally would be real)
    class MockAlpacaClient:
        def get_positions(self):
            return [
                {
                    'symbol': 'AAPL',
                    'qty': 100,
                    'avg_cost': 150.25,
                    'market_value': 15200.00
                },
                {
                    'symbol': 'GOOGL', 
                    'qty': 50,
                    'avg_cost': 2501.00,
                    'market_value': 126000.00
                }
            ]
        
        def get_account(self):
            return {
                'cash': 25000.00,
                'portfolio_value': 166200.00
            }
        
        def get_orders(self, status=None, limit=None):
            return [
                {
                    'id': 'order_1',
                    'symbol': 'AAPL',
                    'qty': 100,
                    'side': 'buy',
                    'created_at': datetime.now().isoformat()
                }
            ]
    
    mock_client = MockAlpacaClient()
    
    # Create reconciliation engine
    reconciliation = Reconciliation(
        position_manager=position_manager,
        trade_ledger=trade_ledger,
        alpaca_client=mock_client,
        reconciliation_interval_minutes=15,
        auto_correct_minor_discrepancies=True
    )
    
    logger.info("  ✓ Reconciliation engine initialized")
    logger.info("  ✓ Would compare internal positions with broker positions")
    logger.info("  ✓ Would detect discrepancies and generate reports")
    logger.info("  ✓ Would auto-correct minor issues")
    logger.info("  ✓ Would flag major discrepancies for manual review")
    
    # Clean up
    Path("demo_reconciliation.db").unlink(missing_ok=True)


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(demonstrate_position_tracking())
    
    # Run reconciliation demo
    asyncio.run(demonstrate_reconciliation())