"""Real-time Position Tracking Integration Example.

This example shows how to integrate the position tracking system with:
- Alpaca paper trading API
- Real-time market data WebSocket
- Existing strategy execution
- Live risk monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sys
from typing import Dict, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from bot.tracking import (
    PositionManager, PnLCalculator, TradeLedger, Reconciliation,
    TradeEntry, TradeType, OrderType, TradeStatus
)
from bot.execution.position_tracker import PositionUpdate
from bot.brokers.alpaca.alpaca_client import AlpacaClient
from bot.brokers.alpaca.alpaca_data import AlpacaDataFeed, MarketDataConfig
from bot.brokers.alpaca.alpaca_executor import AlpacaExecutor
from bot.risk.simple_risk_manager import SimpleRiskManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("realtime_tracking_integration")


class RealTimeTrackingSystem:
    """Integrated real-time position tracking system."""
    
    def __init__(
        self,
        alpaca_api_key: str = None,
        alpaca_secret_key: str = None,
        is_paper: bool = True
    ):
        """Initialize the integrated tracking system."""
        
        # Core components
        self.alpaca_client = None
        self.alpaca_executor = None
        self.data_feed = None
        
        if alpaca_api_key and alpaca_secret_key:
            # Initialize Alpaca components
            self.alpaca_client = AlpacaClient(
                api_key=alpaca_api_key,
                secret_key=alpaca_secret_key,
                paper=is_paper
            )
            
            self.alpaca_executor = AlpacaExecutor(self.alpaca_client)
            
            # Setup data feed
            data_config = MarketDataConfig(
                symbols=['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'SPY'],
                subscribe_quotes=True,
                subscribe_trades=True,
                max_buffer_size=5000
            )
            
            self.data_feed = AlpacaDataFeed(self.alpaca_client, data_config)
        
        # Tracking components
        self.position_manager = PositionManager(
            data_feed=self.data_feed,
            update_interval=1.0,
            snapshot_interval=30.0
        )
        
        self.pnl_calculator = PnLCalculator(risk_free_rate=0.02)
        
        self.trade_ledger = TradeLedger(
            ledger_path=Path("realtime_tracking.db"),
            use_database=True,
            max_memory_trades=10000
        )
        
        # Risk manager integration
        self.risk_manager = SimpleRiskManager(
            max_position_size=10000,
            max_portfolio_value=500000
        )
        
        # Reconciliation (if Alpaca client available)
        self.reconciliation = None
        if self.alpaca_client:
            self.reconciliation = Reconciliation(
                position_manager=self.position_manager,
                trade_ledger=self.trade_ledger,
                alpaca_client=self.alpaca_client,
                reconciliation_interval_minutes=15,
                auto_correct_minor_discrepancies=True
            )
        
        # State tracking
        self.is_running = False
        self.active_portfolios: Dict[str, Dict] = {}
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event handlers for real-time updates."""
        
        # Position update handler
        def on_position_update(portfolio_id: str, snapshot):
            logger.info(f"Position updated: {portfolio_id} - "
                       f"Value: ${snapshot.total_value:,.2f}, "
                       f"P&L: ${snapshot.total_pnl:,.2f}")
            
            # Add to P&L calculator
            self.pnl_calculator.add_portfolio_value(
                portfolio_id, snapshot.timestamp, snapshot.total_value
            )
            
            # Check risk limits
            self._check_risk_limits(portfolio_id, snapshot)
        
        # Portfolio update handler
        def on_portfolio_update(portfolio_id: str, snapshot):
            logger.debug(f"Portfolio snapshot: {portfolio_id} - "
                        f"{snapshot.position_count} positions")
            
            # Store latest snapshot
            self.active_portfolios[portfolio_id] = {
                'last_snapshot': snapshot,
                'last_update': datetime.now()
            }
        
        self.position_manager.add_position_update_handler(on_position_update)
        self.position_manager.add_portfolio_update_handler(on_portfolio_update)
    
    async def start(self):
        """Start the real-time tracking system."""
        if self.is_running:
            logger.warning("Tracking system already running")
            return
        
        logger.info("Starting real-time position tracking system...")
        
        # Start position manager
        await self.position_manager.start()
        
        # Start data feed if available
        if self.data_feed:
            await self.data_feed.connect()
            logger.info("Connected to real-time data feed")
        
        # Start reconciliation if available
        if self.reconciliation:
            await self.reconciliation.start_automatic_reconciliation()
            logger.info("Started automatic reconciliation")
        
        self.is_running = True
        logger.info("✓ Real-time tracking system started")
    
    async def stop(self):
        """Stop the real-time tracking system."""
        if not self.is_running:
            return
        
        logger.info("Stopping real-time tracking system...")
        
        # Stop components
        await self.position_manager.stop()
        
        if self.data_feed:
            await self.data_feed.disconnect()
        
        if self.reconciliation:
            await self.reconciliation.stop_automatic_reconciliation()
        
        self.is_running = False
        logger.info("✓ Real-time tracking system stopped")
    
    def create_portfolio(
        self,
        portfolio_id: str,
        strategy_name: str,
        initial_cash: Decimal = Decimal("100000")
    ) -> bool:
        """Create a new portfolio for tracking."""
        
        success = self.position_manager.create_portfolio(
            portfolio_id=portfolio_id,
            initial_cash=initial_cash,
            strategy_name=strategy_name
        )
        
        if success:
            self.active_portfolios[portfolio_id] = {
                'strategy_name': strategy_name,
                'created_at': datetime.now(),
                'initial_cash': initial_cash
            }
            logger.info(f"Created portfolio: {portfolio_id} with ${initial_cash:,.2f}")
        
        return success
    
    async def execute_trade(
        self,
        portfolio_id: str,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: int,
        order_type: str = "market",
        limit_price: float = None
    ) -> Dict:
        """Execute a trade and update tracking."""
        
        if portfolio_id not in self.active_portfolios:
            return {'success': False, 'error': 'Portfolio not found'}
        
        try:
            # Check risk limits first
            current_snapshot = self.position_manager.get_portfolio_snapshot(portfolio_id)
            if current_snapshot:
                risk_check = self._check_pre_trade_risk(
                    portfolio_id, symbol, side, quantity, current_snapshot
                )
                if not risk_check['allowed']:
                    return {'success': False, 'error': f"Risk check failed: {risk_check['reason']}"}
            
            # Execute trade via Alpaca (if available)
            execution_result = None
            if self.alpaca_executor:
                if order_type == "market":
                    execution_result = self.alpaca_executor.submit_market_order(
                        symbol=symbol,
                        side=side,
                        qty=quantity
                    )
                elif order_type == "limit" and limit_price:
                    execution_result = self.alpaca_executor.submit_limit_order(
                        symbol=symbol,
                        side=side,
                        qty=quantity,
                        limit_price=limit_price
                    )
            
            # Generate trade ID
            trade_id = f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
            
            # Simulate execution if no real broker
            if not execution_result:
                execution_result = self._simulate_execution(symbol, side, quantity, order_type)
            
            if execution_result.success:
                # Create position update
                signed_quantity = quantity if side.lower() == 'buy' else -quantity
                price = Decimal(str(limit_price if limit_price else execution_result.order_data.get('price', 100.00)))
                
                position_update = PositionUpdate(
                    symbol=symbol,
                    quantity=signed_quantity,
                    price=price,
                    timestamp=datetime.now(),
                    trade_id=trade_id,
                    commission=Decimal("2.50")
                )
                
                # Update position manager
                self.position_manager.add_trade(portfolio_id, position_update)
                
                # Create trade ledger entry
                trade_entry = TradeEntry(
                    trade_id=trade_id,
                    portfolio_id=portfolio_id,
                    strategy_name=self.active_portfolios[portfolio_id].get('strategy_name', 'Unknown'),
                    symbol=symbol,
                    trade_type=TradeType.BUY if side.lower() == 'buy' else TradeType.SELL,
                    quantity=quantity,
                    price=price,
                    timestamp=datetime.now(),
                    order_id=execution_result.order_id,
                    order_type=OrderType.MARKET if order_type == "market" else OrderType.LIMIT,
                    status=TradeStatus.FILLED,
                    filled_quantity=quantity,
                    avg_fill_price=price,
                    commission=Decimal("2.50"),
                    source="realtime_system"
                )
                
                # Add to trade ledger
                self.trade_ledger.add_trade(trade_entry)
                
                logger.info(f"✓ Trade executed: {side.upper()} {quantity} {symbol} @ ${price}")
                
                return {
                    'success': True,
                    'trade_id': trade_id,
                    'order_id': execution_result.order_id,
                    'execution_price': float(price),
                    'commission': float(Decimal("2.50"))
                }
            else:
                return {
                    'success': False,
                    'error': execution_result.error or 'Trade execution failed'
                }
        
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_portfolio_status(self, portfolio_id: str) -> Dict:
        """Get comprehensive portfolio status."""
        
        if portfolio_id not in self.active_portfolios:
            return {'error': 'Portfolio not found'}
        
        # Get current snapshot
        snapshot = self.position_manager.get_portfolio_snapshot(portfolio_id)
        if not snapshot:
            return {'error': 'No snapshot available'}
        
        # Get P&L metrics
        metrics = self.pnl_calculator.calculate_pnl_metrics(
            identifier=portfolio_id,
            start_date=datetime.now() - timedelta(days=30)
        )
        
        # Get recent trades
        recent_trades = self.trade_ledger.get_trades(
            portfolio_id=portfolio_id,
            start_date=datetime.now() - timedelta(hours=24),
            limit=10
        )
        
        return {
            'portfolio_id': portfolio_id,
            'timestamp': snapshot.timestamp,
            'status': snapshot.status.value,
            'total_value': float(snapshot.total_value),
            'cash': float(snapshot.cash),
            'total_pnl': float(snapshot.total_pnl),
            'realized_pnl': float(snapshot.realized_pnl),
            'unrealized_pnl': float(snapshot.unrealized_pnl),
            'total_return_pct': snapshot.total_return_pct,
            'daily_return_pct': snapshot.daily_return_pct,
            'sharpe_ratio': snapshot.sharpe_ratio,
            'max_drawdown_pct': snapshot.max_drawdown_pct,
            'position_count': snapshot.position_count,
            'gross_exposure': float(snapshot.gross_exposure),
            'net_exposure': float(snapshot.net_exposure),
            'win_rate': snapshot.win_rate,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'market_value': float(pos.market_value),
                    'unrealized_pnl': float(pos.unrealized_pnl),
                    'total_pnl': float(pos.total_pnl),
                    'average_price': float(pos.average_price)
                }
                for pos in snapshot.positions
            ],
            'metrics': {
                'volatility_pct': metrics.volatility_pct,
                'var_95_pct': metrics.var_95_pct,
                'calmar_ratio': metrics.calmar_ratio,
                'total_trades': metrics.total_trades,
                'winning_trades': metrics.winning_trades,
                'losing_trades': metrics.losing_trades
            },
            'recent_trades': len(recent_trades)
        }
    
    async def run_reconciliation(self, portfolio_id: str = None) -> Dict:
        """Run position reconciliation."""
        
        if not self.reconciliation:
            return {'error': 'Reconciliation not available (no Alpaca client)'}
        
        try:
            if portfolio_id:
                report = await self.reconciliation.reconcile_portfolio(portfolio_id, force=True)
                return {
                    'portfolio_id': portfolio_id,
                    'status': report.status.value,
                    'discrepancies_found': report.discrepancies_found,
                    'is_reconciled': report.is_reconciled,
                    'value_difference': float(report.value_difference),
                    'auto_corrections': report.auto_corrections_applied,
                    'critical_issues': report.has_critical_issues
                }
            else:
                reports = await self.reconciliation.reconcile_all_portfolios(force=True)
                return {
                    'portfolios_reconciled': len(reports),
                    'total_discrepancies': sum(r.discrepancies_found for r in reports.values()),
                    'critical_issues': sum(1 for r in reports.values() if r.has_critical_issues)
                }
        
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            return {'error': str(e)}
    
    def get_performance_report(self, portfolio_id: str, days: int = 30) -> Dict:
        """Generate comprehensive performance report."""
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Get P&L metrics
        metrics = self.pnl_calculator.calculate_pnl_metrics(
            identifier=portfolio_id,
            start_date=start_date
        )
        
        # Get performance metrics
        perf_metrics = self.pnl_calculator.calculate_performance_metrics(
            identifier=portfolio_id,
            start_date=start_date
        )
        
        # Get trade ledger summary
        ledger_summary = self.trade_ledger.get_ledger_summary(
            portfolio_id=portfolio_id,
            start_date=start_date
        )
        
        # Get rolling metrics
        rolling_returns = self.pnl_calculator.calculate_rolling_metrics(
            identifier=portfolio_id,
            window_days=7,
            metric_type="return"
        )
        
        return {
            'portfolio_id': portfolio_id,
            'period_days': days,
            'generated_at': datetime.now().isoformat(),
            'pnl_metrics': {
                'total_pnl': float(metrics.total_pnl),
                'realized_pnl': float(metrics.realized_pnl),
                'unrealized_pnl': float(metrics.unrealized_pnl),
                'total_return_pct': metrics.total_return_pct,
                'annualized_return_pct': metrics.annualized_return_pct,
                'volatility_pct': metrics.volatility_pct,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown_pct': metrics.max_drawdown_pct,
                'calmar_ratio': metrics.calmar_ratio
            },
            'trade_statistics': {
                'total_trades': ledger_summary.total_trades,
                'win_rate': ledger_summary.win_rate,
                'avg_trade_size': float(ledger_summary.avg_trade_size),
                'largest_win': float(ledger_summary.largest_win),
                'largest_loss': float(ledger_summary.largest_loss),
                'total_commission': float(ledger_summary.total_commission)
            },
            'rolling_metrics': {
                'avg_7day_return': rolling_returns.avg_return,
                'min_7day_return': rolling_returns.min_return,
                'max_7day_return': rolling_returns.max_return,
                'return_skewness': rolling_returns.return_skewness
            },
            'benchmark_comparison': {
                'excess_return_pct': perf_metrics.excess_return_pct,
                'tracking_error_pct': perf_metrics.tracking_error_pct,
                'beta': perf_metrics.beta
            }
        }
    
    def _check_risk_limits(self, portfolio_id: str, snapshot):
        """Check risk limits and generate alerts."""
        
        # Portfolio value limit
        if snapshot.total_value > Decimal("500000"):
            logger.warning(f"Portfolio {portfolio_id} exceeds value limit: ${snapshot.total_value:,.2f}")
        
        # Drawdown limit
        if snapshot.max_drawdown_pct > 20.0:
            logger.warning(f"Portfolio {portfolio_id} exceeds drawdown limit: {snapshot.max_drawdown_pct:.1f}%")
        
        # Concentration limit
        if snapshot.positions:
            max_position_pct = max(
                abs(float(pos.market_value)) / float(snapshot.total_value) * 100
                for pos in snapshot.positions
            )
            if max_position_pct > 25.0:
                logger.warning(f"Portfolio {portfolio_id} has concentrated position: {max_position_pct:.1f}%")
    
    def _check_pre_trade_risk(
        self,
        portfolio_id: str,
        symbol: str,
        side: str,
        quantity: int,
        snapshot
    ) -> Dict:
        """Check risk limits before executing trade."""
        
        # Simple risk checks
        estimated_trade_value = quantity * 100  # Estimate using $100/share
        
        # Check position size limit
        if estimated_trade_value > 25000:  # $25k position limit
            return {'allowed': False, 'reason': 'Position size exceeds limit'}
        
        # Check portfolio value limit
        if side.lower() == 'buy' and snapshot.cash < estimated_trade_value:
            return {'allowed': False, 'reason': 'Insufficient cash'}
        
        # Check concentration limit
        current_value = float(snapshot.total_value)
        if current_value > 0:
            trade_pct = estimated_trade_value / current_value * 100
            if trade_pct > 30.0:
                return {'allowed': False, 'reason': 'Trade would create concentration risk'}
        
        return {'allowed': True, 'reason': 'Risk checks passed'}
    
    def _simulate_execution(self, symbol: str, side: str, quantity: int, order_type: str):
        """Simulate trade execution for demo purposes."""
        
        from bot.brokers.alpaca.alpaca_executor import ExecutionResult
        
        # Simulate realistic prices
        price_map = {
            'AAPL': 150.00,
            'GOOGL': 2500.00,
            'TSLA': 800.00,
            'MSFT': 300.00,
            'SPY': 420.00
        }
        
        price = price_map.get(symbol, 100.00)
        
        # Add small random variation
        import random
        price *= (1 + random.uniform(-0.001, 0.001))
        
        return ExecutionResult.success_result(
            order_id=f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            order_data={
                'symbol': symbol,
                'side': side,
                'qty': quantity,
                'price': price,
                'status': 'filled'
            },
            message="Simulated execution"
        )


async def demo_realtime_tracking():
    """Demonstrate the real-time tracking system."""
    
    logger.info("=== Real-time Position Tracking Integration Demo ===")
    
    # Initialize system (without real Alpaca credentials for demo)
    tracking_system = RealTimeTrackingSystem()
    
    try:
        # Start the system
        await tracking_system.start()
        
        # Create demo portfolios
        portfolios = [
            ('MOMENTUM_STRATEGY', 'Momentum Trading Strategy'),
            ('MEAN_REVERSION', 'Mean Reversion Strategy'),
            ('SWING_TRADING', 'Swing Trading Strategy')
        ]
        
        for portfolio_id, strategy_name in portfolios:
            tracking_system.create_portfolio(
                portfolio_id=portfolio_id,
                strategy_name=strategy_name,
                initial_cash=Decimal("100000")
            )
        
        # Simulate some trades
        logger.info("\nExecuting demo trades...")
        
        trades = [
            ('MOMENTUM_STRATEGY', 'AAPL', 'buy', 100),
            ('MOMENTUM_STRATEGY', 'GOOGL', 'buy', 40),
            ('MEAN_REVERSION', 'TSLA', 'buy', 50),
            ('MEAN_REVERSION', 'MSFT', 'buy', 80),
            ('SWING_TRADING', 'SPY', 'buy', 200),
            ('MOMENTUM_STRATEGY', 'AAPL', 'sell', 25),  # Partial close
        ]
        
        for portfolio_id, symbol, side, quantity in trades:
            result = await tracking_system.execute_trade(
                portfolio_id=portfolio_id,
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            
            if result['success']:
                logger.info(f"  ✓ {side.upper()} {quantity} {symbol} in {portfolio_id}")
            else:
                logger.error(f"  ✗ Failed to {side} {symbol}: {result['error']}")
            
            # Small delay between trades
            await asyncio.sleep(0.5)
        
        # Wait for position updates to process
        await asyncio.sleep(2)
        
        # Show portfolio status
        logger.info("\nPortfolio Status:")
        for portfolio_id, _ in portfolios:
            status = tracking_system.get_portfolio_status(portfolio_id)
            
            if 'error' not in status:
                logger.info(f"\n--- {portfolio_id} ---")
                logger.info(f"  Total Value: ${status['total_value']:,.2f}")
                logger.info(f"  Cash: ${status['cash']:,.2f}")
                logger.info(f"  Total P&L: ${status['total_pnl']:,.2f}")
                logger.info(f"  Return: {status['total_return_pct']:.2f}%")
                logger.info(f"  Positions: {status['position_count']}")
                logger.info(f"  Gross Exposure: ${status['gross_exposure']:,.2f}")
                
                if status['positions']:
                    logger.info("  Holdings:")
                    for pos in status['positions']:
                        logger.info(f"    {pos['symbol']}: {pos['quantity']} shares, "
                                  f"P&L: ${pos['total_pnl']:.2f}")
        
        # Generate performance report
        logger.info("\nPerformance Report (MOMENTUM_STRATEGY):")
        report = tracking_system.get_performance_report('MOMENTUM_STRATEGY', days=1)
        
        pnl = report['pnl_metrics']
        trades = report['trade_statistics']
        
        logger.info(f"  Total P&L: ${pnl['total_pnl']:.2f}")
        logger.info(f"  Return: {pnl['total_return_pct']:.2f}%")
        logger.info(f"  Sharpe Ratio: {pnl['sharpe_ratio']:.2f}")
        logger.info(f"  Max Drawdown: {pnl['max_drawdown_pct']:.2f}%")
        logger.info(f"  Total Trades: {trades['total_trades']}")
        logger.info(f"  Win Rate: {trades['win_rate']:.1f}%")
        
        # Run reconciliation (simulated)
        logger.info("\nRunning reconciliation...")
        recon_result = await tracking_system.run_reconciliation()
        
        if 'error' in recon_result:
            logger.info(f"  Note: {recon_result['error']}")
        else:
            logger.info(f"  ✓ Reconciled {recon_result['portfolios_reconciled']} portfolios")
        
        logger.info("\n=== Demo Complete ===")
        logger.info("Real-time tracking system provides:")
        logger.info("  ✓ Live position updates from market data")
        logger.info("  ✓ Real-time P&L calculation")
        logger.info("  ✓ Risk monitoring and alerts")
        logger.info("  ✓ Comprehensive performance analysis")
        logger.info("  ✓ Trade execution integration")
        logger.info("  ✓ Position reconciliation with broker")
        logger.info("  ✓ Historical analysis and reporting")
        
    finally:
        # Clean up
        await tracking_system.stop()
        
        # Remove demo database
        Path("realtime_tracking.db").unlink(missing_ok=True)


if __name__ == "__main__":
    # Run the integration demo
    asyncio.run(demo_realtime_tracking())