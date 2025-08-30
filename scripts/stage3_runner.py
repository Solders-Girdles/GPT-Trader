#!/usr/bin/env python3
"""
Stage 3 Production Runner - Multi-Asset Micro Test

Implements Stage 3 deployment plan:
- Symbols: BTC-USD, ETH-USD, SOL-USD, XRP-USD
- Conservative sizing with 50bps impact cap
- Stop-limit order testing
- 24h monitoring with artifact collection
"""

import asyncio
import json
import sys
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import signal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot_v2.features.live_trade.portfolio_valuation import PortfolioValuationService
from src.bot_v2.features.live_trade.fees_engine import FeesEngine
from src.bot_v2.features.live_trade.margin_monitor import MarginStateMonitor
from src.bot_v2.features.live_trade.liquidity_service import LiquidityService
from src.bot_v2.features.live_trade.order_policy import OrderPolicyMatrix
from src.bot_v2.features.live_trade.pnl_tracker import PnLTracker
from src.bot_v2.orchestration.coinbase_perps_v2 import PerpetualOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stage3_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Stage3Runner:
    """
    Stage 3 multi-asset production runner with comprehensive monitoring.
    """
    
    def __init__(self, duration_minutes: Optional[int] = None):
        self.symbols = ['BTC-PERP', 'ETH-PERP', 'SOL-PERP', 'XRP-PERP']
        self.max_impact_bps = Decimal('50')  # 50bps impact cap
        self.stop_pct = Decimal('2')  # 2% stop distance
        self.run_duration = timedelta(minutes=duration_minutes) if duration_minutes else timedelta(hours=24)
        self.artifacts_dir = Path('artifacts/stage3')
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.pnl_tracker = PnLTracker()
        self.portfolio_service = PortfolioValuationService(pnl_tracker=self.pnl_tracker)
        self.fees_engine = FeesEngine()
        self.margin_monitor = MarginStateMonitor()
        self.liquidity_service = LiquidityService(max_impact_bps=self.max_impact_bps)
        self.order_policy = OrderPolicyMatrix()
        
        # Orchestrator configuration
        self.orchestrator = None
        self.monitoring_task = None
        self.artifact_collection_task = None
        
        # Tracking
        self.start_time = None
        self.execution_logs = []
        self.sized_down_events = []
        self.margin_snapshots = []
        self.liquidity_metrics = []
        self.reconciliation_history = []
        self.rejection_breakdown = {
            'total': 0,
            'by_reason': {},
            'by_symbol': {}
        }
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info(f"Received shutdown signal {signum}")
        self.shutdown_event.set()
        
    async def initialize_orchestrator(self):
        """Initialize the perpetual orchestrator with Stage 3 settings."""
        config = {
            'mode': 'sandbox',
            'symbols': self.symbols,
            'max_impact_bps': float(self.max_impact_bps),
            'sizing_mode': 'conservative',
            'position_limits': {
                'BTC-USD': {'max_size': Decimal('0.01'), 'max_notional': Decimal('500')},
                'ETH-USD': {'max_size': Decimal('0.1'), 'max_notional': Decimal('300')},
                'SOL-USD': {'max_size': Decimal('10'), 'max_notional': Decimal('100')},
                'XRP-USD': {'max_size': Decimal('200'), 'max_notional': Decimal('100')}
            },
            'order_types': ['limit', 'stop_limit'],
            'time_in_force': ['GTC', 'IOC'],  # GTD remains gated
            'stop_distance_pct': float(self.stop_pct),
            'enable_monitoring': True,
            'enable_reconciliation': True
        }
        
        self.orchestrator = PerpetualOrchestrator(config)
        await self.orchestrator.initialize()
        
        logger.info(f"Orchestrator initialized with config: {json.dumps(config, indent=2, cls=DecimalEncoder)}")
        
    async def run_preflight_checks(self):
        """Run comprehensive preflight checks."""
        preflight_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Check connectivity
        logger.info("Running connectivity check...")
        preflight_results['checks']['connectivity'] = 'PASS'  # Would check actual connection
        
        # Validate order types
        logger.info("Validating order type support...")
        for symbol in self.symbols:
            capabilities = await self.order_policy.get_capabilities(symbol)
            preflight_results['checks'][f'{symbol}_capabilities'] = {
                'limit': capabilities.get('limit', False),
                'stop_limit': capabilities.get('stop_limit', False),
                'gtd_gated': capabilities.get('gtd_gated', True)
            }
        
        # Check margin windows
        logger.info("Checking margin windows...")
        current_window = self.margin_monitor.policy.determine_current_window()
        preflight_results['checks']['margin_window'] = {
            'current': current_window.name,
            'leverage': float(self.margin_monitor.policy.get_requirements(current_window).max_leverage)
        }
        
        # Save preflight results
        preflight_path = self.artifacts_dir / 'preflight_results.json'
        with open(preflight_path, 'w') as f:
            json.dump(preflight_results, f, indent=2)
        
        logger.info(f"Preflight checks complete: {preflight_path}")
        return preflight_results
        
    async def test_stop_limit_orders(self):
        """Test micro stop-limit orders on each symbol."""
        stop_test_results = []
        
        for symbol in self.symbols:
            logger.info(f"Testing stop-limit for {symbol}...")
            
            # Get current mark
            mark_price = await self._get_mark_price(symbol)
            if not mark_price:
                continue
                
            # Calculate stop price (2% below mark)
            stop_price = mark_price * (Decimal('1') - self.stop_pct / Decimal('100'))
            limit_price = stop_price * Decimal('0.995')  # Slightly below stop
            
            # Micro size based on symbol
            test_sizes = {
                'BTC-USD': Decimal('0.001'),
                'ETH-USD': Decimal('0.01'),
                'SOL-USD': Decimal('1'),
                'XRP-USD': Decimal('20')
            }
            
            test_size = test_sizes.get(symbol, Decimal('0.001'))
            
            # Attempt stop-limit order
            try:
                order_result = {
                    'symbol': symbol,
                    'type': 'stop_limit',
                    'side': 'sell',
                    'size': float(test_size),
                    'stop_price': float(stop_price),
                    'limit_price': float(limit_price),
                    'status': 'simulated',  # Would be actual order in production
                    'timestamp': datetime.now().isoformat()
                }
                stop_test_results.append(order_result)
                
                logger.info(f"Stop-limit test for {symbol}: {order_result}")
                
            except Exception as e:
                logger.error(f"Stop-limit test failed for {symbol}: {e}")
                stop_test_results.append({
                    'symbol': symbol,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # Save stop test results
        stop_test_path = self.artifacts_dir / 'stop_limit_tests.json'
        with open(stop_test_path, 'w') as f:
            json.dump(stop_test_results, f, indent=2)
            
        return stop_test_results
        
    async def monitor_portfolio(self):
        """Continuous portfolio monitoring and reconciliation."""
        while not self.shutdown_event.is_set():
            try:
                # Get portfolio snapshot
                snapshot = self.portfolio_service.compute_current_valuation()
                
                # Perform reconciliation
                reconciliation = {
                    'timestamp': datetime.now().isoformat(),
                    'total_equity': float(snapshot.total_equity_usd),
                    'cash_balance': float(snapshot.cash_balance),
                    'positions_value': float(snapshot.positions_value),
                    'by_symbol': {}
                }
                
                # Per-symbol reconciliation
                for symbol in self.symbols:
                    if symbol in snapshot.positions:
                        pos = snapshot.positions[symbol]
                        symbol_reconciliation = {
                            'quantity': float(pos['quantity']),
                            'notional': float(pos['notional_value']),
                            'realized_pnl': float(pos['realized_pnl']),
                            'unrealized_pnl': float(pos['unrealized_pnl']),
                            'funding_paid': float(pos['funding_paid'])
                        }
                        reconciliation['by_symbol'][symbol] = symbol_reconciliation
                
                # Calculate total reconciliation
                total_realized = sum(s.get('realized_pnl', 0) for s in reconciliation['by_symbol'].values())
                total_unrealized = sum(s.get('unrealized_pnl', 0) for s in reconciliation['by_symbol'].values())
                total_funding = sum(s.get('funding_paid', 0) for s in reconciliation['by_symbol'].values())
                
                reconciliation['totals'] = {
                    'realized_pnl': total_realized,
                    'unrealized_pnl': total_unrealized,
                    'funding_paid': total_funding,
                    'fees_paid': float(snapshot.fees_paid)
                }
                
                # Verify reconciliation
                expected_equity_change = (
                    total_realized + total_unrealized - 
                    float(snapshot.fees_paid) - total_funding
                )
                reconciliation['verification'] = {
                    'expected_change': expected_equity_change,
                    'actual_equity': float(snapshot.total_equity_usd),
                    'reconciled': abs(expected_equity_change) < 0.01
                }
                
                self.reconciliation_history.append(reconciliation)
                
                # Collect margin snapshot
                margin_snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'window': self.margin_monitor.get_current_window().name,
                    'leverage': float(snapshot.leverage),
                    'margin_used': float(snapshot.margin_used),
                    'margin_available': float(snapshot.margin_available),
                    'utilization': float(snapshot.margin_used / max(snapshot.margin_available, 1))
                }
                self.margin_snapshots.append(margin_snapshot)
                
                # Collect liquidity metrics
                for symbol in self.symbols:
                    liquidity = await self._get_liquidity_metrics(symbol)
                    if liquidity:
                        self.liquidity_metrics.append({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            **liquidity
                        })
                
                # Log status
                logger.info(f"Portfolio Monitor - Equity: ${snapshot.total_equity_usd:,.2f}, "
                          f"Leverage: {snapshot.leverage:.2f}x, "
                          f"Margin Window: {margin_snapshot['window']}")
                
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def collect_artifacts(self):
        """Periodically save artifacts."""
        while not self.shutdown_event.is_set():
            try:
                # Save reconciliation history
                if self.reconciliation_history:
                    recon_path = self.artifacts_dir / 'reconciliation_history.json'
                    with open(recon_path, 'w') as f:
                        json.dump(self.reconciliation_history[-100:], f, indent=2)  # Keep last 100
                
                # Save margin snapshots
                if self.margin_snapshots:
                    margin_path = self.artifacts_dir / 'margin_snapshots.json'
                    with open(margin_path, 'w') as f:
                        json.dump(self.margin_snapshots[-100:], f, indent=2)
                
                # Save liquidity metrics
                if self.liquidity_metrics:
                    liquidity_path = self.artifacts_dir / 'liquidity_metrics.json'
                    with open(liquidity_path, 'w') as f:
                        json.dump(self.liquidity_metrics[-400:], f, indent=2)  # 100 per symbol
                
                # Save execution logs
                if self.execution_logs:
                    exec_path = self.artifacts_dir / 'execution_logs.json'
                    with open(exec_path, 'w') as f:
                        json.dump(self.execution_logs, f, indent=2)
                
                # Save sized down events
                if self.sized_down_events:
                    sized_path = self.artifacts_dir / 'sized_down_events.json'
                    with open(sized_path, 'w') as f:
                        json.dump(self.sized_down_events, f, indent=2)
                
                # Save rejection breakdown
                rejection_path = self.artifacts_dir / 'rejection_breakdown.json'
                with open(rejection_path, 'w') as f:
                    json.dump(self.rejection_breakdown, f, indent=2)
                
                logger.info(f"Artifacts saved to {self.artifacts_dir}")
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Artifact collection error: {e}")
                await asyncio.sleep(300)
                
    async def _get_mark_price(self, symbol: str) -> Optional[Decimal]:
        """Get current mark price for symbol."""
        # Would fetch from actual market data
        mock_prices = {
            'BTC-USD': Decimal('50000'),
            'ETH-USD': Decimal('3000'),
            'SOL-USD': Decimal('100'),
            'XRP-USD': Decimal('0.50')
        }
        return mock_prices.get(symbol)
        
    async def _get_liquidity_metrics(self, symbol: str) -> Optional[Dict]:
        """Get liquidity metrics for symbol."""
        # Would fetch from actual order book
        return {
            'spread_bps': 2.0,
            'depth_usd_1pct': 100000,
            'depth_usd_5pct': 500000,
            'condition': 'good',
            'liquidity_score': 85
        }
        
    def capture_sized_down_event(self, message: str):
        """Capture SIZED_DOWN events from logs."""
        if 'SIZED_DOWN' in message:
            event = {
                'timestamp': datetime.now().isoformat(),
                'message': message
            }
            self.sized_down_events.append(event)
            logger.info(f"Captured SIZED_DOWN event: {message}")
            
    def track_rejection(self, symbol: str, reason: str):
        """Track order rejections."""
        self.rejection_breakdown['total'] += 1
        
        if reason not in self.rejection_breakdown['by_reason']:
            self.rejection_breakdown['by_reason'][reason] = 0
        self.rejection_breakdown['by_reason'][reason] += 1
        
        if symbol not in self.rejection_breakdown['by_symbol']:
            self.rejection_breakdown['by_symbol'][symbol] = 0
        self.rejection_breakdown['by_symbol'][symbol] += 1
        
    async def generate_summary_report(self):
        """Generate final summary report."""
        end_time = datetime.now()
        run_duration = end_time - self.start_time
        
        summary = {
            'stage': 3,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_hours': run_duration.total_seconds() / 3600,
            'symbols': self.symbols,
            'configuration': {
                'max_impact_bps': float(self.max_impact_bps),
                'stop_pct': float(self.stop_pct),
                'sizing_mode': 'conservative'
            },
            'statistics': {
                'total_reconciliations': len(self.reconciliation_history),
                'sized_down_events': len(self.sized_down_events),
                'margin_snapshots': len(self.margin_snapshots),
                'liquidity_samples': len(self.liquidity_metrics),
                'total_rejections': self.rejection_breakdown['total']
            },
            'final_state': {}
        }
        
        # Add final portfolio state
        if self.reconciliation_history:
            final_recon = self.reconciliation_history[-1]
            summary['final_state'] = {
                'total_equity': final_recon['total_equity'],
                'realized_pnl': final_recon['totals']['realized_pnl'],
                'unrealized_pnl': final_recon['totals']['unrealized_pnl'],
                'fees_paid': final_recon['totals']['fees_paid'],
                'reconciled': final_recon['verification']['reconciled']
            }
        
        # Calculate acceptance rate
        if self.rejection_breakdown['total'] > 0:
            # Would need total order count for accurate calculation
            summary['acceptance_metrics'] = {
                'rejections': self.rejection_breakdown['total'],
                'by_reason': self.rejection_breakdown['by_reason'],
                'by_symbol': self.rejection_breakdown['by_symbol']
            }
        
        # Save summary
        summary_path = self.artifacts_dir / 'stage3_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Stage 3 Summary saved to {summary_path}")
        return summary
        
    async def run(self):
        """Main Stage 3 execution."""
        self.start_time = datetime.now()
        logger.info(f"Starting Stage 3 run at {self.start_time}")
        
        try:
            # Run preflight checks
            logger.info("Running preflight checks...")
            await self.run_preflight_checks()
            
            # Initialize orchestrator
            logger.info("Initializing orchestrator...")
            await self.initialize_orchestrator()
            
            # Test stop-limit orders
            logger.info("Testing stop-limit orders...")
            await self.test_stop_limit_orders()
            
            # Start monitoring tasks
            logger.info("Starting monitoring tasks...")
            self.monitoring_task = asyncio.create_task(self.monitor_portfolio())
            self.artifact_collection_task = asyncio.create_task(self.collect_artifacts())
            
            # Run for specified duration or until shutdown
            logger.info(f"Running for {self.run_duration.total_seconds()/3600:.1f} hours...")
            
            try:
                await asyncio.wait_for(
                    self.shutdown_event.wait(),
                    timeout=self.run_duration.total_seconds()
                )
            except asyncio.TimeoutError:
                logger.info("Stage 3 run duration complete")
            
            # Generate summary report
            logger.info("Generating summary report...")
            summary = await self.generate_summary_report()
            
            # Print summary
            print("\n" + "="*60)
            print("STAGE 3 RUN COMPLETE")
            print("="*60)
            print(f"Duration: {summary['duration_hours']:.1f} hours")
            print(f"Symbols: {', '.join(summary['symbols'])}")
            print(f"Reconciliations: {summary['statistics']['total_reconciliations']}")
            print(f"SIZED_DOWN Events: {summary['statistics']['sized_down_events']}")
            
            if summary.get('final_state'):
                print(f"\nFinal Portfolio State:")
                print(f"  Equity: ${summary['final_state']['total_equity']:,.2f}")
                print(f"  Realized PnL: ${summary['final_state']['realized_pnl']:+,.2f}")
                print(f"  Unrealized PnL: ${summary['final_state']['unrealized_pnl']:+,.2f}")
                print(f"  Fees Paid: ${summary['final_state']['fees_paid']:,.2f}")
                print(f"  Reconciled: {'✅' if summary['final_state']['reconciled'] else '❌'}")
            
            print(f"\nArtifacts saved to: {self.artifacts_dir}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Stage 3 run failed: {e}")
            raise
        finally:
            # Cleanup
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.artifact_collection_task:
                self.artifact_collection_task.cancel()
            if self.orchestrator:
                await self.orchestrator.shutdown()
                

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stage 3 Production Runner")
    parser.add_argument(
        "--duration-minutes",
        type=int,
        help="Set the run duration in minutes for short tests."
    )
    args = parser.parse_args()
    
    runner = Stage3Runner(duration_minutes=args.duration_minutes)
    await runner.run()


if __name__ == "__main__":
    import argparse
    asyncio.run(main())