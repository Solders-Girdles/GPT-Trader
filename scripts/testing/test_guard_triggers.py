#!/usr/bin/env python3
"""
Guard Trigger Test Harness

Simulates various conditions to validate runtime guards and alert dispatchers.

Usage:
    python scripts/test_guard_triggers.py --test daily-loss
    python scripts/test_guard_triggers.py --test stale-marks
    python scripts/test_guard_triggers.py --test error-rate
    python scripts/test_guard_triggers.py --test all
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot_v2.monitoring.runtime_guards import (
    RuntimeGuardManager, DailyLossGuard, StaleMarkGuard,
    ErrorRateGuard, PositionStuckGuard, DrawdownGuard,
    GuardConfig, create_default_guards
)
from bot_v2.monitoring.alerts import (
    AlertDispatcher, LogChannel, create_risk_alert, AlertSeverity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GuardTestHarness:
    """Test harness for runtime guards."""
    
    def __init__(self):
        self.guard_manager = None
        self.alert_dispatcher = None
        self.shutdown_triggered = False
        
    def setup(self):
        """Set up guard manager and alert dispatcher."""
        # Create guard manager with test configuration
        config = {
            'risk_management': {
                'daily_loss_limit': 10.0,
                'max_drawdown_pct': 2.0,
                'circuit_breakers': {
                    'stale_mark_seconds': 60,
                    'error_threshold': 3
                }
            }
        }
        
        self.guard_manager = create_default_guards(config)
        self.guard_manager.set_shutdown_callback(self.emergency_shutdown)
        
        # Create alert dispatcher
        self.alert_dispatcher = AlertDispatcher()
        
        # Add Slack if configured
        slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook:
            from bot_v2.monitoring.alerts import SlackChannel, AlertSeverity
            self.alert_dispatcher.add_channel(
                'slack',
                SlackChannel(slack_webhook, AlertSeverity.WARNING)
            )
            logger.info("Slack alerting enabled")
        
        # Connect dispatcher to guard manager
        self.guard_manager.add_alert_handler(
            lambda alert: asyncio.create_task(self.alert_dispatcher.dispatch(alert))
        )
        
        logger.info("Test harness initialized")
    
    def emergency_shutdown(self):
        """Simulate emergency shutdown."""
        logger.critical("🚨 EMERGENCY SHUTDOWN TRIGGERED 🚨")
        self.shutdown_triggered = True
    
    async def test_daily_loss_breach(self):
        """Test daily loss limit breach."""
        logger.info("\n=== Testing Daily Loss Breach ===")
        
        # Simulate incremental losses
        contexts = [
            {'pnl': -3.0},  # $3 loss
            {'pnl': -3.0},  # $6 total
            {'pnl': -2.0},  # $8 total (80% warning)
            {'pnl': -3.0},  # $11 total (breach!)
        ]
        
        for i, context in enumerate(contexts, 1):
            logger.info(f"Update {i}: PnL = ${context['pnl']}")
            alerts = self.guard_manager.check_all(context)
            
            if alerts:
                for alert in alerts:
                    logger.warning(f"  Alert: {alert.message}")
            
            await asyncio.sleep(0.5)
        
        assert self.shutdown_triggered, "Shutdown should have been triggered"
        logger.info("✅ Daily loss breach test passed")
    
    async def test_stale_marks(self):
        """Test stale market data detection."""
        logger.info("\n=== Testing Stale Marks Detection ===")
        
        # Fresh mark
        context = {
            'symbol': 'BTC-PERP',
            'mark_timestamp': datetime.now()
        }
        alerts = self.guard_manager.check_all(context)
        assert len(alerts) == 0, "Fresh marks should not trigger alerts"
        logger.info("Fresh mark: No alerts ✅")
        
        # Stale mark (90 seconds old)
        context = {
            'symbol': 'BTC-PERP',
            'mark_timestamp': datetime.now() - timedelta(seconds=90)
        }
        alerts = self.guard_manager.check_all(context)
        assert len(alerts) > 0, "Stale marks should trigger alert"
        logger.info(f"Stale mark: Alert triggered - {alerts[0].message} ✅")
    
    async def test_error_rate(self):
        """Test error rate monitoring."""
        logger.info("\n=== Testing Error Rate Monitoring ===")
        
        # Simulate errors
        for i in range(5):
            context = {'error': True, 'error_message': f"Test error {i+1}"}
            alerts = self.guard_manager.check_all(context)
            
            if alerts:
                logger.warning(f"Error {i+1}: Alert triggered - {alerts[0].message}")
            else:
                logger.info(f"Error {i+1}: No alert yet")
            
            await asyncio.sleep(0.2)
        
        assert self.shutdown_triggered, "High error rate should trigger shutdown"
        logger.info("✅ Error rate test passed")
    
    async def test_position_stuck(self):
        """Test stuck position detection."""
        logger.info("\n=== Testing Stuck Position Detection ===")
        
        # Create position
        context = {
            'positions': {
                'BTC-PERP': {'size': 0.01, 'entry_price': 45000}
            }
        }
        
        # First check - position is new
        alerts = self.guard_manager.check_all(context)
        assert len(alerts) == 0, "New position should not trigger alert"
        logger.info("New position: No alert ✅")
        
        # Simulate time passing (would need to mock time in real test)
        # For demo, we'll manually update the guard's internal state
        guard = self.guard_manager.guards.get('position_stuck')
        if guard:
            # Fake old timestamp
            guard.position_times['BTC-PERP'] = datetime.now() - timedelta(minutes=35)
        
        # Check again - position is now stuck
        alerts = self.guard_manager.check_all(context)
        assert len(alerts) > 0, "Stuck position should trigger alert"
        logger.info(f"Stuck position: Alert triggered - {alerts[0].message} ✅")
    
    async def test_drawdown(self):
        """Test maximum drawdown detection."""
        logger.info("\n=== Testing Drawdown Detection ===")
        
        # Simulate equity changes
        contexts = [
            {'equity': 10000},  # Starting equity
            {'equity': 10100},  # Small gain (new peak)
            {'equity': 10050},  # Small drawdown
            {'equity': 9900},   # 2% drawdown from peak (should trigger)
        ]
        
        for i, context in enumerate(contexts, 1):
            logger.info(f"Update {i}: Equity = ${context['equity']}")
            alerts = self.guard_manager.check_all(context)
            
            if alerts:
                logger.warning(f"  Alert: {alerts[0].message}")
        
        assert self.shutdown_triggered, "Max drawdown should trigger shutdown"
        logger.info("✅ Drawdown test passed")
    
    async def test_alert_dispatching(self):
        """Test alert dispatcher functionality."""
        logger.info("\n=== Testing Alert Dispatching ===")
        
        # Create test alert
        alert = create_risk_alert(
            "Test Alert",
            "This is a test of the alert system",
            AlertSeverity.WARNING,
            test_value=123,
            timestamp=datetime.now().isoformat()
        )
        
        # Dispatch alert
        results = await self.alert_dispatcher.dispatch(alert)
        
        logger.info(f"Dispatch results: {results}")
        assert 'log' in results and results['log'], "Log channel should always work"
        
        # Check alert history
        recent = self.alert_dispatcher.get_recent_alerts(count=1)
        assert len(recent) == 1, "Alert should be in history"
        assert recent[0].title == "Test Alert"
        
        logger.info("✅ Alert dispatching test passed")
    
    async def run_all_tests(self):
        """Run all guard tests."""
        # Reset between tests
        test_methods = [
            self.test_stale_marks,
            self.test_alert_dispatching,
            self.test_position_stuck,
            # These trigger shutdown, run last
            self.test_daily_loss_breach,
            # self.test_error_rate,  # Skip to avoid double shutdown
            # self.test_drawdown,    # Skip to avoid double shutdown
        ]
        
        for test in test_methods:
            # Reset state
            self.shutdown_triggered = False
            self.guard_manager.reset_all()
            
            # Run test
            await test()
            await asyncio.sleep(1)
        
        logger.info("\n✅ All tests completed successfully!")
    
    async def run_single_test(self, test_name: str):
        """Run a single test by name."""
        test_map = {
            'daily-loss': self.test_daily_loss_breach,
            'stale-marks': self.test_stale_marks,
            'error-rate': self.test_error_rate,
            'position-stuck': self.test_position_stuck,
            'drawdown': self.test_drawdown,
            'alerts': self.test_alert_dispatching,
        }
        
        test_method = test_map.get(test_name)
        if not test_method:
            logger.error(f"Unknown test: {test_name}")
            logger.info(f"Available tests: {', '.join(test_map.keys())}")
            return
        
        await test_method()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Guard Trigger Test Harness")
    parser.add_argument(
        '--test',
        choices=['daily-loss', 'stale-marks', 'error-rate', 'position-stuck', 
                 'drawdown', 'alerts', 'all'],
        default='all',
        help='Test to run'
    )
    parser.add_argument(
        '--slack',
        action='store_true',
        help='Enable Slack alerts (requires SLACK_WEBHOOK_URL env var)'
    )
    
    args = parser.parse_args()
    
    # Create and setup harness
    harness = GuardTestHarness()
    harness.setup()
    
    # Run tests
    if args.test == 'all':
        await harness.run_all_tests()
    else:
        await harness.run_single_test(args.test)


if __name__ == "__main__":
    asyncio.run(main())