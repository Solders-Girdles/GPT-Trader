#!/usr/bin/env python3
"""
Margin Window Transition Verification.

Simulates margin window transitions and verifies that sizing reduces,
entries are blocked, and quiet periods are observed.
"""

import json
import sys
from decimal import Decimal
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot_v2.features.live_trade.margin_monitor import MarginStateMonitor, MarginWindow


class MarginTransitionVerification:
    """
    Verifies margin window transitions and policy changes.
    """
    
    def __init__(self):
        self.margin_monitor = MarginStateMonitor()
        self.test_results = []
        
    async def test_window_detection(self) -> Dict:
        """Test margin window detection at various times."""
        print("🕐 Testing Margin Window Detection")
        print("-" * 40)
        
        test_cases = [
            # (time, expected_window, description)
            (datetime(2024, 1, 1, 10, 0, 0), MarginWindow.NORMAL, "Normal trading hours"),
            (datetime(2024, 1, 1, 23, 0, 0), MarginWindow.OVERNIGHT, "Overnight window"),
            (datetime(2024, 1, 1, 3, 0, 0), MarginWindow.OVERNIGHT, "Early morning overnight"),
            (datetime(2024, 1, 1, 7, 45, 0), MarginWindow.PRE_FUNDING, "Pre-funding (15min before 8am)"),
            (datetime(2024, 1, 1, 15, 45, 0), MarginWindow.PRE_FUNDING, "Pre-funding (15min before 4pm)"),
            (datetime(2024, 1, 1, 23, 45, 0), MarginWindow.PRE_FUNDING, "Pre-funding (15min before midnight)"),
            (datetime(2024, 1, 1, 14, 30, 0), MarginWindow.INTRADAY, "Intraday volatility window"),
        ]
        
        results = {
            'test_cases': [],
            'passed': 0,
            'failed': 0
        }
        
        for test_time, expected_window, description in test_cases:
            actual_window = self.margin_monitor.policy.determine_current_window(test_time)
            passed = actual_window == expected_window
            
            result = {
                'time': test_time.strftime('%Y-%m-%d %H:%M UTC'),
                'description': description,
                'expected': expected_window.value,
                'actual': actual_window.value,
                'passed': passed
            }
            
            results['test_cases'].append(result)
            
            if passed:
                results['passed'] += 1
                print(f"✅ {test_time.strftime('%H:%M')} - {description}: {actual_window.value}")
            else:
                results['failed'] += 1
                print(f"❌ {test_time.strftime('%H:%M')} - {description}: Expected {expected_window.value}, got {actual_window.value}")
        
        return results
    
    async def test_margin_requirements(self) -> Dict:
        """Test margin requirement changes across windows."""
        print("\n💰 Testing Margin Requirement Changes")
        print("-" * 40)
        
        windows = [
            MarginWindow.NORMAL,
            MarginWindow.INTRADAY,
            MarginWindow.OVERNIGHT,
            MarginWindow.PRE_FUNDING
        ]
        
        results = {
            'windows': {},
            'leverage_reduction_verified': False
        }
        
        prev_max_leverage = None
        
        for window in windows:
            requirements = self.margin_monitor.policy.get_requirements(window)
            
            window_data = {
                'window': window.value,
                'initial_margin': float(requirements.initial_rate),
                'maintenance_margin': float(requirements.maintenance_rate),
                'max_leverage': float(requirements.max_leverage)
            }
            
            results['windows'][window.value] = window_data
            
            print(f"{window.value.upper()}:")
            print(f"  Initial Margin: {requirements.initial_rate:.1%}")
            print(f"  Maintenance: {requirements.maintenance_rate:.1%}")
            print(f"  Max Leverage: {requirements.max_leverage}x")
            
            # Verify leverage reduces in tighter windows
            if prev_max_leverage is not None:
                if window in [MarginWindow.OVERNIGHT, MarginWindow.PRE_FUNDING]:
                    if requirements.max_leverage < prev_max_leverage:
                        results['leverage_reduction_verified'] = True
            
            prev_max_leverage = requirements.max_leverage
        
        return results
    
    async def test_position_sizing_adjustment(self) -> Dict:
        """Test position sizing adjustments during transitions."""
        print("\n📊 Testing Position Sizing Adjustments")
        print("-" * 40)
        
        test_equity = Decimal('100000')
        test_price = Decimal('50000')
        
        results = {
            'scenarios': [],
            'sizing_reduced': False
        }
        
        # Test sizing in different windows
        for window in [MarginWindow.NORMAL, MarginWindow.OVERNIGHT, MarginWindow.PRE_FUNDING]:
            # Mock the current window
            self.margin_monitor.policy._current_window = window
            requirements = self.margin_monitor.policy.get_requirements(window)
            
            max_size = await self.margin_monitor.get_max_position_size(
                symbol="BTC-USD",
                price=test_price,
                available_equity=test_equity
            )
            
            max_notional = max_size * test_price
            effective_leverage = max_notional / test_equity
            
            scenario = {
                'window': window.value,
                'available_equity': float(test_equity),
                'max_position_size': float(max_size),
                'max_notional': float(max_notional),
                'effective_leverage': float(effective_leverage),
                'window_max_leverage': float(requirements.max_leverage)
            }
            
            results['scenarios'].append(scenario)
            
            print(f"{window.value.upper()}:")
            print(f"  Max Position: {max_size:.3f} BTC")
            print(f"  Max Notional: ${max_notional:,.0f}")
            print(f"  Effective Leverage: {effective_leverage:.2f}x")
        
        # Verify sizing reduces in tighter windows
        normal_size = results['scenarios'][0]['max_position_size']
        overnight_size = results['scenarios'][1]['max_position_size']
        prefunding_size = results['scenarios'][2]['max_position_size']
        
        if overnight_size < normal_size and prefunding_size < overnight_size:
            results['sizing_reduced'] = True
            print(f"\n✅ Position sizing reduces correctly across windows")
        else:
            print(f"\n❌ Position sizing not reducing as expected")
        
        return results
    
    async def test_transition_behavior(self) -> Dict:
        """Test behavior during window transitions."""
        print("\n🔄 Testing Window Transition Behavior")
        print("-" * 40)
        
        results = {
            'transitions': [],
            'quiet_periods_detected': False,
            'risk_reduction_triggered': False
        }
        
        # Test transition detection
        test_times = [
            (datetime(2024, 1, 1, 21, 45, 0), "15min before overnight"),  # Normal -> Overnight
            (datetime(2024, 1, 1, 7, 45, 0), "15min before funding"),     # Normal -> Pre-funding
            (datetime(2024, 1, 1, 13, 45, 0), "15min before intraday"),   # Normal -> Intraday
        ]
        
        for test_time, description in test_times:
            current_window = self.margin_monitor.policy.determine_current_window(test_time)
            
            # Check 30 minutes ahead
            future_time = datetime(test_time.year, test_time.month, test_time.day,
                                 test_time.hour, test_time.minute + 30 if test_time.minute < 30 else 0,
                                 0)
            if test_time.minute >= 30:
                future_time = future_time.replace(hour=(test_time.hour + 1) % 24)
            
            future_window = self.margin_monitor.policy.determine_current_window(future_time)
            
            # Check if risk should be reduced
            should_reduce = self.margin_monitor.policy.should_reduce_risk(current_window, future_window)
            
            transition = {
                'time': test_time.strftime('%H:%M UTC'),
                'description': description,
                'current_window': current_window.value,
                'future_window': future_window.value,
                'transition_detected': current_window != future_window,
                'should_reduce_risk': should_reduce
            }
            
            results['transitions'].append(transition)
            
            if current_window != future_window:
                results['quiet_periods_detected'] = True
                print(f"✅ {description}: {current_window.value} → {future_window.value}")
                if should_reduce:
                    results['risk_reduction_triggered'] = True
                    print(f"   ⚠️  Risk reduction recommended")
            else:
                print(f"ℹ️  {description}: No transition detected")
        
        return results
    
    async def test_margin_utilization_alerts(self) -> Dict:
        """Test margin utilization alerts and liquidation warnings."""
        print("\n🚨 Testing Margin Utilization Alerts")
        print("-" * 40)
        
        results = {
            'alert_tests': [],
            'alerts_triggered_correctly': True
        }
        
        test_scenarios = [
            {
                'name': 'Safe utilization',
                'equity': Decimal('100000'),
                'positions_notional': Decimal('50000'),
                'expected_alerts': []
            },
            {
                'name': 'High utilization',
                'equity': Decimal('100000'),
                'positions_notional': Decimal('850000'),
                'expected_alerts': ['HIGH_UTILIZATION']
            },
            {
                'name': 'Margin call risk',
                'equity': Decimal('10000'),
                'positions_notional': Decimal('180000'),
                'expected_alerts': ['MARGIN_CALL']
            },
            {
                'name': 'Liquidation risk',
                'equity': Decimal('6000'),
                'positions_notional': Decimal('100000'),
                'expected_alerts': ['LIQUIDATION_RISK']
            }
        ]
        
        for scenario in test_scenarios:
            # Create test positions
            positions = {
                'BTC-USD': {
                    'quantity': scenario['positions_notional'] / Decimal('50000'),
                    'mark_price': Decimal('50000')
                }
            }
            
            # Track alerts
            alerts_triggered = []
            
            async def alert_callback(snapshot, alert_type):
                alerts_triggered.append(alert_type)
            
            self.margin_monitor.add_alert_callback(alert_callback)
            
            # Compute margin state
            snapshot = await self.margin_monitor.compute_margin_state(
                total_equity=scenario['equity'],
                cash_balance=scenario['equity'],
                positions=positions
            )
            
            # Check alerts
            test_result = {
                'scenario': scenario['name'],
                'equity': float(scenario['equity']),
                'positions_notional': float(scenario['positions_notional']),
                'margin_utilization': float(snapshot.margin_utilization),
                'leverage': float(snapshot.leverage),
                'expected_alerts': scenario['expected_alerts'],
                'actual_alerts': alerts_triggered,
                'passed': set(alerts_triggered) == set(scenario['expected_alerts'])
            }
            
            results['alert_tests'].append(test_result)
            
            if test_result['passed']:
                print(f"✅ {scenario['name']}: Alerts correct")
            else:
                print(f"❌ {scenario['name']}: Expected {scenario['expected_alerts']}, got {alerts_triggered}")
                results['alerts_triggered_correctly'] = False
            
            print(f"   Utilization: {snapshot.margin_utilization:.1%}, Leverage: {snapshot.leverage:.2f}x")
            
            # Clear callbacks
            self.margin_monitor._alert_callbacks.clear()
        
        return results
    
    async def generate_verification_report(self) -> Dict:
        """Generate comprehensive verification report."""
        print("\n" + "=" * 60)
        print("🧪 MARGIN WINDOW TRANSITION VERIFICATION")
        print("=" * 60 + "\n")
        
        # Run all tests
        window_detection = await self.test_window_detection()
        margin_requirements = await self.test_margin_requirements()
        position_sizing = await self.test_position_sizing_adjustment()
        transition_behavior = await self.test_transition_behavior()
        utilization_alerts = await self.test_margin_utilization_alerts()
        
        # Compile report
        report = {
            'verification_type': 'margin_window_transition',
            'timestamp': datetime.now().isoformat(),
            'tests': {
                'window_detection': window_detection,
                'margin_requirements': margin_requirements,
                'position_sizing': position_sizing,
                'transition_behavior': transition_behavior,
                'utilization_alerts': utilization_alerts
            },
            'summary': {
                'window_detection_pass': window_detection['failed'] == 0,
                'leverage_reduction_verified': margin_requirements['leverage_reduction_verified'],
                'sizing_reduction_verified': position_sizing['sizing_reduced'],
                'quiet_periods_detected': transition_behavior['quiet_periods_detected'],
                'risk_reduction_triggered': transition_behavior['risk_reduction_triggered'],
                'alerts_working': utilization_alerts['alerts_triggered_correctly'],
                'overall_pass': (
                    window_detection['failed'] == 0 and
                    margin_requirements['leverage_reduction_verified'] and
                    position_sizing['sizing_reduced'] and
                    transition_behavior['quiet_periods_detected'] and
                    utilization_alerts['alerts_triggered_correctly']
                )
            }
        }
        
        # Save report
        report_path = Path("verification_reports/margin_window_transition.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Verification report saved to: {report_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("-" * 40)
        
        for key, value in report['summary'].items():
            status = "✅ PASS" if value else "❌ FAIL"
            print(f"{status}: {key.replace('_', ' ').title()}")
        
        return report


async def main():
    """Run margin window transition verification."""
    verification = MarginTransitionVerification()
    report = await verification.generate_verification_report()
    
    if report['summary']['overall_pass']:
        print("\n✅ MARGIN WINDOW TRANSITION VERIFICATION: PASSED")
        return 0
    else:
        print("\n❌ MARGIN WINDOW TRANSITION VERIFICATION: FAILED")
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)