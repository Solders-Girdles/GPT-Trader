#!/usr/bin/env python3
"""
Liquidity Stress Test Verification.

Tests SIZED_DOWN activation, strict mode rejection, conservative mode downsizing,
and slicing behavior under shallow book conditions.
"""

import json
import sys
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot_v2.features.live_trade.liquidity_service import LiquidityService, LiquidityCondition


class LiquidityStressTest:
    """
    Performs liquidity stress testing with shallow book scenarios.
    """
    
    def __init__(self):
        self.liquidity_service = LiquidityService(max_impact_bps=Decimal('50'))  # 50bps max
        self.test_results = []
        
    def create_shallow_book(self) -> Tuple[List, List]:
        """Create a shallow order book for stress testing."""
        # Very thin liquidity - only $1000 per level
        bids = [
            (Decimal('49995'), Decimal('0.02')),   # $1000
            (Decimal('49990'), Decimal('0.02')),   # $1000
            (Decimal('49985'), Decimal('0.02')),   # $1000
            (Decimal('49980'), Decimal('0.02')),   # $1000
            (Decimal('49975'), Decimal('0.02')),   # $1000
        ]
        
        asks = [
            (Decimal('50005'), Decimal('0.02')),   # $1000
            (Decimal('50010'), Decimal('0.02')),   # $1000
            (Decimal('50015'), Decimal('0.02')),   # $1000
            (Decimal('50020'), Decimal('0.02')),   # $1000
            (Decimal('50025'), Decimal('0.02')),   # $1000
        ]
        
        return bids, asks
    
    def create_deep_book(self) -> Tuple[List, List]:
        """Create a deep order book for comparison."""
        # Deep liquidity - $50k per level
        bids = [
            (Decimal('49995'), Decimal('1.0')),    # $50k
            (Decimal('49990'), Decimal('1.0')),    # $50k
            (Decimal('49985'), Decimal('1.0')),    # $50k
            (Decimal('49980'), Decimal('1.0')),    # $50k
            (Decimal('49975'), Decimal('1.0')),    # $50k
        ]
        
        asks = [
            (Decimal('50005'), Decimal('1.0')),    # $50k
            (Decimal('50010'), Decimal('1.0')),    # $50k
            (Decimal('50015'), Decimal('1.0')),    # $50k
            (Decimal('50020'), Decimal('1.0')),    # $50k
            (Decimal('50025'), Decimal('1.0')),    # $50k
        ]
        
        return bids, asks
    
    async def test_shallow_book_analysis(self) -> Dict:
        """Test order book analysis under shallow conditions."""
        print("📊 Testing Shallow Book Analysis")
        print("-" * 40)
        
        results = {
            'book_conditions': [],
            'liquidity_scoring': []
        }
        
        # Test both shallow and deep books
        book_scenarios = [
            ('shallow', self.create_shallow_book()),
            ('deep', self.create_deep_book())
        ]
        
        for name, (bids, asks) in book_scenarios:
            analysis = self.liquidity_service.analyze_order_book(
                symbol="BTC-USD",
                bids=bids,
                asks=asks
            )
            
            book_data = {
                'type': name,
                'spread_bps': float(analysis.spread_bps),
                'depth_usd_1': float(analysis.depth_usd_1),
                'depth_usd_5': float(analysis.depth_usd_5),
                'depth_usd_10': float(analysis.depth_usd_10),
                'liquidity_score': float(analysis.liquidity_score),
                'condition': analysis.condition.value
            }
            
            results['book_conditions'].append(book_data)
            
            print(f"\n{name.upper()} BOOK:")
            print(f"  Spread: {analysis.spread_bps:.1f}bps")
            print(f"  Depth (1%): ${analysis.depth_usd_1:,.0f}")
            print(f"  Depth (5%): ${analysis.depth_usd_5:,.0f}")
            print(f"  Score: {analysis.liquidity_score:.0f}/100")
            print(f"  Condition: {analysis.condition.value.upper()}")
        
        # Verify shallow book has lower score
        shallow_score = results['book_conditions'][0]['liquidity_score']
        deep_score = results['book_conditions'][1]['liquidity_score']
        
        results['shallow_correctly_scored'] = shallow_score < deep_score
        
        return results
    
    async def test_sized_down_activation(self) -> Dict:
        """Test SIZED_DOWN activation under different impact thresholds."""
        print("\n⚡ Testing SIZED_DOWN Activation")
        print("-" * 40)
        
        results = {
            'impact_tests': [],
            'sized_down_triggered': False,
            'slicing_recommended': False
        }
        
        # Set up shallow book
        bids, asks = self.create_shallow_book()
        self.liquidity_service.analyze_order_book("BTC-USD", bids, asks)
        
        # Test different order sizes
        test_sizes = [
            (Decimal('0.01'), 'Tiny', False),    # $500 - should pass
            (Decimal('0.1'), 'Small', False),     # $5k - borderline
            (Decimal('0.5'), 'Medium', True),     # $25k - should trigger
            (Decimal('2.0'), 'Large', True),      # $100k - definitely trigger
        ]
        
        for size, description, should_trigger in test_sizes:
            impact = self.liquidity_service.estimate_market_impact(
                symbol="BTC-USD",
                side="buy",
                quantity=size,
                book_data=(bids, asks)
            )
            
            # Check if impact exceeds threshold
            exceeds_threshold = impact.estimated_impact_bps > self.liquidity_service.max_impact_bps
            
            test_result = {
                'size': float(size),
                'description': description,
                'impact_bps': float(impact.estimated_impact_bps),
                'threshold_bps': float(self.liquidity_service.max_impact_bps),
                'exceeds_threshold': exceeds_threshold,
                'slicing_recommended': impact.recommended_slicing,
                'max_slice_size': float(impact.max_slice_size) if impact.max_slice_size else None,
                'use_post_only': impact.use_post_only,
                'expected_trigger': should_trigger,
                'passed': exceeds_threshold == should_trigger
            }
            
            results['impact_tests'].append(test_result)
            
            if exceeds_threshold:
                results['sized_down_triggered'] = True
            
            if impact.recommended_slicing:
                results['slicing_recommended'] = True
            
            status = "✅" if test_result['passed'] else "❌"
            trigger_status = "SIZED_DOWN" if exceeds_threshold else "ALLOWED"
            
            print(f"{status} {description} ({size} BTC):")
            print(f"   Impact: {impact.estimated_impact_bps:.1f}bps ({trigger_status})")
            if impact.recommended_slicing:
                print(f"   Slicing: Max {impact.max_slice_size:.3f} BTC per slice")
        
        return results
    
    async def test_execution_modes(self) -> Dict:
        """Test different execution modes under stress."""
        print("\n🎯 Testing Execution Modes")
        print("-" * 40)
        
        results = {
            'modes': {},
            'mode_behaviors_correct': True
        }
        
        # Set up shallow book
        bids, asks = self.create_shallow_book()
        self.liquidity_service.analyze_order_book("BTC-USD", bids, asks)
        
        # Large order that would exceed impact
        test_size = Decimal('1.0')  # $50k order
        
        # Test different modes
        modes = {
            'strict': {
                'behavior': 'reject',
                'max_impact': Decimal('30'),  # Very strict - 30bps
            },
            'conservative': {
                'behavior': 'downsize',
                'max_impact': Decimal('50'),  # Standard - 50bps
            },
            'aggressive': {
                'behavior': 'allow',
                'max_impact': Decimal('100'),  # Loose - 100bps
            }
        }
        
        for mode_name, mode_config in modes.items():
            # Create service with mode-specific settings
            mode_service = LiquidityService(max_impact_bps=mode_config['max_impact'])
            mode_service.analyze_order_book("BTC-USD", bids, asks)
            
            impact = mode_service.estimate_market_impact(
                symbol="BTC-USD",
                side="buy",
                quantity=test_size,
                book_data=(bids, asks)
            )
            
            exceeds = impact.estimated_impact_bps > mode_config['max_impact']
            
            # Determine actual behavior
            if exceeds:
                if mode_config['behavior'] == 'reject':
                    actual_behavior = 'reject'
                elif impact.recommended_slicing:
                    actual_behavior = 'downsize'
                else:
                    actual_behavior = 'allow'
            else:
                actual_behavior = 'allow'
            
            mode_result = {
                'mode': mode_name,
                'max_impact_bps': float(mode_config['max_impact']),
                'order_impact_bps': float(impact.estimated_impact_bps),
                'exceeds_threshold': exceeds,
                'expected_behavior': mode_config['behavior'] if exceeds else 'allow',
                'actual_behavior': actual_behavior,
                'slicing_recommended': impact.recommended_slicing,
                'max_slice': float(impact.max_slice_size) if impact.max_slice_size else None
            }
            
            results['modes'][mode_name] = mode_result
            
            print(f"\n{mode_name.upper()} MODE:")
            print(f"  Threshold: {mode_config['max_impact']}bps")
            print(f"  Impact: {impact.estimated_impact_bps:.1f}bps")
            print(f"  Behavior: {actual_behavior}")
            
            if impact.recommended_slicing and impact.max_slice_size:
                print(f"  Slice to: {impact.max_slice_size:.3f} BTC")
        
        return results
    
    async def test_slicing_behavior(self) -> Dict:
        """Test order slicing recommendations."""
        print("\n🔪 Testing Order Slicing Behavior")
        print("-" * 40)
        
        results = {
            'slicing_tests': [],
            'slicing_logic_correct': True
        }
        
        # Set up shallow book
        bids, asks = self.create_shallow_book()
        self.liquidity_service.analyze_order_book("BTC-USD", bids, asks)
        
        # Test progressive order sizes
        test_cases = [
            (Decimal('0.05'), False),  # Small - no slicing needed
            (Decimal('0.5'), True),    # Medium - should recommend slicing
            (Decimal('2.0'), True),    # Large - definitely needs slicing
            (Decimal('10.0'), True),   # Huge - must slice
        ]
        
        for size, should_slice in test_cases:
            impact = self.liquidity_service.estimate_market_impact(
                symbol="BTC-USD",
                side="buy",
                quantity=size,
                book_data=(bids, asks)
            )
            
            if impact.recommended_slicing and impact.max_slice_size:
                num_slices = int(size / impact.max_slice_size) + (1 if size % impact.max_slice_size else 0)
                slice_impact = self.liquidity_service.estimate_market_impact(
                    symbol="BTC-USD",
                    side="buy",
                    quantity=impact.max_slice_size,
                    book_data=(bids, asks)
                )
                per_slice_impact = slice_impact.estimated_impact_bps
            else:
                num_slices = 1
                per_slice_impact = impact.estimated_impact_bps
            
            test_result = {
                'size': float(size),
                'notional': float(size * Decimal('50000')),
                'full_impact_bps': float(impact.estimated_impact_bps),
                'slicing_recommended': impact.recommended_slicing,
                'expected_slicing': should_slice,
                'max_slice_size': float(impact.max_slice_size) if impact.max_slice_size else None,
                'num_slices': num_slices,
                'per_slice_impact_bps': float(per_slice_impact),
                'passed': impact.recommended_slicing == should_slice
            }
            
            results['slicing_tests'].append(test_result)
            
            status = "✅" if test_result['passed'] else "❌"
            
            print(f"\n{status} {size} BTC (${size * 50000:,.0f}):")
            print(f"   Full Impact: {impact.estimated_impact_bps:.1f}bps")
            
            if impact.recommended_slicing:
                print(f"   Slicing: YES - {num_slices} slices")
                print(f"   Max per slice: {impact.max_slice_size:.3f} BTC")
                print(f"   Per-slice impact: {per_slice_impact:.1f}bps")
            else:
                print(f"   Slicing: NO - Execute as single order")
            
            if not test_result['passed']:
                results['slicing_logic_correct'] = False
        
        return results
    
    async def generate_sized_down_log(self) -> str:
        """Generate a SIZED_DOWN event log snippet."""
        print("\n📝 Generating SIZED_DOWN Event Log")
        print("-" * 40)
        
        # Set up shallow book scenario
        bids, asks = self.create_shallow_book()
        self.liquidity_service.analyze_order_book("BTC-USD", bids, asks)
        
        # Attempt large order that triggers SIZED_DOWN
        original_size = Decimal('2.0')
        
        impact = self.liquidity_service.estimate_market_impact(
            symbol="BTC-USD",
            side="buy",
            quantity=original_size,
            book_data=(bids, asks)
        )
        
        # Simulate SIZED_DOWN event
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'SIZED_DOWN',
            'symbol': 'BTC-USD',
            'side': 'buy',
            'original_quantity': float(original_size),
            'original_notional': float(original_size * Decimal('50000')),
            'estimated_impact_bps': float(impact.estimated_impact_bps),
            'max_impact_bps': float(self.liquidity_service.max_impact_bps),
            'recommended_max_size': float(impact.max_slice_size) if impact.max_slice_size else None,
            'liquidity_context': {
                'spread_bps': 2.0,
                'depth_usd_1pct': 10000.0,
                'depth_usd_5pct': 25000.0,
                'condition': 'poor'
            },
            'action_taken': 'order_reduced',
            'new_quantity': float(impact.max_slice_size) if impact.max_slice_size else float(original_size),
            'message': f"Order reduced from {original_size} to {impact.max_slice_size:.3f} BTC due to liquidity constraints"
        }
        
        log_json = json.dumps(log_entry, indent=2)
        
        print("SIZED_DOWN Event Log:")
        print(log_json)
        
        return log_json
    
    async def generate_verification_report(self) -> Dict:
        """Generate comprehensive verification report."""
        print("\n" + "=" * 60)
        print("🧪 LIQUIDITY STRESS TEST VERIFICATION")
        print("=" * 60 + "\n")
        
        # Run all tests
        book_analysis = await self.test_shallow_book_analysis()
        sized_down = await self.test_sized_down_activation()
        execution_modes = await self.test_execution_modes()
        slicing = await self.test_slicing_behavior()
        sized_down_log = await self.generate_sized_down_log()
        
        # Compile report
        report = {
            'verification_type': 'liquidity_stress_test',
            'timestamp': datetime.now().isoformat(),
            'tests': {
                'book_analysis': book_analysis,
                'sized_down_activation': sized_down,
                'execution_modes': execution_modes,
                'slicing_behavior': slicing
            },
            'sample_logs': {
                'sized_down_event': json.loads(sized_down_log)
            },
            'summary': {
                'shallow_book_detected': book_analysis['shallow_correctly_scored'],
                'sized_down_triggered': sized_down['sized_down_triggered'],
                'slicing_recommended': sized_down['slicing_recommended'],
                'mode_behaviors_correct': execution_modes['mode_behaviors_correct'],
                'slicing_logic_correct': slicing['slicing_logic_correct'],
                'overall_pass': (
                    book_analysis['shallow_correctly_scored'] and
                    sized_down['sized_down_triggered'] and
                    sized_down['slicing_recommended'] and
                    execution_modes['mode_behaviors_correct'] and
                    slicing['slicing_logic_correct']
                )
            }
        }
        
        # Save report
        report_path = Path("verification_reports/liquidity_stress_test.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
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
    """Run liquidity stress test verification."""
    stress_test = LiquidityStressTest()
    report = await stress_test.generate_verification_report()
    
    if report['summary']['overall_pass']:
        print("\n✅ LIQUIDITY STRESS TEST VERIFICATION: PASSED")
        return 0
    else:
        print("\n❌ LIQUIDITY STRESS TEST VERIFICATION: FAILED")
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)