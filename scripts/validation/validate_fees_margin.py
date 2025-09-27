#!/usr/bin/env python3
"""
Validation script for Fees Engine and Margin State Monitor.

Tests fee calculations, margin requirements, and integration
with mock and real market data.
"""

import asyncio
import sys
from decimal import Decimal
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from bot_v2.features.live_trade.fees_engine import FeesEngine, FeeType, FeeTier
from bot_v2.features.live_trade.margin_monitor import MarginStateMonitor, MarginWindow
from bot_v2.features.live_trade.portfolio_valuation import PortfolioValuationService


class ValidationResults:
    """Tracks validation test results."""
    
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def add_test(self, name: str, passed: bool, message: str = ""):
        """Add test result."""
        status = "PASS" if passed else "FAIL"
        self.tests.append(f"{status}: {name} - {message}")
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {len(self.tests)}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {self.passed/len(self.tests)*100:.1f}%")
        print(f"{'='*60}")
        
        for test in self.tests:
            print(test)


async def validate_fees_engine():
    """Validate FeesEngine functionality."""
    print("🧪 Validating FeesEngine...")
    results = ValidationResults()
    
    # Create fees engine
    fees_engine = FeesEngine(client=None)
    
    # Test 1: Fee tier determination
    tier_resolver = fees_engine.tier_resolver
    default_tier = tier_resolver.DEFAULT_TIERS[0]
    
    results.add_test(
        "Default tier loading",
        default_tier.tier_name == "Tier 1",
        f"Got tier: {default_tier.tier_name}"
    )
    
    # Test 2: Fee calculation - maker order
    notional = Decimal('10000')  # $10k notional
    fee_calc = await fees_engine.calculate_order_fee(
        symbol="BTC-USD",
        notional=notional,
        is_post_only=True,
        is_reduce_only=False
    )
    
    expected_maker_fee = notional * default_tier.maker_rate
    fee_correct = abs(fee_calc.fee_amount - expected_maker_fee) < Decimal('0.01')
    
    results.add_test(
        "Maker fee calculation",
        fee_correct and fee_calc.fee_type == FeeType.MAKER,
        f"Expected: {expected_maker_fee}, Got: {fee_calc.fee_amount}"
    )
    
    # Test 3: Fee calculation - taker order
    fee_calc_taker = await fees_engine.calculate_order_fee(
        symbol="BTC-USD",
        notional=notional,
        is_post_only=False,
        is_reduce_only=False
    )
    
    expected_taker_fee = notional * default_tier.taker_rate
    taker_fee_correct = abs(fee_calc_taker.fee_amount - expected_taker_fee) < Decimal('0.01')
    
    results.add_test(
        "Taker fee calculation",
        taker_fee_correct and fee_calc_taker.fee_type == FeeType.TAKER,
        f"Expected: {expected_taker_fee}, Got: {fee_calc_taker.fee_amount}"
    )
    
    # Test 4: Reduce-only fee discount
    fee_calc_reduce = await fees_engine.calculate_order_fee(
        symbol="BTC-USD",
        notional=notional,
        is_post_only=False,
        is_reduce_only=True
    )
    
    expected_reduced_fee = notional * default_tier.taker_rate * Decimal('0.8')
    reduced_fee_correct = abs(fee_calc_reduce.fee_amount - expected_reduced_fee) < Decimal('0.01')
    
    results.add_test(
        "Reduce-only fee discount",
        reduced_fee_correct,
        f"Expected: {expected_reduced_fee}, Got: {fee_calc_reduce.fee_amount}"
    )
    
    # Test 5: Trade cost estimation
    trade_cost = await fees_engine.estimate_trade_cost(
        symbol="BTC-USD",
        side="buy",
        quantity=Decimal('1'),
        price=Decimal('50000'),
        is_post_only=True
    )
    
    expected_notional = Decimal('50000')
    expected_fee = expected_notional * default_tier.maker_rate
    expected_total = expected_notional + expected_fee
    
    cost_correct = abs(Decimal(str(trade_cost['total_cost'])) - expected_total) < Decimal('1')
    
    results.add_test(
        "Trade cost estimation",
        cost_correct,
        f"Expected total: {expected_total}, Got: {trade_cost['total_cost']}"
    )
    
    # Test 6: Profitability check
    entry_price = Decimal('50000')
    exit_price = Decimal('50100')  # $100 profit
    total_fee_rate = default_tier.taker_rate * 2  # Entry + exit taker fees
    
    is_profitable = fees_engine.is_trade_profitable(
        entry_price=entry_price,
        exit_price=exit_price,
        side="long",
        fee_rate=total_fee_rate
    )
    
    # Calculate expected profitability
    gross_return = (exit_price - entry_price) / entry_price  # 0.2%
    net_return = gross_return - total_fee_rate  # 0.2% - 1.6% = -1.4%
    expected_profitable = net_return > 0
    
    results.add_test(
        "Profitability calculation",
        is_profitable == expected_profitable,
        f"Gross return: {gross_return:.4f}, Fee rate: {total_fee_rate:.4f}, Net: {net_return:.4f}"
    )
    
    # Test 7: Minimum profit target
    min_exit = await fees_engine.get_minimum_profit_target(
        entry_price=entry_price,
        side="long",
        symbol="BTC-USD"
    )
    
    # Should be entry price + total fees + safety margin
    current_tier = await fees_engine.tier_resolver.get_current_tier()
    total_rate = current_tier.taker_rate * 2 + Decimal('0.001')  # safety margin
    expected_min_exit = entry_price * (1 + total_rate)
    
    min_exit_correct = abs(Decimal(str(min_exit)) - expected_min_exit) < Decimal('10')
    
    results.add_test(
        "Minimum profit target",
        min_exit_correct,
        f"Expected: {expected_min_exit}, Got: {min_exit}"
    )
    
    return results


async def validate_margin_monitor():
    """Validate MarginStateMonitor functionality."""
    print("🧪 Validating MarginStateMonitor...")
    results = ValidationResults()
    
    # Create margin monitor
    margin_monitor = MarginStateMonitor(client=None)
    
    # Test 1: Window determination
    # Test normal window (10:00 UTC)
    test_time = datetime(2024, 1, 1, 10, 0, 0)
    window = margin_monitor.policy.determine_current_window(test_time)
    
    results.add_test(
        "Normal window detection",
        window == MarginWindow.NORMAL,
        f"Expected: NORMAL, Got: {window.value}"
    )
    
    # Test overnight window (23:00 UTC)
    test_time = datetime(2024, 1, 1, 23, 0, 0)
    window = margin_monitor.policy.determine_current_window(test_time)
    
    results.add_test(
        "Overnight window detection",
        window == MarginWindow.OVERNIGHT,
        f"Expected: OVERNIGHT, Got: {window.value}"
    )
    
    # Test pre-funding window (7:45 UTC, 15 min before 8:00 funding)
    test_time = datetime(2024, 1, 1, 7, 45, 0)
    window = margin_monitor.policy.determine_current_window(test_time)
    
    results.add_test(
        "Pre-funding window detection",
        window == MarginWindow.PRE_FUNDING,
        f"Expected: PRE_FUNDING, Got: {window.value}"
    )
    
    # Test 2: Margin requirements by window
    normal_req = margin_monitor.policy.get_requirements(MarginWindow.NORMAL)
    overnight_req = margin_monitor.policy.get_requirements(MarginWindow.OVERNIGHT)
    
    results.add_test(
        "Normal margin requirements",
        normal_req.initial_rate == Decimal('0.10') and normal_req.max_leverage == Decimal('10'),
        f"Initial: {normal_req.initial_rate}, Leverage: {normal_req.max_leverage}"
    )
    
    results.add_test(
        "Overnight margin tighter than normal",
        overnight_req.initial_rate > normal_req.initial_rate,
        f"Normal: {normal_req.initial_rate}, Overnight: {overnight_req.initial_rate}"
    )
    
    # Test 3: Margin state calculation
    total_equity = Decimal('100000')  # $100k
    cash_balance = Decimal('50000')   # $50k cash
    positions = {
        'BTC-USD': {
            'quantity': Decimal('2'),      # 2 BTC
            'mark_price': Decimal('50000') # $50k per BTC = $100k notional
        }
    }
    
    snapshot = await margin_monitor.compute_margin_state(
        total_equity=total_equity,
        cash_balance=cash_balance,
        positions=positions
    )
    
    # Validate calculations
    expected_positions_notional = Decimal('100000')  # 2 * $50k
    expected_leverage = expected_positions_notional / total_equity  # 1.0x
    
    results.add_test(
        "Position notional calculation",
        abs(snapshot.positions_notional - expected_positions_notional) < Decimal('1'),
        f"Expected: {expected_positions_notional}, Got: {snapshot.positions_notional}"
    )
    
    results.add_test(
        "Leverage calculation",
        abs(snapshot.leverage - expected_leverage) < Decimal('0.01'),
        f"Expected: {expected_leverage}, Got: {snapshot.leverage}"
    )
    
    # Test 4: Maximum position size calculation
    max_size = await margin_monitor.get_max_position_size(
        symbol="BTC-USD",
        price=Decimal('50000'),
        available_equity=Decimal('50000')  # $50k available
    )
    
    # With 10% initial margin, max notional = $50k / 0.1 = $500k
    # Max quantity = $500k / $50k = 10 BTC
    expected_max_size = Decimal('10')
    
    results.add_test(
        "Maximum position size calculation",
        abs(max_size - expected_max_size) < Decimal('0.1'),
        f"Expected: {expected_max_size}, Got: {max_size}"
    )
    
    # Test 5: Margin call detection
    # Simulate low equity scenario
    low_equity = Decimal('6000')  # Only $6k equity
    positions_high_risk = {
        'BTC-USD': {
            'quantity': Decimal('2'),
            'mark_price': Decimal('50000')  # Still $100k notional
        }
    }
    
    risky_snapshot = await margin_monitor.compute_margin_state(
        total_equity=low_equity,
        cash_balance=low_equity,
        positions=positions_high_risk
    )
    
    # With $100k notional and 5% maintenance, need $5k minimum
    # With only $6k equity, should be close to margin call
    results.add_test(
        "Margin call detection",
        risky_snapshot.is_liquidation_risk or risky_snapshot.is_margin_call,
        f"Equity: {low_equity}, Margin used: {risky_snapshot.margin_used}, Is margin call: {risky_snapshot.is_margin_call}"
    )
    
    return results


async def validate_integration():
    """Validate integration between components."""
    print("🧪 Validating component integration...")
    results = ValidationResults()
    
    # Create components
    fees_engine = FeesEngine(client=None)
    margin_monitor = MarginStateMonitor(client=None)
    portfolio_service = PortfolioValuationService()
    
    # Test 1: Fee-aware position sizing
    # Check if trade costs reduce available margin
    entry_price = Decimal('50000')
    quantity = Decimal('1')
    
    trade_cost = await fees_engine.estimate_trade_cost(
        symbol="BTC-USD",
        side="buy",
        quantity=quantity,
        price=entry_price,
        is_post_only=True
    )
    
    total_cost = Decimal(str(trade_cost['total_cost']))
    available_equity = Decimal('100000')
    
    # Calculate max size considering fees
    cost_adjusted_equity = available_equity - total_cost
    max_size = await margin_monitor.get_max_position_size(
        symbol="BTC-USD",
        price=entry_price,
        available_equity=cost_adjusted_equity
    )
    
    # Should be less than without considering fees
    max_size_no_fees = await margin_monitor.get_max_position_size(
        symbol="BTC-USD",
        price=entry_price,
        available_equity=available_equity
    )
    
    results.add_test(
        "Fee-adjusted position sizing",
        max_size < max_size_no_fees,
        f"With fees: {max_size}, Without fees: {max_size_no_fees}"
    )
    
    # Test 2: Portfolio valuation with fees
    # Simulate trade and check if fees reduce equity
    portfolio_service.update_trade(
        symbol="BTC-USD",
        side="buy",
        quantity=quantity,
        price=entry_price,
        fees=Decimal(str(trade_cost['fee_amount']))
    )
    
    # Update mark price for unrealized PnL
    portfolio_service.update_mark_prices({
        "BTC-USD": entry_price  # Same price, so no unrealized PnL
    })
    
    # Add mock account data
    from bot_v2.features.brokerages.core.interfaces import Balance, Position
    
    balances = [Balance(currency="USD", total=available_equity, available=available_equity)]
    positions = [Position(symbol="BTC-USD", quantity=quantity, avg_price=entry_price)]
    
    portfolio_service.update_account_data(balances, positions)
    
    valuation = portfolio_service.compute_current_valuation()
    
    # Total equity should reflect fee impact
    results.add_test(
        "Portfolio valuation includes fees",
        valuation.total_equity_usd < available_equity,
        f"Initial: {available_equity}, After fees: {valuation.total_equity_usd}"
    )
    
    return results


async def main():
    """Run all validation tests."""
    print("🚀 Starting Production Readiness Validation")
    print("=" * 60)
    
    # Run individual component validations
    fees_results = await validate_fees_engine()
    margin_results = await validate_margin_monitor()
    integration_results = await validate_integration()
    
    # Combine results
    all_results = ValidationResults()
    all_results.tests.extend(fees_results.tests)
    all_results.tests.extend(margin_results.tests)
    all_results.tests.extend(integration_results.tests)
    all_results.passed = fees_results.passed + margin_results.passed + integration_results.passed
    all_results.failed = fees_results.failed + margin_results.failed + integration_results.failed
    
    # Print summary
    all_results.print_summary()
    
    # Exit with appropriate code
    success_rate = all_results.passed / len(all_results.tests) if all_results.tests else 0
    if success_rate >= 0.9:  # 90% pass rate
        print(f"\n✅ Validation PASSED - {success_rate:.1%} success rate")
        return 0
    else:
        print(f"\n❌ Validation FAILED - {success_rate:.1%} success rate")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)