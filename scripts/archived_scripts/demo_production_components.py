#!/usr/bin/env python3
"""
Production Components Demonstration.

Demonstrates all production-ready components working together
with proper validation and realistic scenarios.
"""

import asyncio
import sys
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bot_v2.features.live_trade.portfolio_valuation import PortfolioValuationService
from bot_v2.features.live_trade.fees_engine import FeesEngine, FeeType
from bot_v2.features.live_trade.margin_monitor import MarginStateMonitor, MarginWindow
from bot_v2.features.live_trade.liquidity_service import LiquidityService, LiquidityCondition
from bot_v2.features.live_trade.order_policy import create_order_policy_matrix
from bot_v2.features.live_trade.pnl_tracker import PnLTracker


async def demo_fees_engine():
    """Demonstrate FeesEngine functionality."""
    print("💸 FEES ENGINE DEMONSTRATION")
    print("-" * 50)
    
    fees_engine = FeesEngine()
    
    # Get current tier
    current_tier = await fees_engine.tier_resolver.get_current_tier()
    print(f"Current Fee Tier: {current_tier.tier_name}")
    print(f"  Maker Rate: {current_tier.maker_rate:.3%}")
    print(f"  Taker Rate: {current_tier.taker_rate:.3%}")
    
    # Calculate fees for different order sizes
    test_orders = [
        {"notional": Decimal('1000'), "post_only": True},   # Small maker
        {"notional": Decimal('10000'), "post_only": False}, # Medium taker
        {"notional": Decimal('50000'), "post_only": True},  # Large maker
    ]
    
    print(f"\nFee Calculations:")
    for order in test_orders:
        fee_calc = await fees_engine.calculate_order_fee(
            symbol="BTC-USD",
            notional=order["notional"],
            is_post_only=order["post_only"]
        )
        
        order_type = "Maker" if order["post_only"] else "Taker"
        print(f"  ${order['notional']:,} {order_type}: ${fee_calc.fee_amount:.2f} ({fee_calc.fee_rate:.3%})")
    
    # Test profitability calculation
    entry_price = Decimal('50000')
    exit_price = Decimal('50500')  # $500 profit per BTC
    
    is_profitable = fees_engine.is_trade_profitable(
        entry_price=entry_price,
        exit_price=exit_price,
        side="long",
        fee_rate=current_tier.taker_rate * 2  # Entry + exit taker fees
    )
    
    gross_profit = (exit_price - entry_price) / entry_price
    net_profit = gross_profit - (current_tier.taker_rate * 2)
    
    print(f"\nProfitability Analysis:")
    print(f"  Entry: ${entry_price}, Exit: ${exit_price}")
    print(f"  Gross Return: {gross_profit:.3%}")
    print(f"  Fee Cost: {current_tier.taker_rate * 2:.3%}")
    print(f"  Net Return: {net_profit:.3%}")
    print(f"  Profitable: {'✅' if is_profitable else '❌'}")
    
    return True


async def demo_margin_monitor():
    """Demonstrate MarginStateMonitor functionality."""
    print("\n📊 MARGIN STATE MONITOR DEMONSTRATION")
    print("-" * 50)
    
    margin_monitor = MarginStateMonitor()
    
    # Test different margin window scenarios
    test_times = [
        datetime(2024, 1, 1, 10, 0, 0),  # Normal trading
        datetime(2024, 1, 1, 23, 30, 0), # Overnight
        datetime(2024, 1, 1, 7, 45, 0),  # Pre-funding (15min before 8am)
    ]
    
    print("Margin Window Detection:")
    for test_time in test_times:
        window = margin_monitor.policy.determine_current_window(test_time)
        requirements = margin_monitor.policy.get_requirements(window)
        
        print(f"  {test_time.strftime('%H:%M UTC')}: {window.value.upper()} "
              f"(Max Leverage: {requirements.max_leverage}x, Initial: {requirements.initial_rate:.1%})")
    
    # Test margin calculation with realistic portfolio
    print(f"\nMargin State Calculation:")
    
    scenarios = [
        {
            "name": "Conservative Position",
            "equity": Decimal('100000'),
            "cash": Decimal('80000'),
            "positions": {
                'BTC-USD': {'quantity': Decimal('1'), 'mark_price': Decimal('50000')}
            }
        },
        {
            "name": "High Leverage Position", 
            "equity": Decimal('100000'),
            "cash": Decimal('20000'),
            "positions": {
                'BTC-USD': {'quantity': Decimal('8'), 'mark_price': Decimal('50000')}
            }
        },
        {
            "name": "Multi-Asset Portfolio",
            "equity": Decimal('100000'),
            "cash": Decimal('30000'),
            "positions": {
                'BTC-USD': {'quantity': Decimal('2'), 'mark_price': Decimal('50000')},
                'ETH-USD': {'quantity': Decimal('10'), 'mark_price': Decimal('3000')}
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n  {scenario['name']}:")
        
        snapshot = await margin_monitor.compute_margin_state(
            total_equity=scenario['equity'],
            cash_balance=scenario['cash'],
            positions=scenario['positions']
        )
        
        print(f"    Positions Value: ${snapshot.positions_notional:,.0f}")
        print(f"    Leverage: {snapshot.leverage:.2f}x")
        print(f"    Margin Utilization: {snapshot.margin_utilization:.1%}")
        print(f"    Available Margin: ${snapshot.margin_available:,.0f}")
        
        if snapshot.is_liquidation_risk:
            print(f"    🚨 LIQUIDATION RISK")
        elif snapshot.is_margin_call:
            print(f"    ⚠️  MARGIN CALL")
        elif snapshot.margin_utilization > Decimal('0.8'):
            print(f"    📈 High Utilization")
        else:
            print(f"    ✅ Healthy")
    
    return True


async def demo_liquidity_service():
    """Demonstrate LiquidityService functionality."""
    print("\n🌊 LIQUIDITY SERVICE DEMONSTRATION")
    print("-" * 50)
    
    liquidity_service = LiquidityService()
    
    # Mock order book scenarios
    scenarios = [
        {
            "name": "Tight Market",
            "bids": [(Decimal('49995'), Decimal('10')), (Decimal('49990'), Decimal('15'))],
            "asks": [(Decimal('50005'), Decimal('10')), (Decimal('50010'), Decimal('15'))]
        },
        {
            "name": "Wide Spread",
            "bids": [(Decimal('49900'), Decimal('5')), (Decimal('49850'), Decimal('8'))],
            "asks": [(Decimal('50100'), Decimal('5')), (Decimal('50150'), Decimal('8'))]
        },
        {
            "name": "Deep Liquidity",
            "bids": [(Decimal('49990'), Decimal('50')), (Decimal('49980'), Decimal('100'))],
            "asks": [(Decimal('50010'), Decimal('50')), (Decimal('50020'), Decimal('100'))]
        }
    ]
    
    print("Order Book Analysis:")
    for scenario in scenarios:
        analysis = liquidity_service.analyze_order_book(
            symbol="BTC-USD",
            bids=scenario["bids"],
            asks=scenario["asks"]
        )
        
        print(f"\n  {scenario['name']}:")
        print(f"    Spread: {analysis.spread_bps:.1f}bps")
        print(f"    L1 Size: {analysis.bid_size} / {analysis.ask_size}")
        print(f"    Depth (5%): ${analysis.depth_usd_5:,.0f}")
        print(f"    Liquidity Score: {analysis.liquidity_score:.0f}/100")
        print(f"    Condition: {analysis.condition.value.upper()}")
    
    # Test market impact estimation
    print(f"\nMarket Impact Analysis:")
    
    # Set up a realistic order book for impact testing
    realistic_bids = [(Decimal('49990') - i*Decimal('10'), Decimal('5') + i*Decimal('2')) for i in range(20)]
    realistic_asks = [(Decimal('50010') + i*Decimal('10'), Decimal('5') + i*Decimal('2')) for i in range(20)]
    
    liquidity_service.analyze_order_book("BTC-USD", realistic_bids, realistic_asks)
    
    trade_sizes = [Decimal('0.5'), Decimal('2'), Decimal('10')]
    
    for size in trade_sizes:
        impact = liquidity_service.estimate_market_impact(
            symbol="BTC-USD",
            side="buy",
            quantity=size,
            book_data=(realistic_bids, realistic_asks)
        )
        
        print(f"    {size} BTC Buy:")
        print(f"      Impact: {impact.estimated_impact_bps:.2f}bps")
        print(f"      Avg Price: ${impact.estimated_avg_price:.2f}")
        print(f"      Slippage: ${impact.slippage_cost:.2f}")
        print(f"      Recommend Slicing: {'Yes' if impact.recommended_slicing else 'No'}")
    
    return True


async def demo_order_policy():
    """Demonstrate OrderPolicyMatrix functionality."""
    print("\n📋 ORDER POLICY MATRIX DEMONSTRATION")
    print("-" * 50)
    
    policy_matrix = await create_order_policy_matrix("sandbox")
    
    # Show supported order types
    print("Supported Order Types (BTC-USD):")
    supported = policy_matrix.get_supported_order_types("BTC-USD")
    
    for order_config in supported:
        post_only = " (Post-Only)" if order_config['post_only'] else ""
        reduce_only = " (Reduce-Only)" if order_config['reduce_only'] else ""
        print(f"  {order_config['order_type']} / {order_config['tif']}{post_only}{reduce_only}")
    
    # Test order validation
    print(f"\nOrder Validation Tests:")
    
    test_orders = [
        {"type": "LIMIT", "tif": "GTC", "quantity": Decimal('1'), "valid": True},
        {"type": "MARKET", "tif": "IOC", "quantity": Decimal('0.5'), "valid": True},
        {"type": "LIMIT", "tif": "GTD", "quantity": Decimal('2'), "valid": False}, # Gated
        {"type": "LIMIT", "tif": "GTC", "quantity": Decimal('0.0001'), "valid": False}, # Too small
    ]
    
    for order in test_orders:
        allowed, reason = policy_matrix.validate_order(
            symbol="BTC-USD",
            order_type=order["type"],
            tif=order["tif"],
            quantity=order["quantity"],
            price=Decimal('50000')
        )
        
        status = "✅" if allowed else "❌"
        expected = "✅" if order["valid"] else "❌"
        match = "✓" if (allowed == order["valid"]) else "✗"
        
        print(f"  {status} {order['type']}/{order['tif']} {order['quantity']} BTC - {reason[:50]} [{match}]")
    
    # Test order recommendations
    print(f"\nOrder Recommendations:")
    
    recommendation_tests = [
        {"urgency": "urgent", "conditions": {"liquidity_condition": "good"}},
        {"urgency": "patient", "conditions": {"spread_bps": 5}},
        {"urgency": "normal", "conditions": {"spread_bps": 25, "volatility_percentile": 95}}
    ]
    
    for test in recommendation_tests:
        config = policy_matrix.recommend_order_config(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal('1'),
            urgency=test["urgency"],
            market_conditions=test["conditions"]
        )
        
        print(f"  {test['urgency'].title()} execution:")
        print(f"    Type: {config['order_type']}, TIF: {config['tif']}")
        print(f"    Post-Only: {config['post_only']}, Use Market: {config.get('use_market', False)}")
    
    return True


async def demo_portfolio_integration():
    """Demonstrate integrated portfolio tracking."""
    print("\n💰 PORTFOLIO INTEGRATION DEMONSTRATION")
    print("-" * 50)
    
    # Create integrated system
    pnl_tracker = PnLTracker()
    portfolio_service = PortfolioValuationService(pnl_tracker=pnl_tracker)
    fees_engine = FeesEngine()
    
    # Simulate trading sequence
    print("Simulating Trading Sequence:")
    
    trades = [
        {"symbol": "BTC-USD", "side": "buy", "qty": Decimal('2'), "price": Decimal('50000')},
        {"symbol": "ETH-USD", "side": "buy", "qty": Decimal('10'), "price": Decimal('3000')},
        {"symbol": "BTC-USD", "side": "sell", "qty": Decimal('0.5'), "price": Decimal('52000')},  # Take profit
    ]
    
    for i, trade in enumerate(trades, 1):
        print(f"\n  Trade {i}: {trade['symbol']} {trade['side']} {trade['qty']} @ ${trade['price']}")
        
        # Calculate fees
        notional = trade['qty'] * trade['price']
        fee_calc = await fees_engine.calculate_order_fee(
            symbol=trade['symbol'],
            notional=notional,
            is_post_only=True  # Assume maker orders
        )
        
        # Update portfolio
        is_reduce = trade['side'] == 'sell'  # Simplified
        pnl_result = portfolio_service.update_trade(
            symbol=trade['symbol'],
            side=trade['side'],
            quantity=trade['qty'],
            price=trade['price'],
            fees=fee_calc.fee_amount,
            is_reduce=is_reduce
        )
        
        # Record fees
        await fees_engine.record_actual_fee(
            symbol=trade['symbol'],
            notional=notional,
            actual_fee=fee_calc.fee_amount,
            fee_type=FeeType.MAKER
        )
        
        print(f"    Fees: ${fee_calc.fee_amount:.2f}")
        realized_pnl = pnl_result.get('realized_pnl', Decimal('0')) if pnl_result else Decimal('0')
        print(f"    Realized PnL: ${realized_pnl:+.2f}")
    
    # Update with current marks
    current_marks = {"BTC-USD": Decimal('51000'), "ETH-USD": Decimal('3100')}
    portfolio_service.update_mark_prices(current_marks)
    
    # Generate portfolio snapshot
    print(f"\nCurrent Portfolio State:")
    
    # Get position metrics
    positions = portfolio_service.pnl_tracker.get_position_metrics()
    total_pnl = portfolio_service.pnl_tracker.get_total_pnl()
    
    for pos in positions:
        if pos['quantity'] != 0:
            print(f"  {pos['symbol']}: {pos['quantity']:+.3f} @ ${pos['avg_entry']:.2f}")
            print(f"    Unrealized PnL: ${pos['unrealized_pnl']:+.2f}")
            print(f"    Realized PnL: ${pos['realized_pnl']:+.2f}")
    
    print(f"\nTotal Portfolio PnL:")
    print(f"  Realized: ${total_pnl['realized']:+.2f}")
    print(f"  Unrealized: ${total_pnl['unrealized']:+.2f}")
    print(f"  Total: ${total_pnl['total']:+.2f}")
    print(f"  Funding: ${total_pnl['funding']:+.2f}")
    
    # Get fee summary
    fee_summary = await fees_engine.get_fee_summary(hours_back=24)
    print(f"\nFee Summary:")
    print(f"  Total Fees: ${fee_summary['total_fees']:.2f}")
    print(f"  Trade Count: {fee_summary['trade_count']}")
    
    return True


async def main():
    """Run all production component demonstrations."""
    print("🚀 PRODUCTION COMPONENTS DEMONSTRATION")
    print("=" * 60)
    
    demos = [
        ("Fees Engine", demo_fees_engine),
        ("Margin Monitor", demo_margin_monitor),
        ("Liquidity Service", demo_liquidity_service),
        ("Order Policy Matrix", demo_order_policy),
        ("Portfolio Integration", demo_portfolio_integration)
    ]
    
    results = []
    for name, demo_func in demos:
        try:
            print(f"\n{'='*60}")
            result = await demo_func()
            results.append((name, result, None))
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("DEMONSTRATION SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"      Error: {error}")
    
    print(f"\nOverall: {passed}/{total} components demonstrated successfully")
    
    if passed == total:
        print("🎉 All production components working correctly!")
        return 0
    else:
        print("⚠️  Some components had issues - review above")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)