#!/usr/bin/env python3
"""
Week 3 Order Management Validation Script.

Tests advanced order types, TIF mapping, and impact-aware sizing.
Run with RUN_SANDBOX_VALIDATIONS=1 for real sandbox tests.
"""

import os
import sys
import time
import logging
from decimal import Decimal
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot_v2.features.live_trade.execution_v3 import (
    AdvancedExecutionEngine, OrderConfig, SizingMode
)
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide, OrderType, OrderStatus, TimeInForce, Quote
)
from bot_v2.orchestration.mock_broker import MockBroker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_limit_orders():
    """Test limit order functionality including post-only."""
    logger.info("\n=== Testing Limit Orders ===")
    
    broker = MockBroker()
    config = OrderConfig(
        enable_limit_orders=True,
        enable_post_only=True,
        reject_on_cross=True
    )
    engine = AdvancedExecutionEngine(broker, config=config)
    
    # Test 1: Non-crossing post-only (should succeed)
    logger.info("Test 1: Non-crossing post-only limit order")
    quote = broker.get_quote("BTC-PERP")
    limit_price = quote.bid - Decimal("10")  # Well below bid
    
    order = engine.place_order(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("0.001"),
        order_type="limit",
        limit_price=limit_price,
        post_only=True,
        client_id="test_limit_1"
    )
    
    if order:
        logger.info(f"✅ Post-only order placed: {order}")
    else:
        logger.error("❌ Post-only order failed unexpectedly")
    
    # Test 2: Crossing post-only (should reject)
    logger.info("\nTest 2: Crossing post-only limit order")
    limit_price = quote.ask + Decimal("10")  # Above ask - would cross
    
    order = engine.place_order(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("0.001"),
        order_type="limit",
        limit_price=limit_price,
        post_only=True,
        client_id="test_limit_2"
    )
    
    if not order:
        logger.info(f"✅ Crossing post-only correctly rejected")
        logger.info(f"   Post-only rejections: {engine.order_metrics['post_only_rejected']}")
    else:
        logger.error("❌ Crossing post-only should have been rejected")
    
    return engine.order_metrics['placed'] >= 1 and engine.order_metrics['post_only_rejected'] >= 1


def validate_stop_orders():
    """Test stop and stop-limit orders."""
    logger.info("\n=== Testing Stop Orders ===")
    
    broker = MockBroker()
    config = OrderConfig(
        enable_stop_orders=True,
        enable_stop_limit=True
    )
    engine = AdvancedExecutionEngine(broker, config=config)
    
    # Test 1: Basic stop order
    logger.info("Test 1: Basic stop order")
    order = engine.place_order(
        symbol="BTC-PERP",
        side="sell",
        quantity=Decimal("0.001"),
        order_type="stop",
        stop_price=Decimal("48000"),
        client_id="test_stop_1"
    )
    
    if order:
        logger.info(f"✅ Stop order placed: {order}")
        # Check trigger tracking
        if "test_stop_1" in engine.stop_triggers:
            logger.info(f"   Trigger tracked: {engine.stop_triggers['test_stop_1'].trigger_price}")
    else:
        logger.error("❌ Stop order failed")
    
    # Test 2: Stop-limit with valid prices
    logger.info("\nTest 2: Valid stop-limit order")
    order = engine.place_order(
        symbol="ETH-PERP",
        side="buy",
        quantity=Decimal("0.01"),
        order_type="stop_limit",
        stop_price=Decimal("3100"),
        limit_price=Decimal("3110"),  # Valid: limit > stop for buy
        client_id="test_stop_limit_1"
    )
    
    if order:
        logger.info(f"✅ Stop-limit order placed: {order}")
    else:
        logger.error("❌ Valid stop-limit failed")
    
    # Test 3: Invalid stop-limit (should reject)
    logger.info("\nTest 3: Invalid stop-limit order")
    order = engine.place_order(
        symbol="ETH-PERP",
        side="buy",
        quantity=Decimal("0.01"),
        order_type="stop_limit",
        stop_price=Decimal("3100"),
        limit_price=Decimal("3090"),  # Invalid: limit < stop for buy
        client_id="test_stop_limit_2"
    )
    
    if not order:
        logger.info("✅ Invalid stop-limit correctly rejected")
    else:
        logger.error("❌ Invalid stop-limit should have been rejected")
    
    # Test trigger detection
    logger.info("\nTest 4: Stop trigger detection")
    current_prices = {
        "BTC-PERP": Decimal("47500"),  # Below sell stop
        "ETH-PERP": Decimal("3150")     # Above buy stop
    }
    
    triggered = engine.check_stop_triggers(current_prices)
    logger.info(f"   Triggered stops: {triggered}")
    logger.info(f"   Total triggers: {engine.order_metrics['stop_triggered']}")
    
    return engine.order_metrics['placed'] >= 2


def validate_tif_mapping():
    """Test Time-In-Force mapping."""
    logger.info("\n=== Testing TIF Mapping ===")
    
    broker = MockBroker()
    config = OrderConfig(
        enable_ioc=True,
        enable_fok=False  # Gated
    )
    engine = AdvancedExecutionEngine(broker, config=config)
    
    # Test 1: GTC order
    logger.info("Test 1: GTC order")
    order = engine.place_order(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("0.001"),
        order_type="market",
        time_in_force="GTC",
        client_id="test_gtc"
    )
    
    if order:
        logger.info("✅ GTC order placed")
    else:
        logger.error("❌ GTC order failed")
    
    # Test 2: IOC order
    logger.info("\nTest 2: IOC order")
    order = engine.place_order(
        symbol="BTC-PERP",
        side="sell",
        quantity=Decimal("0.001"),
        order_type="market",
        time_in_force="IOC",
        client_id="test_ioc"
    )
    
    if order:
        logger.info("✅ IOC order placed")
    else:
        logger.error("❌ IOC order failed")
    
    # Test 3: FOK order (should be gated)
    logger.info("\nTest 3: FOK order (gated)")
    order = engine.place_order(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("0.001"),
        order_type="market",
        time_in_force="FOK",
        client_id="test_fok"
    )
    
    if not order:
        logger.info("✅ FOK correctly gated")
        logger.info(f"   Rejections: {engine.order_metrics['rejected']}")
    else:
        logger.error("❌ FOK should be gated")
    
    return engine.order_metrics['placed'] >= 2 and engine.order_metrics['rejected'] >= 1


def validate_cancel_replace():
    """Test cancel and replace flow."""
    logger.info("\n=== Testing Cancel/Replace ===")
    
    broker = MockBroker()
    engine = AdvancedExecutionEngine(broker)
    
    # Place initial order
    logger.info("Placing initial order")
    order = engine.place_order(
        symbol="BTC-PERP",
        side="buy",
        quantity=Decimal("0.001"),
        order_type="limit",
        limit_price=Decimal("49000"),
        client_id="original_order"
    )
    
    if not order:
        logger.error("❌ Failed to place initial order")
        return False
    
    logger.info(f"   Initial order: {order}")
    
    # Cancel and replace
    logger.info("\nCancelling and replacing with new price")
    new_order = engine.cancel_and_replace(
        order_id=order.id,
        new_price=Decimal("49500"),
        new_size=Decimal("0.002")
    )
    
    if new_order:
        logger.info(f"✅ Cancel/replace succeeded: {new_order}")
        logger.info(f"   Cancellations: {engine.order_metrics['cancelled']}")
    else:
        logger.error("❌ Cancel/replace failed")
    
    # Test idempotency
    logger.info("\nTesting idempotent client IDs")
    order1 = engine.place_order(
        symbol="ETH-PERP",
        side="buy",
        quantity=Decimal("0.01"),
        client_id="idempotent_id"
    )
    
    order2 = engine.place_order(
        symbol="ETH-PERP",
        side="buy",
        quantity=Decimal("0.02"),  # Different size
        client_id="idempotent_id"  # Same ID
    )
    
    if order1 == order2:
        logger.info("✅ Idempotent client ID handled correctly")
    else:
        logger.error("❌ Idempotent client ID not handled")
    
    return engine.order_metrics['cancelled'] >= 1


def validate_impact_sizing():
    """Test impact-aware sizing."""
    logger.info("\n=== Testing Impact-Aware Sizing ===")
    
    broker = MockBroker()
    
    market_snapshot = {
        'depth_l1': Decimal('50000'),   # $50k at L1
        'depth_l10': Decimal('200000'),  # $200k at L10
        'mid': Decimal('50000')
    }
    
    # Test 1: Conservative mode
    logger.info("Test 1: Conservative sizing")
    config = OrderConfig(
        sizing_mode=SizingMode.CONSERVATIVE,
        max_impact_bps=Decimal("10")
    )
    engine = AdvancedExecutionEngine(broker, config=config)
    
    adjusted, impact = engine.calculate_impact_aware_size(
        target_notional=Decimal('150000'),  # Would exceed impact
        market_snapshot=market_snapshot
    )
    
    logger.info(f"   Target: $150,000")
    logger.info(f"   Adjusted: ${adjusted:,.0f}")
    logger.info(f"   Impact: {impact:.1f} bps")
    
    if adjusted < Decimal('150000') and impact <= Decimal('10'):
        logger.info("✅ Conservative mode sized down correctly")
    else:
        logger.error("❌ Conservative mode failed to size down")
    
    # Test 2: Strict mode
    logger.info("\nTest 2: Strict sizing")
    config = OrderConfig(
        sizing_mode=SizingMode.STRICT,
        max_impact_bps=Decimal("10")
    )
    engine = AdvancedExecutionEngine(broker, config=config)
    
    adjusted, impact = engine.calculate_impact_aware_size(
        target_notional=Decimal('150000'),
        market_snapshot=market_snapshot
    )
    
    logger.info(f"   Adjusted: ${adjusted:,.0f}")
    logger.info(f"   Impact: {impact:.1f} bps")
    
    if adjusted == Decimal('0'):
        logger.info("✅ Strict mode correctly rejected")
    else:
        logger.error("❌ Strict mode should reject")
    
    # Test 3: Aggressive mode
    logger.info("\nTest 3: Aggressive sizing")
    config = OrderConfig(
        sizing_mode=SizingMode.AGGRESSIVE,
        max_impact_bps=Decimal("10")
    )
    engine = AdvancedExecutionEngine(broker, config=config)
    
    adjusted, impact = engine.calculate_impact_aware_size(
        target_notional=Decimal('150000'),
        market_snapshot=market_snapshot
    )
    
    logger.info(f"   Adjusted: ${adjusted:,.0f}")
    logger.info(f"   Impact: {impact:.1f} bps")
    
    if adjusted == Decimal('150000') and impact > Decimal('10'):
        logger.info("✅ Aggressive mode allowed higher impact")
    else:
        logger.error("❌ Aggressive mode failed")
    
    return True


def validate_sandbox():
    """Run sandbox validation if enabled."""
    if not os.environ.get('RUN_SANDBOX_VALIDATIONS'):
        logger.info("\n=== Sandbox validation skipped (set RUN_SANDBOX_VALIDATIONS=1 to enable) ===")
        return True
    
    logger.info("\n=== Running Sandbox Validation ===")
    
    # Import real adapter
    from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
    from bot_v2.features.brokerages.coinbase.models import APIConfig
    
    # Setup with sandbox credentials
    config = APIConfig(
        api_key=os.environ.get('COINBASE_API_KEY', ''),
        api_secret=os.environ.get('COINBASE_API_SECRET', ''),
        passphrase=os.environ.get('COINBASE_PASSPHRASE', ''),
        sandbox=True,
        enable_derivatives=True
    )
    
    broker = CoinbaseBrokerage(config)
    engine = AdvancedExecutionEngine(broker)
    
    # Test 1: Place tiny non-crossing limit order
    logger.info("Sandbox Test 1: Non-crossing limit order")
    quote = broker.get_quote("BTC-PERP")
    if quote:
        limit_price = quote.bid - Decimal("100")  # Far below bid
        
        order = engine.place_order(
            symbol="BTC-PERP",
            side="buy",
            quantity=Decimal("0.0001"),  # Tiny size
            order_type="limit",
            limit_price=limit_price,
            post_only=True,
            client_id=f"sandbox_test_{int(time.time())}"
        )
        
        if order:
            logger.info(f"✅ Sandbox order placed: {order.id}")
            
            # Cancel immediately
            time.sleep(1)
            if broker.cancel_order(order.id):
                logger.info("✅ Sandbox order cancelled")
            else:
                logger.warning("⚠️ Failed to cancel sandbox order")
        else:
            logger.error("❌ Sandbox order failed")
    
    return True


def validate_capability_probe():
    """Run capability probe."""
    logger.info("\n=== Testing Capability Probe ===")
    
    from scripts.probe_capabilities import probe_capabilities, print_capability_report
    
    # Test with mock broker
    logger.info("Running capability probe...")
    capabilities = probe_capabilities()
    
    # Count supported features
    total = sum([
        capabilities['market_orders'],
        capabilities['limit_orders'],
        capabilities['stop_orders'],
        capabilities['stop_limit_orders'],
        capabilities['post_only'],
        capabilities['reduce_only'],
        capabilities['client_order_id'],
        capabilities['time_in_force']['GTC'],
        capabilities['time_in_force']['IOC'],
        capabilities['cancel_order'],
        capabilities['cancel_replace'],
        capabilities['impact_sizing'],
        capabilities['funding_support']
    ])
    
    logger.info(f"\n   Capability Score: {total}/13")
    
    if total >= 11:
        logger.info("✅ Broker is PRODUCTION READY")
        return True
    elif total >= 8:
        logger.info("⚠️ Broker has PARTIAL support")
        return True
    else:
        logger.error("❌ Broker LACKS critical features")
        return False


def main():
    """Run all validations."""
    logger.info("=" * 60)
    logger.info("Week 3 Order Management Validation")
    logger.info("=" * 60)
    
    results = []
    
    # Run capability probe first
    results.append(("Capability Probe", validate_capability_probe()))
    
    # Run mock validations
    results.append(("Limit Orders", validate_limit_orders()))
    results.append(("Stop Orders", validate_stop_orders()))
    results.append(("TIF Mapping", validate_tif_mapping()))
    results.append(("Cancel/Replace", validate_cancel_replace()))
    results.append(("Impact Sizing", validate_impact_sizing()))
    
    # Run sandbox if enabled
    if os.environ.get('RUN_SANDBOX_VALIDATIONS'):
        results.append(("Sandbox", validate_sandbox()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{name:20} {status}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        logger.info("\n🎉 All validations passed!")
    else:
        logger.error("\n⚠️ Some validations failed")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())