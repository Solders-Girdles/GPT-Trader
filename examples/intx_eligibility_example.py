"""
Example: INTX Eligibility Verification with Fail-Closed Logic

Demonstrates how to integrate INTX eligibility checking into your trading bot
with fail-closed behavior to prevent derivatives orders when permissions are missing.

Components demonstrated:
1. IntxEligibilityChecker - Core eligibility verification with caching
2. IntxStartupValidator - Validates on bot startup (fail-closed)
3. IntxPreTradeValidator - Validates before every derivatives order
4. IntxRuntimeMonitor - Periodic re-verification during operation
"""

from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.risk.intx_pre_trade import (
    IntxEligibilityViolation,
    create_intx_validator,
)
from bot_v2.orchestration.intx_eligibility import create_fail_closed_checker
from bot_v2.orchestration.intx_runtime_monitor import create_runtime_monitor
from bot_v2.orchestration.intx_startup_validator import (
    IntxStartupValidationError,
    validate_intx_on_startup,
)
from bot_v2.persistence.event_store import EventStore


def example_startup_validation(intx_service, enable_derivatives=True):
    """
    Example 1: Startup Validation

    Validates INTX eligibility when bot starts. If derivatives enabled but
    INTX not available, bot fails to start (fail-closed).
    """
    print("=" * 80)
    print("EXAMPLE 1: STARTUP VALIDATION")
    print("=" * 80)
    print()

    # Create eligibility checker
    eligibility_checker = create_fail_closed_checker(intx_service)

    # Validate on startup
    try:
        print("Validating INTX eligibility on startup...")
        validate_intx_on_startup(
            eligibility_checker=eligibility_checker,
            enable_derivatives=enable_derivatives,
            fail_closed=True,  # Fail startup if not eligible
        )
        print("✅ Startup validation PASSED - INTX is eligible")
        print()

    except IntxStartupValidationError as e:
        print("❌ Startup validation FAILED")
        print()
        print(str(e))
        print()
        print("Bot will NOT start. Fix INTX eligibility or disable derivatives.")
        return False

    return True


def example_pre_trade_validation(intx_service, event_store, enable_derivatives=True):
    """
    Example 2: Pre-Trade Validation

    Validates INTX eligibility before each derivatives order. Orders rejected
    if not eligible (fail-closed).
    """
    print("=" * 80)
    print("EXAMPLE 2: PRE-TRADE VALIDATION")
    print("=" * 80)
    print()

    # Create eligibility checker
    eligibility_checker = create_fail_closed_checker(intx_service)

    # Create pre-trade validator
    intx_validator = create_intx_validator(
        eligibility_checker=eligibility_checker,
        event_store=event_store,
        enable_derivatives=enable_derivatives,
    )

    # Simulate some orders
    orders = [
        {"symbol": "BTC-PERP", "side": "buy", "quantity": Decimal("0.1"), "market_type": MarketType.PERPETUAL},
        {"symbol": "ETH-PERP", "side": "sell", "quantity": Decimal("1.0"), "market_type": MarketType.PERPETUAL},
        {"symbol": "BTC-USD", "side": "buy", "quantity": Decimal("0.05"), "market_type": MarketType.SPOT},
    ]

    for order in orders:
        product = Product(
            symbol=order["symbol"],
            base_asset=order["symbol"].split("-")[0],
            quote_asset="USD",
            market_type=order["market_type"],
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
        )

        try:
            print(f"Validating order: {order['symbol']} {order['side']} {order['quantity']}")

            intx_validator.validate_intx_eligibility(
                symbol=order["symbol"],
                side=order["side"],
                quantity=order["quantity"],
                product=product,
            )

            print(f"  ✅ Validation PASSED - order allowed")

        except IntxEligibilityViolation as e:
            print(f"  ❌ Validation FAILED - order REJECTED")
            print(f"  Reason: {str(e)}")

        print()

    # Show stats
    stats = intx_validator.get_stats()
    print("Pre-Trade Validation Stats:")
    print(f"  Total Checks: {stats['total_checks']}")
    print(f"  Approvals: {stats['approvals']}")
    print(f"  Rejections: {stats['rejections']}")
    print(f"  Rejection Rate: {stats['rejection_rate']:.1%}")
    print()


def example_runtime_monitoring(intx_service, event_store, enable_derivatives=True):
    """
    Example 3: Runtime Monitoring

    Periodically re-checks INTX eligibility during bot operation to detect
    permission changes or revocations.
    """
    print("=" * 80)
    print("EXAMPLE 3: RUNTIME MONITORING")
    print("=" * 80)
    print()

    # Create eligibility checker
    eligibility_checker = create_fail_closed_checker(intx_service)

    # Create runtime monitor
    monitor = create_runtime_monitor(
        eligibility_checker=eligibility_checker,
        event_store=event_store,
        enable_derivatives=enable_derivatives,
        check_interval_minutes=60,  # Check every hour
    )

    # Simulate trading loop
    print("Simulating trading loop with periodic eligibility checks...")
    print()

    for cycle in range(3):
        print(f"Trading Cycle {cycle + 1}")

        # Check if eligibility check is due
        if monitor.check_if_due():
            print("  Running periodic INTX eligibility check...")
            monitor.run_periodic_check()
        else:
            print("  Eligibility check not due yet")

        # In real usage, you'd do trading here
        print("  (Trading logic would run here)")
        print()

    # Show monitoring status
    status = monitor.get_status_summary()
    print("Runtime Monitor Status:")
    print(f"  Last Check: {status['last_check']}")
    print(f"  Last Status: {status['last_status']}")
    print(f"  Permission Loss Detected: {status['permission_loss_detected']}")
    print(f"  Check Interval: {status['check_interval_minutes']} minutes")
    print()


def example_complete_integration(intx_service, event_store, enable_derivatives=True):
    """
    Example 4: Complete Integration

    Shows how all components work together in a real bot.
    """
    print("=" * 80)
    print("EXAMPLE 4: COMPLETE INTEGRATION")
    print("=" * 80)
    print()

    # Step 1: Create eligibility checker
    print("1. Creating eligibility checker...")
    eligibility_checker = create_fail_closed_checker(intx_service)
    print("   ✅ Eligibility checker created")
    print()

    # Step 2: Validate on startup
    print("2. Validating INTX eligibility on startup...")
    try:
        validate_intx_on_startup(
            eligibility_checker=eligibility_checker,
            enable_derivatives=enable_derivatives,
            fail_closed=True,
        )
        print("   ✅ Startup validation passed")
    except IntxStartupValidationError as e:
        print("   ❌ Startup validation failed - bot will not start")
        return

    print()

    # Step 3: Create pre-trade validator
    print("3. Creating pre-trade validator...")
    intx_validator = create_intx_validator(
        eligibility_checker=eligibility_checker,
        event_store=event_store,
        enable_derivatives=enable_derivatives,
    )
    print("   ✅ Pre-trade validator created")
    print()

    # Step 4: Create runtime monitor
    print("4. Creating runtime monitor...")
    monitor = create_runtime_monitor(
        eligibility_checker=eligibility_checker,
        event_store=event_store,
        enable_derivatives=enable_derivatives,
        check_interval_minutes=60,
    )
    print("   ✅ Runtime monitor created")
    print()

    # Step 5: Simulate trading
    print("5. Simulating trading with all checks enabled...")
    print()

    product = Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
    )

    # Trading loop
    for i in range(2):
        print(f"   Trading Cycle {i + 1}:")

        # Check runtime monitor
        if monitor.check_if_due():
            print("     Running periodic eligibility check...")
            monitor.run_periodic_check()

        # Attempt to place order
        try:
            print(f"     Validating order: BTC-PERP buy 0.1")
            intx_validator.validate_intx_eligibility(
                symbol="BTC-PERP",
                side="buy",
                quantity=Decimal("0.1"),
                product=product,
            )
            print(f"     ✅ Order allowed - would execute here")
        except IntxEligibilityViolation as e:
            print(f"     ❌ Order rejected: {e}")

        print()

    print("6. Complete - all INTX checks working together")
    print()


def main():
    """Run all examples."""
    print()
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║      INTX ELIGIBILITY VERIFICATION EXAMPLES                     ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

    # Note: In real usage, you'd get these from your bot configuration
    # For this example, we'll create mock versions

    # Mock IntxPortfolioService
    class MockIntxService:
        def supports_intx(self):
            return True  # Change to False to test ineligible case

        def get_portfolio_uuid(self, refresh=False):
            return "mock-portfolio-uuid-12345"  # Return None to test ineligible

        def snapshot(self):
            return {
                "supports_intx": True,
                "portfolio_uuid": "mock-portfolio-uuid-12345",
                "override_uuid": None,
            }

    intx_service = MockIntxService()
    event_store = EventStore(storage_dir=".")  # Mock event store

    # Configuration
    enable_derivatives = True  # Set to False to skip all checks

    # Run examples
    print("Running examples with derivatives ENABLED and INTX ELIGIBLE")
    print("(Change MockIntxService to test fail-closed behavior)")
    print()

    # Example 1: Startup Validation
    if not example_startup_validation(intx_service, enable_derivatives):
        print("Startup validation failed, stopping examples.")
        return

    # Example 2: Pre-Trade Validation
    example_pre_trade_validation(intx_service, event_store, enable_derivatives)

    # Example 3: Runtime Monitoring
    example_runtime_monitoring(intx_service, event_store, enable_derivatives)

    # Example 4: Complete Integration
    example_complete_integration(intx_service, event_store, enable_derivatives)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✅ All INTX eligibility examples completed")
    print()
    print("Key Takeaways:")
    print("  1. Startup validation ensures INTX is available before trading")
    print("  2. Pre-trade validation blocks ineligible orders (fail-closed)")
    print("  3. Runtime monitoring detects permission changes")
    print("  4. All components work together for comprehensive protection")
    print()
    print("Integration Checklist:")
    print("  [ ] Add IntxEligibilityChecker to bot initialization")
    print("  [ ] Call validate_intx_on_startup() in startup sequence")
    print("  [ ] Integrate IntxPreTradeValidator into execution flow")
    print("  [ ] Add monitor.run_periodic_check() to main trading loop")
    print()


if __name__ == "__main__":
    main()
