#!/usr/bin/env python3
"""
Test script for Production Orchestrator

This script tests the production orchestrator with fallback strategies
when ML components are not available.

Usage:
    poetry run python scripts/test_production_orchestrator.py
"""

import sys

sys.path.insert(0, "src")

import asyncio
from datetime import datetime, timedelta

import pandas as pd
from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.live.production_orchestrator import (
    OrchestratorConfig,
    ProductionOrchestrator,
    StrategyMode,
)
from bot.logging import get_logger

logger = get_logger(__name__)


async def test_orchestrator_basic():
    """Test basic orchestrator functionality"""
    print("\n" + "=" * 60)
    print("TESTING PRODUCTION ORCHESTRATOR - BASIC FUNCTIONALITY")
    print("=" * 60)

    # Create configuration for fallback mode (since ML components aren't working)
    config = OrchestratorConfig(
        strategy_mode=StrategyMode.FALLBACK_ONLY,
        fallback_strategy="demo_ma",
        confidence_threshold=0.6,
        enable_auto_retraining=False,  # Disable since dependencies missing
    )

    # Create orchestrator
    print("\n1. Creating Production Orchestrator...")
    orchestrator = ProductionOrchestrator(config)

    # Initialize
    print("\n2. Initializing orchestrator...")
    success = await orchestrator.initialize()

    if not success:
        print("❌ Failed to initialize orchestrator")
        return False

    print("✅ Orchestrator initialized successfully")

    # Get health status
    print("\n3. Checking orchestrator health...")
    health = orchestrator.get_health_summary()
    print(f"   State: {health['state']}")
    print(f"   ML Pipeline: {health['ml_pipeline_loaded']}")
    print(f"   Strategy Selector: {health['strategy_selector_loaded']}")
    print(f"   Available Strategies: {health['strategies_available']}")
    print(f"   Fallback Strategy: {health['fallback_strategy']}")

    if health["warnings"]:
        print(f"   Warnings: {len(health['warnings'])} warnings")
        for i, warning in enumerate(health["warnings"][:3], 1):
            print(f"     {i}. {warning}")

    # Get sample market data
    print("\n4. Fetching market data...")
    try:
        data_source = YFinanceSource()
        # Use correct method and date format
        from datetime import datetime, timedelta

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        market_data = data_source.get_daily_bars("AAPL", start_date, end_date)
        print(f"   ✅ Fetched {len(market_data)} bars for AAPL")
        print(f"   Date range: {market_data.index[0].date()} to {market_data.index[-1].date()}")
    except Exception as e:
        print(f"   ❌ Failed to fetch market data: {e}")
        return False

    # Test strategy selection
    print("\n5. Testing strategy selection...")
    try:
        strategy, confidence, metadata = await orchestrator.select_strategy(market_data)
        print(f"   ✅ Selected strategy: {strategy}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Selection method: {metadata['method']}")
        print(f"   Reason: {metadata.get('reason', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Strategy selection failed: {e}")
        return False

    # Test signal generation
    print("\n6. Testing signal generation...")
    try:
        signals = orchestrator.generate_trading_signals(strategy, market_data)
        print(f"   ✅ Generated signals: {len(signals)} periods")

        # Show recent signals
        recent_signals = signals.tail(5)
        if not recent_signals.empty and "signal" in recent_signals.columns:
            signal_counts = recent_signals["signal"].value_counts()
            print(f"   Recent signals: {dict(signal_counts)}")

            # Show signal summary
            total_signals = signals["signal"].value_counts() if "signal" in signals.columns else {}
            print(f"   Total signals: {dict(total_signals)}")
        else:
            print("   No valid signals generated")

    except Exception as e:
        print(f"   ❌ Signal generation failed: {e}")
        return False

    # Test multiple strategy selections
    print("\n7. Testing multiple strategy selections...")
    for i in range(3):
        try:
            strategy, confidence, metadata = await orchestrator.select_strategy(market_data)
            print(f"   Selection {i+1}: {strategy} (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"   ❌ Selection {i+1} failed: {e}")

    # Show performance metrics
    print("\n8. Performance metrics...")
    status = orchestrator.get_status()
    if status.performance_metrics:
        for metric, value in status.performance_metrics.items():
            print(f"   {metric}: {value}")
    else:
        print("   No performance metrics available yet")

    # Shutdown
    print("\n9. Shutting down orchestrator...")
    await orchestrator.shutdown()
    print("   ✅ Shutdown complete")

    return True


async def test_ml_fallback_behavior():
    """Test ML with fallback behavior"""
    print("\n" + "=" * 60)
    print("TESTING ML WITH FALLBACK BEHAVIOR")
    print("=" * 60)

    # Create configuration that tries ML first, falls back if needed
    config = OrchestratorConfig(
        strategy_mode=StrategyMode.ML_WITH_FALLBACK,
        fallback_strategy="trend_breakout",
        confidence_threshold=0.8,  # High threshold to force fallback
        enable_auto_retraining=False,
    )

    orchestrator = ProductionOrchestrator(config)

    print("\n1. Initializing orchestrator in ML+Fallback mode...")
    success = await orchestrator.initialize()

    if not success:
        print("❌ Failed to initialize")
        return False

    print("✅ Orchestrator initialized")

    # Get market data
    print("\n2. Fetching market data...")
    data_source = YFinanceSource()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    market_data = data_source.get_daily_bars("MSFT", start_date, end_date)
    print(f"   ✅ Fetched {len(market_data)} bars for MSFT")

    # Test strategy selection (should fallback due to missing ML)
    print("\n3. Testing strategy selection (should use fallback)...")
    strategy, confidence, metadata = await orchestrator.select_strategy(market_data)
    print(f"   Selected: {strategy}")
    print(f"   Method: {metadata['method']}")
    print(f"   Confidence: {confidence:.3f}")

    if metadata["method"] == "fallback":
        print("   ✅ Correctly used fallback strategy")
    else:
        print("   ⚠️  Expected fallback but got different method")

    # Test signal generation with fallback strategy
    print("\n4. Testing signal generation with fallback...")
    signals = orchestrator.generate_trading_signals(strategy, market_data)
    print(f"   ✅ Generated {len(signals)} signals")

    await orchestrator.shutdown()
    return True


async def test_error_handling():
    """Test error handling and recovery"""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING AND RECOVERY")
    print("=" * 60)

    config = OrchestratorConfig(
        strategy_mode=StrategyMode.FALLBACK_ONLY, fallback_strategy="demo_ma"
    )

    orchestrator = ProductionOrchestrator(config)
    await orchestrator.initialize()

    print("\n1. Testing with invalid market data...")

    # Test with empty DataFrame
    try:
        empty_data = pd.DataFrame()
        strategy, confidence, metadata = await orchestrator.select_strategy(empty_data)
        print(f"   ✅ Handled empty data: {strategy} (method: {metadata['method']})")
    except Exception as e:
        print(f"   ❌ Failed on empty data: {e}")

    # Test with malformed data
    try:
        bad_data = pd.DataFrame({"bad_column": [1, 2, 3]})
        signals = orchestrator.generate_trading_signals("demo_ma", bad_data)
        print(f"   ✅ Handled bad data: {len(signals)} signals")
    except Exception as e:
        print(f"   ⚠️  Error with bad data (expected): {e}")

    # Test with non-existent strategy
    print("\n2. Testing with non-existent strategy...")
    data_source = YFinanceSource()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    market_data = data_source.get_daily_bars("GOOGL", start_date, end_date)

    signals = orchestrator.generate_trading_signals("nonexistent_strategy", market_data)
    print(f"   ✅ Handled nonexistent strategy: {len(signals)} signals")

    await orchestrator.shutdown()
    return True


async def main():
    """Run all tests"""
    print("Production Orchestrator Test Suite")
    print(f"Started at: {datetime.now()}")

    tests = [
        ("Basic Functionality", test_orchestrator_basic),
        ("ML with Fallback", test_ml_fallback_behavior),
        ("Error Handling", test_error_handling),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"RUNNING TEST: {test_name}")
        print(f"{'='*80}")

        try:
            result = await test_func()
            results.append((test_name, result))
            print(f"\n✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"\n❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:<30} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Production Orchestrator is working.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the logs above.")

    return passed == total


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
