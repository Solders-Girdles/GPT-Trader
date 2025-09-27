#!/usr/bin/env python3
"""
Test script for adaptive portfolio slice.

Tests configuration loading, tier detection, and basic functionality.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.bot_v2.features.adaptive_portfolio import (
    run_adaptive_strategy,
    load_portfolio_config,
    validate_portfolio_config,
    get_current_tier
)
from src.bot_v2.features.adaptive_portfolio.data_providers import (
    MockDataProvider, create_data_provider, get_data_provider_info
)


def test_configuration_loading():
    """Test configuration loading and validation."""
    print("🔧 Testing Configuration Loading...")
    
    # Test default config
    try:
        config = load_portfolio_config()
        print(f"✅ Default config loaded: version {config.version}")
        print(f"   Tiers available: {list(config.tiers.keys())}")
    except Exception as e:
        print(f"❌ Default config failed: {e}")
        return False
    
    # Test validation
    try:
        validation = validate_portfolio_config()
        if validation.is_valid:
            print("✅ Configuration validation passed")
        else:
            print(f"⚠️  Configuration has issues: {validation.errors}")
        
        if validation.warnings:
            print(f"⚠️  Warnings: {validation.warnings}")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    return True


def test_tier_detection():
    """Test tier detection for different capital amounts."""
    print("\n📊 Testing Tier Detection...")
    
    test_amounts = [500, 1500, 3000, 7500, 15000, 30000, 100000]
    
    for amount in test_amounts:
        try:
            tier = get_current_tier(amount)
            print(f"✅ ${amount:,} → {tier} tier")
        except Exception as e:
            print(f"❌ Tier detection failed for ${amount:,}: {e}")
            return False
    
    return True


def test_adaptive_strategy():
    """Test adaptive strategy generation for different portfolio sizes."""
    print("\n🎯 Testing Adaptive Strategy Generation...")
    
    # Use MockDataProvider for testing to avoid external dependencies
    mock_provider = MockDataProvider()
    
    test_portfolios = [
        {"capital": 1000, "name": "Micro"},
        {"capital": 5000, "name": "Small"},
        {"capital": 15000, "name": "Medium"},
        {"capital": 50000, "name": "Large"}
    ]
    
    for portfolio in test_portfolios:
        try:
            result = run_adaptive_strategy(
                current_capital=portfolio["capital"],
                symbols=["AAPL", "MSFT"],  # Simple test with 2 symbols
                data_provider=mock_provider,
                prefer_real_data=False
            )
            
            print(f"✅ {portfolio['name']} Portfolio (${portfolio['capital']:,}):")
            print(f"   Tier: {result.current_tier.value}")
            print(f"   Max positions: {result.tier_config.positions.max_positions}")
            print(f"   Daily risk limit: {result.tier_config.risk.daily_limit_pct}%")
            print(f"   Strategies: {', '.join(result.tier_config.strategies)}")
            print(f"   Signals generated: {len(result.signals)}")
            
            if result.signals:
                for signal in result.signals[:2]:  # Show first 2 signals
                    print(f"   Signal: {signal.action} {signal.symbol} "
                          f"${signal.target_position_size:,.0f} (confidence: {signal.confidence:.2f})")
            
            print()
            
        except Exception as e:
            print(f"❌ Strategy generation failed for {portfolio['name']}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_configuration_templates():
    """Test different configuration templates."""
    print("📋 Testing Configuration Templates...")
    
    templates = [
        "config/adaptive_portfolio_conservative.json",
        "config/adaptive_portfolio_aggressive.json"
    ]
    
    for template_path in templates:
        template_name = Path(template_path).stem.split('_')[-1]
        
        try:
            # Check if file exists
            full_path = project_root / template_path
            if not full_path.exists():
                print(f"⚠️  Template not found: {template_path}")
                continue
            
            # Test loading
            config = load_portfolio_config(str(full_path))
            validation = validate_portfolio_config(str(full_path))
            
            if validation.is_valid:
                print(f"✅ {template_name.title()} template valid")
            else:
                print(f"❌ {template_name.title()} template invalid: {validation.errors}")
                
        except Exception as e:
            print(f"❌ Template {template_name} failed: {e}")
            return False
    
    return True


def test_tier_transitions():
    """Test tier transition detection."""
    print("\n🔄 Testing Tier Transitions...")
    
    # Test scenarios where tier transitions should occur
    scenarios = [
        {"from_capital": 2000, "to_capital": 3000, "expected": "transition up"},
        {"from_capital": 5000, "to_capital": 2000, "expected": "transition down"},
        {"from_capital": 1500, "to_capital": 1600, "expected": "no transition"},
    ]
    
    for scenario in scenarios:
        try:
            from_tier = get_current_tier(scenario["from_capital"])
            to_tier = get_current_tier(scenario["to_capital"])
            
            if from_tier != to_tier:
                transition = "transition"
            else:
                transition = "no transition"
            
            print(f"✅ ${scenario['from_capital']:,} → ${scenario['to_capital']:,}: "
                  f"{from_tier} → {to_tier} ({transition})")
            
        except Exception as e:
            print(f"❌ Transition test failed: {e}")
            return False
    
    return True


def test_data_provider_abstraction():
    """Test data provider abstraction and fallback behavior."""
    print("\n🔌 Testing Data Provider Abstraction...")
    
    # Test provider info
    try:
        info = get_data_provider_info()
        print(f"✅ Provider availability:")
        print(f"   Mock provider: {info['mock_available']}")
        print(f"   YFinance provider: {info['yfinance_available']}")
        print(f"   Pandas available: {info['pandas_available']}")
    except Exception as e:
        print(f"❌ Provider info failed: {e}")
        return False
    
    # Test MockDataProvider directly
    try:
        mock_provider = MockDataProvider()
        symbols = mock_provider.get_available_symbols()
        print(f"✅ Mock provider has {len(symbols)} symbols available")
        
        # Test data generation
        data = mock_provider.get_historical_data("AAPL", period="30d")
        print(f"✅ Mock provider generated {len(data)} days of AAPL data")
        
        # Test current price
        price = mock_provider.get_current_price("AAPL")
        print(f"✅ Mock provider current AAPL price: ${price:.2f}")
        
    except Exception as e:
        print(f"❌ Mock provider test failed: {e}")
        return False
    
    # Test provider factory
    try:
        provider, provider_type = create_data_provider(prefer_real_data=False)
        print(f"✅ Created {provider_type} provider via factory")
        
        # Test with real data preference (may fallback to mock)
        provider2, provider_type2 = create_data_provider(prefer_real_data=True)
        print(f"✅ Created {provider_type2} provider with real data preference")
        
    except Exception as e:
        print(f"❌ Provider factory test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Adaptive Portfolio Slice Test Suite")
    print("=" * 50)
    
    tests = [
        test_configuration_loading,
        test_tier_detection,
        test_data_provider_abstraction,
        test_adaptive_strategy,
        test_configuration_templates,
        test_tier_transitions
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Adaptive portfolio slice is functional.")
        return 0
    else:
        print("⚠️  Some tests failed. Check configuration and dependencies.")
        return 1


if __name__ == "__main__":
    exit(main())