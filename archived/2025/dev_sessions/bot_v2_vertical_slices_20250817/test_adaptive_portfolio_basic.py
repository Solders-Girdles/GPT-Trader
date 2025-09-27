#!/usr/bin/env python3
"""
Basic test script for adaptive portfolio slice.

Tests core functionality without external dependencies.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.bot_v2.features.adaptive_portfolio.config_manager import (
    load_portfolio_config,
    validate_portfolio_config,
    get_current_tier
)

from src.bot_v2.features.adaptive_portfolio.types import (
    PortfolioTier, PositionInfo, PortfolioSnapshot
)


def test_configuration_loading():
    """Test configuration loading and validation."""
    print("🔧 Testing Configuration Loading...")
    
    # Test default config
    try:
        config = load_portfolio_config()
        print(f"✅ Default config loaded: version {config.version}")
        print(f"   Tiers available: {list(config.tiers.keys())}")
        
        # Verify tier structure
        for tier_name, tier_config in config.tiers.items():
            print(f"   {tier_name}: ${tier_config.range[0]:,} - ${tier_config.range[1]:,}")
            
    except Exception as e:
        print(f"❌ Default config failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test validation
    try:
        validation = validate_portfolio_config()
        if validation.is_valid:
            print("✅ Configuration validation passed")
        else:
            print(f"❌ Configuration has errors: {validation.errors}")
            return False
        
        if validation.warnings:
            print(f"⚠️  Warnings: {validation.warnings}")
        
        if validation.suggestions:
            print(f"💡 Suggestions: {validation.suggestions}")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_tier_detection():
    """Test tier detection for different capital amounts."""
    print("\n📊 Testing Tier Detection...")
    
    test_amounts = [500, 1000, 1500, 2500, 5000, 10000, 25000, 50000, 100000]
    
    for amount in test_amounts:
        try:
            tier = get_current_tier(amount)
            print(f"✅ ${amount:,} → {tier} tier")
        except Exception as e:
            print(f"❌ Tier detection failed for ${amount:,}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_tier_manager():
    """Test tier manager functionality."""
    print("\n⚙️ Testing Tier Manager...")
    
    try:
        from src.bot_v2.features.adaptive_portfolio.tier_manager import TierManager
        
        config = load_portfolio_config()
        tier_manager = TierManager(config)
        
        # Test tier detection
        test_capital = 5000
        tier, tier_config = tier_manager.detect_tier(test_capital)
        print(f"✅ Tier detection: ${test_capital:,} → {tier.value}")
        print(f"   Config: {tier_config.positions.min_positions}-{tier_config.positions.max_positions} positions")
        print(f"   Risk: {tier_config.risk.daily_limit_pct}% daily limit")
        
        # Test transition logic
        should_transition, target_tier = tier_manager.should_transition(tier, test_capital)
        print(f"✅ Transition check: {should_transition} (target: {target_tier})")
        
    except Exception as e:
        print(f"❌ Tier manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_risk_manager():
    """Test risk manager functionality."""
    print("\n🛡️ Testing Risk Manager...")
    
    try:
        from src.bot_v2.features.adaptive_portfolio.risk_manager import AdaptiveRiskManager
        
        config = load_portfolio_config()
        risk_manager = AdaptiveRiskManager(config)
        
        # Create test portfolio snapshot
        test_positions = [
            PositionInfo(
                symbol="AAPL",
                shares=10,
                entry_price=150.0,
                current_price=155.0,
                position_value=1550.0,
                unrealized_pnl=50.0,
                unrealized_pnl_pct=3.33,
                days_held=5
            )
        ]
        
        portfolio_snapshot = PortfolioSnapshot(
            total_value=5000.0,
            cash=3450.0,
            positions=test_positions,
            daily_pnl=50.0,
            daily_pnl_pct=1.0,
            quarterly_pnl_pct=5.0,
            current_tier=PortfolioTier.SMALL,
            positions_count=1,
            largest_position_pct=31.0,
            sector_exposures={}
        )
        
        # Test risk metrics calculation
        tier_config = config.tiers["small"]
        risk_metrics = risk_manager.calculate_risk_metrics(portfolio_snapshot, tier_config)
        
        print(f"✅ Risk metrics calculated:")
        print(f"   Daily risk: {risk_metrics['daily_risk_pct']:.2f}%")
        print(f"   Risk utilization: {risk_metrics['daily_risk_utilization_pct']:.1f}%")
        print(f"   Tier compliant: {risk_metrics['tier_compliant']}")
        
        # Test position sizing
        position_size = risk_manager.calculate_position_size(
            total_portfolio_value=5000,
            tier_config=tier_config,
            confidence=0.8
        )
        print(f"✅ Position sizing: ${position_size:,.0f} for 80% confidence signal")
        
    except Exception as e:
        print(f"❌ Risk manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_configuration_templates():
    """Test different configuration templates."""
    print("\n📋 Testing Configuration Templates...")
    
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
                
                # Show key differences
                micro_tier = config.tiers["micro"]
                print(f"   Micro tier daily risk: {micro_tier.risk.daily_limit_pct}%")
                print(f"   Micro tier max positions: {micro_tier.positions.max_positions}")
                
            else:
                print(f"❌ {template_name.title()} template invalid: {validation.errors}")
                return False
                
        except Exception as e:
            print(f"❌ Template {template_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_json_configuration_structure():
    """Test JSON configuration structure."""
    print("\n📄 Testing JSON Configuration Structure...")
    
    try:
        config_path = project_root / "config" / "adaptive_portfolio_config.json"
        
        with open(config_path, 'r') as f:
            raw_config = json.load(f)
        
        # Check required top-level keys
        required_keys = ["version", "tiers", "costs", "validation", "market_constraints"]
        for key in required_keys:
            if key not in raw_config:
                print(f"❌ Missing required key: {key}")
                return False
        
        print("✅ All required top-level keys present")
        
        # Check tier structure
        for tier_name, tier_data in raw_config["tiers"].items():
            required_tier_keys = ["name", "range", "positions", "strategies", "risk", "trading"]
            for key in required_tier_keys:
                if key not in tier_data:
                    print(f"❌ Missing key '{key}' in tier '{tier_name}'")
                    return False
        
        print("✅ All tier structures valid")
        
        # Check value ranges make sense
        for tier_name, tier_data in raw_config["tiers"].items():
            daily_risk = tier_data["risk"]["daily_limit_pct"]
            if daily_risk <= 0 or daily_risk > 10:
                print(f"❌ Invalid daily risk limit in {tier_name}: {daily_risk}%")
                return False
        
        print("✅ Risk limits are reasonable")
        
    except Exception as e:
        print(f"❌ JSON structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Adaptive Portfolio Slice - Basic Test Suite")
    print("=" * 60)
    
    tests = [
        test_json_configuration_structure,
        test_configuration_loading,
        test_tier_detection,
        test_tier_manager,
        test_risk_manager,
        test_configuration_templates
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ PASSED\n")
            else:
                failed += 1
                print("❌ FAILED\n")
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print("💥 CRASHED\n")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic tests passed! Core functionality is working.")
        print("💡 Note: Strategy generation tests require yfinance dependency")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation and configuration.")
        return 1


if __name__ == "__main__":
    exit(main())