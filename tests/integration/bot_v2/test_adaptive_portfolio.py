#!/usr/bin/env python3
"""
Test script for adaptive portfolio slice.

Tests configuration loading, tier detection, and basic functionality.
"""
import pytest
from pathlib import Path

from bot_v2.features.adaptive_portfolio import (
    run_adaptive_strategy,
    load_portfolio_config,
    validate_portfolio_config,
    get_current_tier
)
from bot_v2.data_providers import (
    MockProvider as MockDataProvider,
    get_data_provider as create_data_provider,
)

pytestmark = pytest.mark.integration


def test_configuration_loading():
    """Assert config loads and validates with non-empty tiers."""
    config = load_portfolio_config()
    assert hasattr(config, "tiers") and isinstance(config.tiers, dict)
    assert len(config.tiers) > 0

    validation = validate_portfolio_config()
    assert validation.is_valid, f"Config validation errors: {validation.errors}"
    # Warnings allowed but should be a list
    assert isinstance(validation.warnings, list)


def test_tier_detection():
    """Assert returned tier names match configured ranges."""
    config = load_portfolio_config()
    test_amounts = [500, 1500, 3000, 7500, 15000, 30000, 100000]

    for amount in test_amounts:
        tier_name = get_current_tier(amount)
        assert tier_name in config.tiers
        lo, hi = config.tiers[tier_name].range
        # Largest tier may be open-ended; only check lower bound in that case
        if hi is not None:
            assert lo <= amount < hi
        else:
            assert amount >= lo


def test_adaptive_strategy_generates_valid_result():
    """Adaptive strategy returns a well-formed result across tiers."""
    mock_provider = MockDataProvider()
    test_capitals = [1000, 5000, 15000, 50000]

    for capital in test_capitals:
        result = run_adaptive_strategy(
            current_capital=capital,
            symbols=["AAPL", "MSFT"],
            data_provider=mock_provider,
            prefer_real_data=False,
        )
        # Tier consistency
        expected_tier = get_current_tier(capital)
        assert result.current_tier.value == expected_tier
        assert result.tier_config is not None
        # Signals shape and bounds
        assert isinstance(result.signals, list)
        assert len(result.signals) <= result.tier_config.positions.max_positions
        # Risk metrics present
        assert isinstance(result.risk_metrics, dict)
        for k in ["total_value", "cash_pct", "positions_count"]:
            assert k in result.risk_metrics
        # Recommendations and warnings are lists
        assert isinstance(result.recommended_actions, list)
        assert isinstance(result.warnings, list)


project_root = Path(__file__).resolve().parents[3]


def test_configuration_templates_if_present():
    """If templates exist, they should load and validate."""
    templates = [
        "config/adaptive_portfolio_conservative.json",
        "config/adaptive_portfolio_aggressive.json",
    ]
    for template_path in templates:
        full_path = project_root / template_path
        if not full_path.exists():
            pytest.skip(f"Template not found: {template_path}")
        config = load_portfolio_config(str(full_path))
        assert config is not None and hasattr(config, "tiers")
        validation = validate_portfolio_config(str(full_path))
        assert validation.is_valid, f"Template invalid: {validation.errors}"


def test_tier_transitions():
    """Tier changes are detected when crossing configured ranges."""
    scenarios = [
        {"from_capital": 2000, "to_capital": 3000, "expected_transition": True},
        {"from_capital": 5000, "to_capital": 2000, "expected_transition": True},
        {"from_capital": 1500, "to_capital": 1600, "expected_transition": False},
    ]
    for s in scenarios:
        from_tier = get_current_tier(s["from_capital"])
        to_tier = get_current_tier(s["to_capital"])
        assert (from_tier != to_tier) is s["expected_transition"]


def test_data_provider_abstraction_basic():
    """Mock provider returns well-formed data and price."""
    mock_provider = MockDataProvider()
    data = mock_provider.get_historical_data("AAPL", period="30d")
    assert len(data) > 0
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in data.columns
    assert mock_provider.get_current_price("AAPL") > 0


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
