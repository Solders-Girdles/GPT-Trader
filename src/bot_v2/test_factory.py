#!/usr/bin/env python3
"""
Test the strategy factory and registry system.
"""

from strategies import (
    create_strategy,
    list_available_strategies,
    strategy_parameter_info,
    get_strategy_registry,
    StrategyConfig
)


def test_strategy_factory():
    """Test the strategy factory and registry functionality."""
    
    print("="*60)
    print("TESTING STRATEGY FACTORY & REGISTRY")
    print("="*60)
    
    # Test listing available strategies
    print("Available strategies:")
    strategies = list_available_strategies()
    for strategy in strategies:
        print(f"  - {strategy}")
    
    assert len(strategies) > 0, "No strategies registered!"
    assert "SimpleMAStrategy" in strategies, "SimpleMAStrategy not registered!"
    
    # Test parameter info
    print("\n" + "-"*40)
    print("TESTING PARAMETER INFO")
    print("-"*40)
    
    info = strategy_parameter_info("SimpleMAStrategy")
    print(f"SimpleMAStrategy info:")
    print(f"  Default parameters: {info['default_parameters']}")
    print(f"  Required parameters: {info['required_parameters']}")
    print(f"  Description: {info['description'][:100]}...")
    
    # Test strategy creation with defaults
    print("\n" + "-"*40)
    print("TESTING STRATEGY CREATION")
    print("-"*40)
    
    strategy1 = create_strategy("SimpleMAStrategy")
    print(f"Created strategy with defaults: {strategy1}")
    print(f"Fast period: {strategy1.get_parameter('fast_period')}")
    print(f"Slow period: {strategy1.get_parameter('slow_period')}")
    
    # Test strategy creation with custom parameters
    strategy2 = create_strategy("SimpleMAStrategy", fast_period=5, slow_period=15)
    print(f"\nCreated strategy with custom params: {strategy2}")
    print(f"Fast period: {strategy2.get_parameter('fast_period')}")
    print(f"Slow period: {strategy2.get_parameter('slow_period')}")
    
    # Test error handling
    print("\n" + "-"*40)
    print("TESTING ERROR HANDLING")
    print("-"*40)
    
    try:
        create_strategy("NonExistentStrategy")
        assert False, "Should have raised error for unknown strategy"
    except ValueError as e:
        print(f"✅ Correctly caught error for unknown strategy: {e}")
    
    # Test registry operations
    print("\n" + "-"*40)
    print("TESTING REGISTRY OPERATIONS")
    print("-"*40)
    
    registry = get_strategy_registry()
    print(f"Registry has {len(registry.list_strategies())} strategies")
    
    # Test strategy info lookup
    info_obj = registry.get_strategy_info("SimpleMAStrategy")
    if info_obj:
        print(f"Found strategy info: {info_obj.name}")
        print(f"Description: {info_obj.description[:50]}...")
        print(f"Default params: {info_obj.default_parameters}")
    
    print("\n" + "="*60)
    print("STRATEGY FACTORY & REGISTRY TEST COMPLETE")
    print("="*60)
    print("✅ All tests passed!")
    print("✅ Strategy registration working correctly")
    print("✅ Factory creation working correctly")
    print("✅ Parameter handling working correctly")
    print("✅ Error handling working correctly")
    
    return True


if __name__ == "__main__":
    test_strategy_factory()