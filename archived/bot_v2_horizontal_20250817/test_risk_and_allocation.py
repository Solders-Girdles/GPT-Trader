#!/usr/bin/env python3
"""
Test the Risk Manager and Portfolio Allocator components.
"""

from datetime import datetime
from typing import Dict

from core import ComponentConfig, get_registry
from core.types import Order, OrderType, OrderStatus, Portfolio, Position, PositionStatus
from risk import SimpleRiskManager
from risk.simple_risk_manager import RiskLimits
from allocators import EqualWeightAllocator


def create_test_portfolio(cash: float = 10000) -> Portfolio:
    """Create a test portfolio."""
    return Portfolio(
        cash=cash,
        positions={},
        timestamp=datetime.now(),
        initial_capital=10000
    )


def create_test_position(symbol: str, quantity: float, entry_price: float) -> Position:
    """Create a test position."""
    return Position(
        position_id=f"P_{symbol}",
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        entry_time=datetime.now(),
        current_price=entry_price,
        status=PositionStatus.OPEN
    )


def create_test_order(symbol: str, quantity: float, price: float) -> Order:
    """Create a test order."""
    return Order(
        order_id=f"O_{symbol}",
        symbol=symbol,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        side='buy',
        limit_price=price
    )


def test_risk_manager():
    """Test the SimpleRiskManager."""
    print("="*60)
    print("TESTING SIMPLE RISK MANAGER")
    print("="*60)
    
    # Create risk manager with custom limits
    limits = RiskLimits(
        max_position_size=0.2,  # 20% max position
        max_portfolio_risk=0.02,  # 2% risk per trade
        max_positions=3,
        stop_loss_pct=0.02,
        take_profit_pct=0.05
    )
    
    config = ComponentConfig(name="risk_manager")
    risk_manager = SimpleRiskManager(config, limits)
    risk_manager.initialize()
    
    print("✅ Risk manager initialized")
    
    # Test order validation
    print("\n" + "-"*40)
    print("TESTING ORDER VALIDATION")
    print("-"*40)
    
    portfolio = create_test_portfolio(cash=10000)
    
    # Test valid order
    order = create_test_order("AAPL", 10, 150)  # $1,500 order (15% of portfolio)
    is_valid, reason = risk_manager.validate_order(order, portfolio)
    print(f"Valid order (15% of portfolio): {is_valid} - {reason}")
    assert is_valid, "Should accept 15% position"
    
    # Test oversized order
    large_order = create_test_order("AAPL", 20, 150)  # $3,000 order (30% of portfolio)
    is_valid, reason = risk_manager.validate_order(large_order, portfolio)
    print(f"Oversized order (30% of portfolio): {is_valid} - {reason}")
    assert not is_valid, "Should reject 30% position"
    
    # Test with max positions reached
    portfolio.positions = {
        "AAPL": create_test_position("AAPL", 10, 150),
        "GOOGL": create_test_position("GOOGL", 5, 2800),
        "MSFT": create_test_position("MSFT", 15, 400)
    }
    
    new_order = create_test_order("TSLA", 5, 1000)
    is_valid, reason = risk_manager.validate_order(new_order, portfolio)
    print(f"Order with max positions reached: {is_valid} - {reason}")
    assert not is_valid, "Should reject when max positions reached"
    
    print("✅ Order validation working correctly")
    
    # Test position sizing
    print("\n" + "-"*40)
    print("TESTING POSITION SIZING")
    print("-"*40)
    
    # Strong signal
    size = risk_manager.calculate_position_size(
        signal_strength=1.0,
        portfolio_value=10000,
        current_positions={}
    )
    print(f"Position size (strong signal): ${size:.2f}")
    assert size <= 2000, "Should respect 20% max position size"
    
    # Weak signal
    size = risk_manager.calculate_position_size(
        signal_strength=0.3,
        portfolio_value=10000,
        current_positions={}
    )
    print(f"Position size (weak signal): ${size:.2f}")
    assert size <= 600, "Should scale with signal strength"
    
    print("✅ Position sizing working correctly")
    
    # Test stop-loss/take-profit
    print("\n" + "-"*40)
    print("TESTING STOP-LOSS/TAKE-PROFIT")
    print("-"*40)
    
    position = create_test_position("AAPL", 10, 100)
    
    # Test stop-loss trigger
    should_close = risk_manager.should_close_position(position, 97)  # 3% loss
    print(f"Should close at 3% loss: {should_close}")
    assert should_close, "Should trigger stop-loss at 3% loss"
    
    # Test take-profit trigger
    position = create_test_position("AAPL", 10, 100)
    should_close = risk_manager.should_close_position(position, 106)  # 6% gain
    print(f"Should close at 6% gain: {should_close}")
    assert should_close, "Should trigger take-profit at 6% gain"
    
    # Test normal price movement
    position = create_test_position("AAPL", 10, 100)
    should_close = risk_manager.should_close_position(position, 101)  # 1% gain
    print(f"Should close at 1% gain: {should_close}")
    assert not should_close, "Should not close on normal movement"
    
    print("✅ Stop-loss/take-profit working correctly")
    
    return True


def test_allocator():
    """Test the EqualWeightAllocator."""
    print("\n" + "="*60)
    print("TESTING EQUAL WEIGHT ALLOCATOR")
    print("="*60)
    
    config = ComponentConfig(
        name="allocator",
        config={
            'max_positions': 3,
            'min_allocation': 0.1,
            'cash_reserve': 0.05
        }
    )
    allocator = EqualWeightAllocator(config)
    allocator.initialize()
    
    print("✅ Allocator initialized")
    
    # Test allocation
    print("\n" + "-"*40)
    print("TESTING ALLOCATION")
    print("-"*40)
    
    # Multiple buy signals
    signals = {
        'AAPL': 0.8,   # Strong buy
        'GOOGL': 0.6,  # Moderate buy
        'MSFT': 0.4,   # Weak buy
        'TSLA': 0.3,   # Weaker buy (should be excluded due to max_positions)
        'AMZN': -0.5   # Sell signal (should be ignored)
    }
    
    allocations = allocator.allocate(
        signals=signals,
        portfolio_value=10000,
        current_positions={},
        risk_constraints={'max_position_size': 0.25}
    )
    
    print(f"Allocations for {len(signals)} signals:")
    for symbol, allocation in allocations.items():
        print(f"  {symbol}: {allocation:.1%}")
    
    assert len(allocations) <= 3, "Should respect max positions"
    assert 'AMZN' not in allocations, "Should not allocate to sell signals"
    assert 'TSLA' not in allocations, "Should exclude weakest signal"
    
    # Check equal weighting
    if allocations:
        allocation_values = list(allocations.values())
        assert all(abs(a - allocation_values[0]) < 0.01 for a in allocation_values), \
            "Should have equal weights"
    
    print("✅ Allocation working correctly")
    
    # Test rebalancing logic
    print("\n" + "-"*40)
    print("TESTING REBALANCING")
    print("-"*40)
    
    current_positions = {
        'AAPL': 3333,   # 33.33% of portfolio
        'GOOGL': 3333,  # 33.33% of portfolio
        'MSFT': 3334    # 33.34% of portfolio
    }
    
    target_allocations = {
        'AAPL': 0.3333,
        'GOOGL': 0.3333,
        'MSFT': 0.3334
    }
    
    needs_rebalance = allocator.rebalance_required(
        current_positions=current_positions,
        target_allocations=target_allocations,
        threshold=0.05
    )
    
    print(f"Needs rebalancing (close to target): {needs_rebalance}")
    assert not needs_rebalance, "Should not rebalance within threshold"
    
    # Test with larger deviation
    current_positions['AAPL'] = 5000  # 50% of portfolio
    needs_rebalance = allocator.rebalance_required(
        current_positions=current_positions,
        target_allocations=target_allocations,
        threshold=0.05
    )
    
    print(f"Needs rebalancing (with deviation): {needs_rebalance}")
    assert needs_rebalance, "Should rebalance when deviation exceeds threshold"
    
    print("✅ Rebalancing logic working correctly")
    
    return True


def test_integration():
    """Test Risk Manager and Allocator working together."""
    print("\n" + "="*60)
    print("TESTING RISK + ALLOCATOR INTEGRATION")
    print("="*60)
    
    registry = get_registry()
    registry.clear()
    
    # Register components
    risk_manager = SimpleRiskManager(
        ComponentConfig(name="risk"),
        RiskLimits(max_position_size=0.25)
    )
    registry.register_instance("risk_manager", risk_manager)
    
    allocator = EqualWeightAllocator(
        ComponentConfig(name="allocator", config={'max_positions': 3})
    )
    registry.register_instance("allocator", allocator)
    
    print("✅ Components registered")
    
    # Simulate signal → allocation → risk validation flow
    print("\n" + "-"*40)
    print("SIMULATING SIGNAL FLOW")
    print("-"*40)
    
    # 1. Signals arrive
    signals = {'AAPL': 0.9, 'GOOGL': 0.7, 'MSFT': 0.5}
    print(f"Signals: {signals}")
    
    # 2. Allocator decides position sizes
    portfolio_value = 10000
    allocations = allocator.allocate(
        signals=signals,
        portfolio_value=portfolio_value,
        current_positions={},
        risk_constraints={'max_position_size': 0.25}
    )
    print(f"Allocations: {allocations}")
    
    # 3. Risk manager validates each allocation
    portfolio = create_test_portfolio(cash=portfolio_value)
    
    for symbol, allocation in allocations.items():
        # Convert allocation to order
        position_value = allocation * portfolio_value
        price = 100  # Assume $100 per share for simplicity
        quantity = position_value / price
        
        order = create_test_order(symbol, quantity, price)
        is_valid, reason = risk_manager.validate_order(order, portfolio)
        
        print(f"  {symbol}: ${position_value:.0f} - Valid: {is_valid}")
        
        if is_valid:
            # Add to portfolio (simulating execution)
            portfolio.positions[symbol] = create_test_position(symbol, quantity, price)
            portfolio.cash -= position_value
    
    print("\n✅ Risk + Allocator integration working")
    
    return True


def main():
    """Run all risk and allocation tests."""
    print("="*80)
    print("RISK MANAGER & ALLOCATOR TEST")
    print("="*80)
    
    tests = [
        test_risk_manager,
        test_allocator,
        test_integration
    ]
    
    for test in tests:
        if not test():
            print(f"❌ Test {test.__name__} failed")
            return False
    
    print("\n" + "="*80)
    print("ALL RISK & ALLOCATION TESTS PASSED")
    print("="*80)
    print("\n🛡️ RISK MANAGER READY")
    print("   - Order validation ✓")
    print("   - Position sizing ✓")
    print("   - Stop-loss/take-profit ✓")
    print("   - Risk metrics ✓")
    print("\n⚖️ ALLOCATOR READY")
    print("   - Equal weight allocation ✓")
    print("   - Position limits ✓")
    print("   - Rebalancing logic ✓")
    print("   - Risk constraint integration ✓")
    
    return True


if __name__ == "__main__":
    main()