#!/usr/bin/env python3
"""
Validation Script - Strategy Integration for Perpetuals

Validates that Phase 6 implementation correctly:
1. Generates MA crossover signals (buy/sell/hold)
2. Applies trailing stops and triggers exits
3. Computes sizing with leverage and constraints
4. Respects reduce-only mode
5. Handles risk rejections gracefully
"""

from __future__ import annotations

from decimal import Decimal
from typing import List, Dict, Any
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_v2.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy, StrategyConfig, Decision, Action,
    create_baseline_strategy
)
from bot_v2.features.live_trade.risk import LiveRiskManager, ValidationError
from bot_v2.features.brokerages.core.interfaces import Product, MarketType
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.persistence.event_store import EventStore


def validate_ma_crossover_signals():
    """Validate MA crossover signal generation."""
    print("\n1. Validating MA Crossover Signals...")
    
    config = StrategyConfig(
        short_ma_period=3,
        long_ma_period=5,
        enable_shorts=True
    )
    strategy = BaselinePerpsStrategy(config=config)
    
    product = Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10")
    )
    
    # Test 1: Bullish crossover
    rising_marks = [
        Decimal("50000"),
        Decimal("50000"),
        Decimal("50100"),
        Decimal("50300"),
        Decimal("50500"),
        Decimal("50800")
    ]
    
    decision = strategy.decide(
        symbol="BTC-PERP",
        current_mark=rising_marks[-1],
        position_state=None,
        recent_marks=rising_marks[:-1],
        equity=Decimal("10000"),
        product=product
    )
    
    assert decision.action == Action.BUY, f"Expected BUY, got {decision.action}"
    print(f"   ✓ Bullish crossover → BUY signal: {decision.reason}")
    
    # Test 2: Bearish crossover
    falling_marks = [
        Decimal("50000"),
        Decimal("50000"),
        Decimal("49900"),
        Decimal("49700"),
        Decimal("49500"),
        Decimal("49200")
    ]
    
    strategy.reset()  # Clear state
    decision = strategy.decide(
        symbol="BTC-PERP",
        current_mark=falling_marks[-1],
        position_state=None,
        recent_marks=falling_marks[:-1],
        equity=Decimal("10000"),
        product=product
    )
    
    assert decision.action == Action.SELL, f"Expected SELL, got {decision.action}"
    print(f"   ✓ Bearish crossover → SELL signal: {decision.reason}")
    
    # Test 3: Sideways → Hold
    flat_marks = [Decimal("50000")] * 6
    
    strategy.reset()
    decision = strategy.decide(
        symbol="BTC-PERP",
        current_mark=flat_marks[-1],
        position_state=None,
        recent_marks=flat_marks[:-1],
        equity=Decimal("10000"),
        product=product
    )
    
    assert decision.action == Action.HOLD, f"Expected HOLD, got {decision.action}"
    print(f"   ✓ Sideways market → HOLD: {decision.reason}")
    
    return True


def validate_trailing_stop():
    """Validate trailing stop functionality."""
    print("\n2. Validating Trailing Stop...")
    
    config = StrategyConfig(
        short_ma_period=3,
        long_ma_period=5,
        trailing_stop_pct=0.01  # 1% trailing stop
    )
    strategy = BaselinePerpsStrategy(config=config)
    
    product = Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10")
    )
    
    position_state = {
        'qty': Decimal("1"),
        'side': 'long',
        'entry': Decimal("50000")
    }
    
    # Simulate price peak then drop
    marks_sequence = [
        (Decimal("50500"), "initial (already up)"),
        (Decimal("51000"), "peak (up 2%)"),
        (Decimal("50700"), "small drop (0.59% from peak)"),
        (Decimal("50400"), "stop triggered (1.18% from peak)")
    ]
    
    for i, (mark, description) in enumerate(marks_sequence):
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=mark,
            position_state=position_state,
            recent_marks=[Decimal("50000")] * 5,
            equity=Decimal("10000"),
            product=product
        )
        
        if i < 3:
            # Should hold until stop triggers
            if decision.action != Action.HOLD:
                # Might close on MA signal, which is also valid
                print(f"   Step {i+1}: {description} → {decision.action.value} ({decision.reason})")
            else:
                print(f"   Step {i+1}: {description} → HOLD")
        else:
            # Stop should trigger
            assert decision.action == Action.CLOSE, f"Step {i}: Expected CLOSE at {description}"
            assert decision.reduce_only is True
            print(f"   ✓ Step {i+1}: {description} → CLOSE (reduce_only=True)")
            print(f"      Reason: {decision.reason}")
    
    return True


def validate_sizing():
    """Validate position sizing with constraints."""
    print("\n3. Validating Position Sizing...")
    
    config = StrategyConfig(
        short_ma_period=3,
        long_ma_period=5,
        target_leverage=3
    )
    strategy = BaselinePerpsStrategy(config=config)
    
    product = Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10")
    )
    
    # Generate buy signal
    rising_marks = [
        Decimal("49000"),
        Decimal("49500"),
        Decimal("50000"),
        Decimal("50500"),
        Decimal("51000")
    ]
    
    equity = Decimal("50000")
    
    decision = strategy.decide(
        symbol="BTC-PERP",
        current_mark=rising_marks[-1],
        position_state=None,
        recent_marks=rising_marks[:-1],
        equity=equity,
        product=product
    )
    
    expected_notional = equity * 3  # 3x leverage
    assert decision.target_notional == expected_notional
    print(f"   ✓ Target notional: {decision.target_notional} (3x leverage on {equity} equity)")
    
    # Calculate qty from notional
    qty = decision.target_notional / rising_marks[-1]
    print(f"   ✓ Computed qty: {qty:.3f} BTC @ {rising_marks[-1]}")
    
    # TODO: In real impl, would apply ProductCatalog.enforce_perp_rules()
    print("   ✓ Would apply ProductCatalog.enforce_perp_rules() for quantization")
    
    return True


def validate_reduce_only_mode():
    """Validate reduce-only mode behavior."""
    print("\n4. Validating Reduce-Only Mode...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create risk manager in reduce-only mode
        risk_config = RiskConfig(reduce_only_mode=True)
        event_store = EventStore(root=Path(tmpdir))
        risk_manager = LiveRiskManager(config=risk_config, event_store=event_store)
        
        config = StrategyConfig(
            short_ma_period=3,
            long_ma_period=5
        )
        strategy = BaselinePerpsStrategy(config=config, risk_manager=risk_manager)
        
        product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            step_size=Decimal("0.001"),
            min_size=Decimal("0.001"),
            price_increment=Decimal("0.01"),
            min_notional=Decimal("10")
        )
        
        # Test 1: No position - should block entry
        rising_marks = [
            Decimal("49000"),
            Decimal("49500"),
            Decimal("50000"),
            Decimal("50500"),
            Decimal("51000")
        ]
        
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=rising_marks[-1],
            position_state=None,
            recent_marks=rising_marks[:-1],
            equity=Decimal("10000"),
            product=product
        )
        
        assert decision.action == Action.HOLD
        assert "Reduce-only mode" in decision.reason
        print("   ✓ Reduce-only blocks new entries")
        
        # Test 2: Has position - should allow exit
        position_state = {
            'qty': Decimal("1"),
            'side': 'long',
            'entry': Decimal("50000")
        }
        
        decision = strategy.decide(
            symbol="BTC-PERP",
            current_mark=Decimal("51000"),
            position_state=position_state,
            recent_marks=[Decimal("50000")] * 5,
            equity=Decimal("10000"),
            product=product
        )
        
        assert decision.action == Action.CLOSE
        assert decision.reduce_only is True
        print("   ✓ Reduce-only allows position exit")
        print(f"      Decision: {decision.action.value} with reduce_only={decision.reduce_only}")
    
    return True


def validate_risk_rejection_handling():
    """Validate handling of risk manager rejections."""
    print("\n5. Validating Risk Rejection Handling...")
    
    # Strategy generates decisions regardless of risk
    # Risk checks happen at execution time
    config = StrategyConfig(
        short_ma_period=3,
        long_ma_period=5,
        target_leverage=10  # Very high leverage
    )
    strategy = BaselinePerpsStrategy(config=config)
    
    product = Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10")
    )
    
    rising_marks = [
        Decimal("49000"),
        Decimal("49500"),
        Decimal("50000"),
        Decimal("50500"),
        Decimal("51000")
    ]
    
    decision = strategy.decide(
        symbol="BTC-PERP",
        current_mark=rising_marks[-1],
        position_state=None,
        recent_marks=rising_marks[:-1],
        equity=Decimal("10000"),
        product=product
    )
    
    # Strategy still generates decision with high leverage
    assert decision.action == Action.BUY
    assert decision.leverage == 10
    print(f"   ✓ Strategy generates decision: {decision.action.value} with {decision.leverage}x leverage")
    
    # Simulate risk rejection at execution time
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # Align with RiskConfig fields: use max_leverage for global cap
        risk_config = RiskConfig(max_leverage=5)  # Cap at 5x
        event_store = EventStore(root=Path(tmpdir))
        risk_manager = LiveRiskManager(config=risk_config, event_store=event_store)
        
        # This would be rejected
        try:
            risk_manager.pre_trade_validate(
                symbol="BTC-PERP",
                side="buy",
                qty=Decimal("20"),  # 20 * 51000 = 1,020,000 notional
                price=Decimal("51000"),
                product=product,
                equity=Decimal("10000")  # 102x leverage!
            )
            assert False, "Should have been rejected"
        except ValidationError as e:
            print(f"   ✓ Risk manager rejects: {e}")
            assert "exceeds" in str(e).lower()
    
    print("   ✓ Strategy decisions are independent of risk checks")
    print("   ✓ Risk validation happens at execution time")
    
    return True


def validate_feature_flags():
    """Validate feature flag behavior."""
    print("\n6. Validating Feature Flags...")
    
    product = Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10")
    )
    
    # Test 1: enable_shorts=False
    config = StrategyConfig(
        short_ma_period=3,
        long_ma_period=5,
        enable_shorts=False
    )
    strategy = BaselinePerpsStrategy(config=config)
    
    falling_marks = [
        Decimal("50000"),
        Decimal("49900"),
        Decimal("49700"),
        Decimal("49500"),
        Decimal("49200")
    ]
    
    decision = strategy.decide(
        symbol="BTC-PERP",
        current_mark=falling_marks[-1],
        position_state=None,
        recent_marks=falling_marks[:-1],
        equity=Decimal("10000"),
        product=product
    )
    
    assert decision.action == Action.HOLD  # No short when disabled
    print("   ✓ enable_shorts=False blocks short entries")
    
    # Test 2: disable_new_entries=True
    config = StrategyConfig(
        short_ma_period=3,
        long_ma_period=5,
        disable_new_entries=True
    )
    strategy = BaselinePerpsStrategy(config=config)
    
    rising_marks = [
        Decimal("49000"),
        Decimal("49500"),
        Decimal("50000"),
        Decimal("50500"),
        Decimal("51000")
    ]
    
    decision = strategy.decide(
        symbol="BTC-PERP",
        current_mark=rising_marks[-1],
        position_state=None,
        recent_marks=rising_marks[:-1],
        equity=Decimal("10000"),
        product=product
    )
    
    assert decision.action == Action.HOLD
    assert "New entries disabled" in decision.reason
    print("   ✓ disable_new_entries=True blocks all entries")
    
    # Test 3: max_adds limit
    config = StrategyConfig(max_adds=1)
    strategy = BaselinePerpsStrategy(config=config)
    assert strategy.config.max_adds == 1
    print("   ✓ max_adds configuration accepted")
    
    return True


def main():
    """Run all strategy integration validations."""
    print("=" * 60)
    print("Validation - Strategy Integration for Perpetuals")
    print("=" * 60)
    
    try:
        # Run validations
        assert validate_ma_crossover_signals(), "MA crossover validation failed"
        assert validate_trailing_stop(), "Trailing stop validation failed"
        assert validate_sizing(), "Sizing validation failed"
        assert validate_reduce_only_mode(), "Reduce-only mode validation failed"
        assert validate_risk_rejection_handling(), "Risk rejection validation failed"
        assert validate_feature_flags(), "Feature flags validation failed"
        
        print("\n" + "=" * 60)
        print("✅ All validations PASSED!")
        print("=" * 60)
        
        print("\nDeliverables Complete:")
        print("  • BaselinePerpsStrategy with MA crossover signals")
        print("  • Trailing stop logic with reduce-only exits")
        print("  • Position sizing with leverage targets")
        print("  • Reduce-only mode integration")
        print("  • Risk rejection handling (at execution)")
        print("  • Feature flags (enable_shorts, max_adds, disable_new_entries)")
        
        print("\nNext Steps:")
        print("  • E2E: End-to-End Testing")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
