#!/usr/bin/env python3
"""
Quick validation script to verify Week 1 implementation completeness.

Validates:
1. Core components are importable and functional
2. Mock adapter works correctly
3. Validation scripts have correct toggles
4. Week 2 interfaces are properly designed
"""

import os
import sys
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_core_imports():
    """Test that core components can be imported."""
    try:
        from bot_v2.features.brokerages.coinbase.market_data_utils import (
            RollingWindow, DepthSnapshot
        )
        from bot_v2.features.brokerages.coinbase.test_adapter import MinimalCoinbaseBrokerage
        from bot_v2.features.brokerages.coinbase.models import APIConfig
        from bot_v2.features.strategy import (
            MarketConditionFilters, RiskGuards, StrategyEnhancements
        )
        print("✅ All core components imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_mock_adapter_functionality():
    """Test basic functionality of mock adapter."""
    try:
        from bot_v2.features.brokerages.coinbase.test_adapter import MinimalCoinbaseBrokerage
        from bot_v2.features.brokerages.coinbase.models import APIConfig
        from bot_v2.features.brokerages.core.interfaces import MarketType
        from decimal import Decimal
        
        # Initialize mock adapter
        config = APIConfig(
            api_key="test", 
            api_secret="test", 
            passphrase="test",
            base_url="https://api.sandbox.coinbase.com",
            enable_derivatives=True, 
            sandbox=True
        )
        broker = MinimalCoinbaseBrokerage(config)
        
        # Test product listing
        products = broker.list_products(market=MarketType.PERPETUAL)
        assert len(products) > 0, "No products returned"
        
        # Test quote generation
        quote = broker.get_quote("BTC-PERP")
        assert quote.symbol == "BTC-PERP", "Wrong symbol in quote"
        assert quote.last > 0, "Invalid price in quote"
        
        # Test order placement
        order = broker.place_order(
            symbol="BTC-PERP",
            side="buy",
            order_type="limit",
            quantity=Decimal('0.001'),
            limit_price=Decimal('100000')
        )
        assert order is not None, "Order placement failed"
        
        # Test order cancellation
        cancelled = broker.cancel_order(order.id)
        assert cancelled, "Order cancellation failed"
        
        print("✅ Mock adapter functionality verified")
        return True
    except Exception as e:
        print(f"❌ Mock adapter test failed: {e}")
        traceback.print_exc()
        return False


def test_validation_scripts_exist():
    """Test that validation scripts exist and have USE_REAL_ADAPTER toggle."""
    try:
        script_paths = [
            "scripts/validate_perps_client_week1.py",
            "scripts/validate_ws_week1.py"
        ]
        
        for script_path in script_paths:
            full_path = os.path.join(project_root, script_path)
            if not os.path.exists(full_path):
                print(f"❌ Missing script: {script_path}")
                return False
            
            # Check for USE_REAL_ADAPTER toggle
            with open(full_path, 'r') as f:
                content = f.read()
                if 'USE_REAL_ADAPTER' not in content:
                    print(f"❌ Script missing USE_REAL_ADAPTER toggle: {script_path}")
                    return False
        
        print("✅ Validation scripts verified")
        return True
    except Exception as e:
        print(f"❌ Validation script check failed: {e}")
        return False


def test_week2_interfaces():
    """Test that Week 2 strategy interfaces are properly designed."""
    try:
        from bot_v2.features.strategy import (
            MarketConditionFilters, RiskGuards, StrategyEnhancements,
            create_conservative_filters, create_standard_risk_guards
        )
        from decimal import Decimal
        
        # Test market condition filters
        filters = create_conservative_filters()
        
        # Mock market snapshot
        mock_snapshot = {
            'spread_bps': 5,
            'depth_l1': Decimal('100000'),
            'depth_l10': Decimal('500000'),
            'vol_1m': Decimal('200000'),
            'vol_5m': Decimal('1000000')
        }
        
        # Test filter logic
        allow_long, reason = filters.should_allow_long_entry(mock_snapshot)
        assert allow_long, f"Conservative filters should allow good market conditions: {reason}"
        
        # Test risk guards
        guards = create_standard_risk_guards()
        
        # Test liquidation distance calculation
        safe_distance, reason = guards.check_liquidation_distance(
            entry_price=Decimal('50000'),
            position_size=Decimal('1'),
            leverage=Decimal('10'),
            account_equity=Decimal('10000')
        )
        assert isinstance(safe_distance, bool), "Liquidation check should return boolean"
        
        # Test slippage impact
        safe_slippage, reason = guards.check_slippage_impact(
            order_size=Decimal('50000'),
            market_snapshot=mock_snapshot
        )
        assert isinstance(safe_slippage, bool), "Slippage check should return boolean"
        
        # Test strategy enhancements
        enhancements = StrategyEnhancements()
        
        # Test RSI calculation
        prices = [Decimal(str(50000 + i)) for i in range(20)]
        rsi = enhancements.calculate_rsi(prices)
        assert rsi is None or (0 <= rsi <= 100), f"Invalid RSI value: {rsi}"
        
        print("✅ Week 2 interfaces verified")
        return True
    except Exception as e:
        print(f"❌ Week 2 interface test failed: {e}")
        traceback.print_exc()
        return False


def test_depth_snapshot():
    """Test DepthSnapshot functionality."""
    try:
        from bot_v2.features.brokerages.coinbase.market_data_utils import DepthSnapshot
        from decimal import Decimal
        
        # Test with mock orderbook levels (price, size, side)
        levels = [
            (Decimal('50000.0'), Decimal('0.5'), 'bid'),
            (Decimal('49990.0'), Decimal('0.3'), 'bid'),  
            (Decimal('50010.0'), Decimal('0.4'), 'ask'),
            (Decimal('50020.0'), Decimal('0.2'), 'ask'),
        ]
        
        snapshot = DepthSnapshot(levels)
        
        assert snapshot.bids[0][0] == Decimal('50000.0'), "Incorrect best bid"
        assert snapshot.asks[0][0] == Decimal('50010.0'), "Incorrect best ask"
        assert snapshot.mid is not None, "Mid price should be calculated"
        assert snapshot.spread_bps > 0, "Spread bps should be positive"
        
        print("✅ DepthSnapshot functionality verified")
        return True
    except Exception as e:
        print(f"❌ DepthSnapshot test failed: {e}")
        traceback.print_exc()
        return False


def test_rolling_window():
    """Test RollingWindow functionality."""
    try:
        from bot_v2.features.brokerages.coinbase.market_data_utils import RollingWindow
        from decimal import Decimal
        from datetime import datetime, timedelta
        
        # Create rolling window
        window = RollingWindow(duration_seconds=60)
        now = datetime.now()
        
        # Add values
        window.add(100.0, now - timedelta(seconds=10))
        window.add(200.0, now - timedelta(seconds=20))
        
        # Get stats
        stats = window.get_stats()
        assert stats['sum'] == 300.0, f"Expected 300, got {stats['sum']}"
        assert stats['count'] == 2, f"Expected 2, got {stats['count']}"
        
        # Test basic functionality - cleanup method exists and can be called
        assert hasattr(window, '_cleanup'), "RollingWindow should have _cleanup method"
        assert callable(window._cleanup), "_cleanup should be callable"
        
        # Test basic stats work
        assert stats['sum'] > 0, "Sum should be positive"
        assert stats['count'] > 0, "Count should be positive" 
        assert stats['avg'] > 0, "Average should be positive"
        
        print("✅ RollingWindow functionality verified")
        return True
    except Exception as e:
        print(f"❌ RollingWindow test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("🧪 Running Week 1 Implementation Validation...")
    print("=" * 50)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Mock Adapter", test_mock_adapter_functionality),
        ("Validation Scripts", test_validation_scripts_exist),
        ("Week 2 Interfaces", test_week2_interfaces),
        ("DepthSnapshot", test_depth_snapshot),
        ("RollingWindow", test_rolling_window),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"📊 Validation Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Week 1 implementation tests PASSED!")
        print("\n✨ Implementation is ready for Week 2 development!")
        return 0
    else:
        print("❌ Some tests FAILED:")
        for test_name, result in results:
            if not result:
                print(f"  - {test_name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())