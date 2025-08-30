#!/usr/bin/env python3
"""
Test Position Sizer
===================

Simple test to verify the regression position sizer works correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.ml.regression_position_sizer import (
    RegressionPositionSizer, 
    PositionSizingConfig,
    DynamicRiskScaler
)


def test_position_sizing():
    """Test basic position sizing functionality."""
    print("🧪 Testing Position Sizing...")
    
    # Create position sizer with custom config
    config = PositionSizingConfig(
        base_risk_budget=0.025,
        max_position_size=0.2,
        transaction_cost=0.001,
        min_confidence=0.5,
        max_confidence=2.0
    )
    
    sizer = RegressionPositionSizer(config)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'High Confidence, Positive Returns',
            'predictions': {'AAPL': 0.05, 'MSFT': 0.03, 'GOOGL': 0.08},
            'confidences': {'AAPL': 0.9, 'MSFT': 0.8, 'GOOGL': 0.95},
            'volatilities': {'AAPL': 0.25, 'MSFT': 0.20, 'GOOGL': 0.30}
        },
        {
            'name': 'Mixed Signals',
            'predictions': {'AAPL': 0.02, 'MSFT': -0.03, 'GOOGL': 0.01},
            'confidences': {'AAPL': 0.6, 'MSFT': 0.7, 'GOOGL': 0.5},
            'volatilities': {'AAPL': 0.15, 'MSFT': 0.18, 'GOOGL': 0.22}
        },
        {
            'name': 'High Volatility',
            'predictions': {'AAPL': 0.04, 'MSFT': 0.06, 'GOOGL': 0.05},
            'confidences': {'AAPL': 0.8, 'MSFT': 0.75, 'GOOGL': 0.85},
            'volatilities': {'AAPL': 0.45, 'MSFT': 0.50, 'GOOGL': 0.40}
        }
    ]
    
    prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 2500.0}
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print("-" * 60)
        
        positions = sizer.calculate_portfolio_positions(
            predictions=scenario['predictions'],
            confidences=scenario['confidences'],
            prices=prices,
            volatilities=scenario['volatilities']
        )
        
        print(f"{'Symbol':<8} {'Prediction':<12} {'Confidence':<12} {'Volatility':<12} {'Position':<12}")
        print("-" * 60)
        
        for symbol in scenario['predictions']:
            pred = scenario['predictions'][symbol]
            conf = scenario['confidences'][symbol]
            vol = scenario['volatilities'][symbol]
            pos = positions.get(symbol, 0.0)
            
            print(f"{symbol:<8} {pred:+10.2%} {conf:10.2f} {vol:10.2%} {pos:+10.2%}")
        
        total_exposure = sum(abs(pos) for pos in positions.values())
        net_exposure = sum(positions.values())
        print(f"\nTotal Exposure: {total_exposure:.2%}")
        print(f"Net Exposure:   {net_exposure:+.2%}")


def test_dynamic_risk_scaling():
    """Test dynamic risk scaling functionality."""
    print("\n🔄 Testing Dynamic Risk Scaling...")
    
    scaler = DynamicRiskScaler(base_risk=0.02)
    
    # Test scenarios
    test_cases = [
        {
            'name': 'Good Performance, Low Vol',
            'returns': [0.01, 0.02, 0.015, 0.008, 0.012, 0.018, 0.005, 0.020, 0.011, 0.009],
            'market_vol': 0.12,
            'model_accuracy': 0.65
        },
        {
            'name': 'Poor Performance, High Vol',
            'returns': [-0.02, -0.01, 0.005, -0.015, 0.003, -0.008, -0.012, 0.001, -0.005, -0.018],
            'market_vol': 0.35,
            'model_accuracy': 0.42
        },
        {
            'name': 'Neutral Performance, Normal Vol',
            'returns': [0.002, -0.001, 0.003, 0.0, -0.002, 0.001, 0.004, -0.003, 0.002, 0.001],
            'market_vol': 0.18,
            'model_accuracy': 0.52
        }
    ]
    
    print(f"{'Scenario':<25} {'Base Risk':<12} {'Adjusted Risk':<15} {'Adjustment':<15}")
    print("-" * 70)
    
    for case in test_cases:
        adjusted_risk = scaler.get_adjusted_risk_budget(
            recent_returns=case['returns'],
            market_volatility=case['market_vol'],
            model_accuracy=case['model_accuracy']
        )
        
        adjustment_factor = adjusted_risk / scaler.base_risk
        
        print(f"{case['name']:<25} {scaler.base_risk:10.2%} {adjusted_risk:13.2%} {adjustment_factor:13.2f}x")


def test_transaction_cost_filter():
    """Test transaction cost filtering."""
    print("\n💰 Testing Transaction Cost Filter...")
    
    config = PositionSizingConfig(
        transaction_cost=0.002,  # Higher cost for testing
        base_risk_budget=0.03
    )
    
    sizer = RegressionPositionSizer(config)
    
    # Set some current positions
    sizer.current_positions = {
        'AAPL': 0.05,   # 5% current position
        'MSFT': -0.03,  # -3% current position
        'GOOGL': 0.0    # No current position
    }
    
    # Test with different predicted returns
    target_positions = {
        'AAPL': 0.08,   # Want to increase position
        'MSFT': -0.01,  # Want to reduce short position
        'GOOGL': 0.02   # Want to open new position
    }
    
    predictions = {
        'AAPL': 0.015,  # Small expected return
        'MSFT': 0.008,  # Small expected return
        'GOOGL': 0.050  # Large expected return
    }
    
    filtered = sizer.apply_transaction_cost_filter(target_positions, predictions)
    
    print(f"{'Symbol':<8} {'Current':<10} {'Target':<10} {'Filtered':<10} {'Pred Return':<12} {'Action'}")
    print("-" * 70)
    
    for symbol in target_positions:
        current = sizer.current_positions.get(symbol, 0.0)
        target = target_positions[symbol]
        final = filtered[symbol]
        pred = predictions[symbol]
        
        if abs(final - current) < 0.001:
            action = "NO TRADE (cost too high)"
        else:
            action = "TRADE"
        
        print(f"{symbol:<8} {current:+8.2%} {target:+8.2%} {final:+8.2%} {pred:+10.2%} {action}")


def main():
    """Main test execution."""
    print("🤖 REGRESSION POSITION SIZER TEST SUITE")
    print("=" * 60)
    
    try:
        test_position_sizing()
        test_dynamic_risk_scaling()
        test_transaction_cost_filter()
        
        print("\n✅ All tests completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("  • Position sizing based on predicted returns and confidence")
        print("  • Dynamic risk scaling based on performance and volatility")
        print("  • Transaction cost filtering to prevent over-trading")
        print("  • Portfolio-level constraints and exposure limits")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())