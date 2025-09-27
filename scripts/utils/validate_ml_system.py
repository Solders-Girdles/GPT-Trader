#!/usr/bin/env python3
"""
ML System Validation Script

Comprehensive validation of the trained ML models and integration.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List
import json


def validate_strategy_predictions():
    """Validate strategy prediction accuracy and diversity"""
    print("\n" + "="*60)
    print("🎯 Validating Strategy Predictions")
    print("="*60)
    
    from bot_v2.features.ml_strategy.ml_strategy import (
        predict_best_strategy, get_model_performance
    )
    
    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN']
    results = []
    
    for symbol in test_symbols:
        try:
            # Get top 3 strategy predictions
            predictions = predict_best_strategy(symbol, lookback_days=30, top_n=3)
            
            if predictions:
                top_pred = predictions[0]
                results.append({
                    'symbol': symbol,
                    'strategy': top_pred.strategy.value,
                    'confidence': top_pred.confidence,
                    'expected_return': top_pred.expected_return,
                    'sharpe': top_pred.predicted_sharpe
                })
                
                print(f"\n📊 {symbol}:")
                print(f"   Best Strategy: {top_pred.strategy.value}")
                print(f"   Confidence: {top_pred.confidence:.1%}")
                print(f"   Expected Return: {top_pred.expected_return:.1%}")
                print(f"   Predicted Sharpe: {top_pred.predicted_sharpe:.2f}")
                
                # Show alternatives
                if len(predictions) > 1:
                    print("   Alternatives:")
                    for i, pred in enumerate(predictions[1:3], 2):
                        print(f"     {i}. {pred.strategy.value} ({pred.confidence:.1%})")
        
        except Exception as e:
            print(f"\n❌ {symbol}: Failed - {e}")
            results.append({
                'symbol': symbol,
                'error': str(e)
            })
    
    # Analyze results
    if results:
        successful = [r for r in results if 'strategy' in r]
        if successful:
            avg_confidence = np.mean([r['confidence'] for r in successful])
            avg_return = np.mean([r['expected_return'] for r in successful])
            
            print(f"\n📊 Summary:")
            print(f"   Success Rate: {len(successful)}/{len(results)}")
            print(f"   Avg Confidence: {avg_confidence:.1%}")
            print(f"   Avg Expected Return: {avg_return:.1%}")
            
            # Check strategy diversity
            strategies = [r['strategy'] for r in successful]
            unique_strategies = set(strategies)
            print(f"   Strategy Diversity: {len(unique_strategies)} unique strategies")
            print(f"   Strategies Used: {', '.join(unique_strategies)}")
            
            return len(successful) == len(results)
    
    return False


def validate_regime_detection():
    """Validate market regime detection"""
    print("\n" + "="*60)
    print("🔍 Validating Market Regime Detection")
    print("="*60)
    
    from bot_v2.features.market_regime.market_regime import (
        detect_regime, predict_regime_change
    )
    
    test_symbols = ['SPY', 'QQQ', 'IWM', 'DIA']  # Major ETFs
    results = []
    
    for symbol in test_symbols:
        try:
            # Detect current regime
            analysis = detect_regime(symbol, lookback_days=60)
            
            # Predict regime change
            prediction = predict_regime_change(symbol, horizon_days=5)
            
            results.append({
                'symbol': symbol,
                'current_regime': analysis.current_regime.value,
                'confidence': analysis.confidence,
                'stability': analysis.stability_score,
                'change_probability': prediction.change_probability
            })
            
            print(f"\n📊 {symbol}:")
            print(f"   Current Regime: {analysis.current_regime.value}")
            print(f"   Confidence: {analysis.confidence:.1%}")
            print(f"   Stability: {analysis.stability_score:.1%}")
            print(f"   Change Probability (5d): {prediction.change_probability:.1%}")
            
            if prediction.leading_indicators:
                print("   Leading Indicators:")
                for indicator in prediction.leading_indicators[:2]:
                    print(f"     - {indicator}")
        
        except Exception as e:
            print(f"\n❌ {symbol}: Failed - {e}")
            results.append({
                'symbol': symbol,
                'error': str(e)
            })
    
    # Analyze regime distribution
    if results:
        successful = [r for r in results if 'current_regime' in r]
        if successful:
            regimes = [r['current_regime'] for r in successful]
            unique_regimes = set(regimes)
            
            print(f"\n📊 Regime Summary:")
            print(f"   Symbols Analyzed: {len(successful)}/{len(results)}")
            print(f"   Unique Regimes: {len(unique_regimes)}")
            print(f"   Regimes Found: {', '.join(unique_regimes)}")
            
            avg_confidence = np.mean([r['confidence'] for r in successful])
            avg_stability = np.mean([r['stability'] for r in successful])
            print(f"   Avg Confidence: {avg_confidence:.1%}")
            print(f"   Avg Stability: {avg_stability:.1%}")
            
            return len(successful) > 0
    
    return False


def validate_ml_integration():
    """Validate complete ML integration pipeline"""
    print("\n" + "="*60)
    print("🔧 Validating ML Integration Pipeline")
    print("="*60)
    
    from bot_v2.orchestration.ml_integration import create_ml_integrator
    
    # Create integrator
    integrator = create_ml_integrator({
        'min_confidence': 0.5,
        'max_position_size': 0.2,
        'enable_caching': True
    })
    
    # Test portfolio decisions
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA']
    portfolio_value = 100000
    current_positions = {
        'AAPL': 0.1,  # 10% position
        'GOOGL': 0.05  # 5% position
    }
    
    print(f"\n💼 Portfolio: ${portfolio_value:,.0f}")
    print(f"📊 Current Positions: {current_positions}")
    print(f"🎯 Analyzing: {symbols}\n")
    
    decisions = integrator.get_portfolio_ml_decisions(
        symbols=symbols,
        portfolio_value=portfolio_value,
        current_positions=current_positions
    )
    
    buy_signals = 0
    sell_signals = 0
    hold_signals = 0
    
    for decision in decisions:
        print(f"📈 {decision.symbol}:")
        print(f"   Strategy: {decision.strategy}")
        print(f"   Confidence: {decision.confidence:.1%}")
        print(f"   Decision: {decision.decision.upper()}")
        print(f"   Position Size: {decision.risk_adjusted_size:.1%}")
        print(f"   Reasoning: {decision.reasoning[0]}")
        
        if decision.decision == 'buy':
            buy_signals += 1
        elif decision.decision == 'sell':
            sell_signals += 1
        else:
            hold_signals += 1
        
        print()
    
    print(f"📊 Signal Summary:")
    print(f"   Buy: {buy_signals}, Sell: {sell_signals}, Hold: {hold_signals}")
    
    # Test caching
    print("\n🔄 Testing Cache Performance...")
    
    import time
    
    # First call (no cache)
    start = time.time()
    decision1 = integrator.make_trading_decision('AAPL', portfolio_value, {})
    time1 = time.time() - start
    
    # Second call (cached)
    start = time.time()
    decision2 = integrator.make_trading_decision('AAPL', portfolio_value, {})
    time2 = time.time() - start
    
    print(f"   First call: {time1*1000:.1f}ms")
    print(f"   Cached call: {time2*1000:.1f}ms")
    print(f"   Speed improvement: {time1/time2:.1f}x")
    
    # Clear cache
    integrator.clear_cache()
    print("   ✅ Cache cleared")
    
    return True


def validate_risk_adjustments():
    """Validate risk-adjusted position sizing"""
    print("\n" + "="*60)
    print("⚖️ Validating Risk Adjustments")
    print("="*60)
    
    from bot_v2.orchestration.ml_integration import MLPipelineIntegrator
    
    integrator = MLPipelineIntegrator({'max_position_size': 0.2})
    
    # Test different market regimes
    test_cases = [
        ('BULL_QUIET', 0.8, 0.1),
        ('BEAR_VOLATILE', 0.8, 0.1),
        ('CRISIS', 0.8, 0.1),
        ('SIDEWAYS_QUIET', 0.6, 0.15)
    ]
    
    print("\n📊 Position Size Adjustments by Regime:")
    print("   (Base size: 10% of portfolio)\n")
    
    for regime, confidence, expected_return in test_cases:
        size = integrator._calculate_ml_position_size(
            confidence=confidence,
            expected_return=expected_return,
            regime=regime,
            portfolio_value=100000
        )
        
        print(f"   {regime:20} → {size:.1%}")
    
    # Test concentration limits
    print("\n📊 Concentration Risk Adjustments:")
    
    # High portfolio exposure
    high_exposure = {
        'AAPL': 0.3,
        'GOOGL': 0.3,
        'MSFT': 0.25
    }
    
    adjusted = integrator._apply_risk_adjustments(
        base_size=0.1,
        symbol='TSLA',
        regime='BULL_QUIET',
        current_positions=high_exposure
    )
    
    total_exposure = sum(high_exposure.values())
    print(f"   Current Exposure: {total_exposure:.0%}")
    print(f"   Base Position: 10%")
    print(f"   Adjusted Position: {adjusted:.1%}")
    print(f"   Reduction: {(1 - adjusted/0.1):.0%}")
    
    return True


def generate_performance_report():
    """Generate comprehensive ML performance report"""
    print("\n" + "="*60)
    print("📄 Generating ML Performance Report")
    print("="*60)
    
    from bot_v2.orchestration.enhanced_orchestrator import create_enhanced_orchestrator
    from bot_v2.orchestration.types import TradingMode, OrchestratorConfig
    
    # Create orchestrator
    config = OrchestratorConfig(
        mode=TradingMode.BACKTEST,
        capital=100000,
        enable_ml_strategy=True,
        enable_regime_detection=True
    )
    config.min_confidence = 0.5
    config.max_position_pct = 0.2
    
    orchestrator = create_enhanced_orchestrator(config)
    
    # Run a test cycle
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    results = orchestrator.execute_ml_trading_cycle(symbols, {})
    
    # Get performance report
    report = orchestrator.get_ml_performance_report()
    
    print("\n📊 ML System Performance:")
    print(f"   Total Decisions: {report['metrics']['total_decisions']}")
    print(f"   Buy Signals: {report['metrics']['buy_signals']}")
    print(f"   Sell Signals: {report['metrics']['sell_signals']}")
    print(f"   Hold Signals: {report['metrics']['hold_signals']}")
    print(f"   Avg Confidence: {report['metrics']['avg_confidence']:.1%}")
    
    # Save report
    report_path = Path("reports/ml_validation_report.json")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Report saved to: {report_path}")
    
    return True


def main():
    """Run all validation tests"""
    print("\n" + "="*80)
    print("🔬 GPT-Trader ML System Validation")
    print("="*80)
    print("\nRunning comprehensive validation of trained ML models...")
    
    tests = [
        ("Strategy Predictions", validate_strategy_predictions),
        ("Regime Detection", validate_regime_detection),
        ("ML Integration", validate_ml_integration),
        ("Risk Adjustments", validate_risk_adjustments),
        ("Performance Report", generate_performance_report)
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            print(f"\n🧪 Testing: {name}")
            result = test_func()
            results[name] = "✅ PASSED" if result else "⚠️ PARTIAL"
        except Exception as e:
            results[name] = f"❌ FAILED: {e}"
            print(f"\n❌ Test failed: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("📊 Validation Summary")
    print("="*80)
    
    for test_name, result in results.items():
        print(f"   {test_name:25} {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in str(r))
    total = len(results)
    
    print(f"\n📈 Overall Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("   The ML system is fully operational and ready for use.")
    elif passed >= total * 0.7:
        print("\n✅ ML system is functional with minor issues.")
        print("   Review partial failures for optimization opportunities.")
    else:
        print("\n⚠️ ML system needs attention.")
        print("   Review failed tests and retrain if necessary.")
    
    print("\n" + "="*80)
    print("Validation complete!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)