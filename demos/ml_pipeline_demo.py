#!/usr/bin/env python3
"""
ML Pipeline Integration Demo

This demo showcases the fully integrated ML pipeline with:
- Market regime detection
- ML strategy selection
- Confidence-based position sizing
- Risk-adjusted trading decisions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
from bot_v2.orchestration.enhanced_orchestrator import create_enhanced_orchestrator
from bot_v2.orchestration.types import TradingMode, OrchestratorConfig


def run_ml_pipeline_demo():
    """Demonstrate the fully integrated ML pipeline"""
    
    print("=" * 80)
    print("🤖 GPT-Trader ML Pipeline Integration Demo")
    print("=" * 80)
    print()
    
    # Configuration
    config = OrchestratorConfig(
        mode=TradingMode.BACKTEST,
        capital=100000,
        enable_ml_strategy=True,
        enable_regime_detection=True
    )
    # Add ML-specific configs after initialization
    config.min_confidence = 0.6
    config.max_position_pct = 0.15
    
    print("📊 Configuration:")
    print(f"  - Mode: {config.mode.value}")
    print(f"  - Capital: ${config.capital:,.2f}")
    print(f"  - Min Confidence: {config.min_confidence:.1%}")
    print(f"  - Max Position: {config.max_position_pct:.1%}")
    print()
    
    # Create enhanced orchestrator with ML
    print("🚀 Initializing Enhanced Orchestrator with ML Pipeline...")
    orchestrator = create_enhanced_orchestrator(config)
    
    # Check available slices
    print(f"✅ Available slices: {list(orchestrator.available_slices.keys())}")
    if orchestrator.failed_slices:
        print(f"⚠️  Failed slices: {list(orchestrator.failed_slices.keys())}")
    print()
    
    # Define portfolio of symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    current_positions = {
        'AAPL': 0.1,   # 10% position
        'GOOGL': 0.05   # 5% position
    }
    
    print(f"📈 Analyzing portfolio: {symbols}")
    print(f"📊 Current positions: {current_positions}")
    print()
    
    # Execute ML trading cycle
    print("=" * 80)
    print("🧠 Executing ML-Driven Trading Cycle")
    print("=" * 80)
    
    results = orchestrator.execute_ml_trading_cycle(
        symbols=symbols,
        current_positions=current_positions
    )
    
    # Display results for each symbol
    for symbol, result in results.items():
        print(f"\n📊 {symbol} Analysis:")
        print("-" * 40)
        
        if result.success:
            data = result.data
            
            # Display ML decision details
            if 'strategy' in data:
                print(f"  Strategy: {data['strategy']}")
            if 'confidence' in data:
                print(f"  Confidence: {data['confidence']:.1%}")
            if 'expected_return' in data:
                print(f"  Expected Return: {data['expected_return']:.1%}")
            if 'regime' in data:
                print(f"  Market Regime: {data['regime']}")
            if 'action' in data:
                print(f"  Decision: {data['action'].upper()}")
            if 'position_size' in data:
                print(f"  Position Size: {data['position_size']:.1%}")
            if 'shares' in data and data.get('shares', 0) > 0:
                print(f"  Shares: {data['shares']}")
            if 'capital' in data and data.get('capital', 0) > 0:
                print(f"  Capital: ${data['capital']:,.2f}")
            
            # Display reasoning
            if 'reasoning' in data and data['reasoning']:
                print(f"  Reasoning:")
                for reason in data['reasoning']:
                    print(f"    - {reason}")
        else:
            print(f"  ❌ Error: {', '.join(result.errors)}")
    
    # Display ML performance metrics
    print("\n" + "=" * 80)
    print("📊 ML Performance Report")
    print("=" * 80)
    
    report = orchestrator.get_ml_performance_report()
    
    metrics = report['metrics']
    print(f"\n📈 Decision Statistics:")
    print(f"  - Total Decisions: {metrics['total_decisions']}")
    print(f"  - Buy Signals: {metrics['buy_signals']}")
    print(f"  - Sell Signals: {metrics['sell_signals']}")
    print(f"  - Hold Signals: {metrics['hold_signals']}")
    print(f"  - Average Confidence: {metrics['avg_confidence']:.1%}")
    
    if report['recent_decisions']:
        print(f"\n🕐 Recent Decisions:")
        for decision in report['recent_decisions'][:5]:
            print(f"  - {decision['symbol']}: {decision['decision']} "
                  f"({decision['strategy']}, {decision['confidence']:.1%})")
    
    print(f"\n💾 Cache Status:")
    print(f"  - Enabled: {report['cache_status']['enabled']}")
    print(f"  - TTL: {report['cache_status']['ttl_minutes']} minutes")
    
    # Demonstrate cache clearing
    print("\n🔄 Clearing ML cache...")
    orchestrator.clear_ml_cache()
    print("✅ Cache cleared")
    
    print("\n" + "=" * 80)
    print("✅ ML Pipeline Integration Demo Complete!")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Market regime detection for each symbol")
    print("  ✓ ML-driven strategy selection with confidence scores")
    print("  ✓ Regime-aware position sizing")
    print("  ✓ Risk-adjusted portfolio decisions")
    print("  ✓ Confidence-based trade filtering")
    print("  ✓ ML prediction caching for performance")
    print("  ✓ Comprehensive decision reasoning")
    print()
    print("🎯 The ML pipeline is now fully integrated and operational!")


def demonstrate_ml_components():
    """Demonstrate individual ML components"""
    
    print("\n" + "=" * 80)
    print("🔬 Individual ML Component Testing")
    print("=" * 80)
    
    from bot_v2.orchestration.ml_integration import create_ml_integrator
    
    # Create ML integrator
    ml_config = {
        'min_confidence': 0.65,
        'max_position_size': 0.20,
        'enable_caching': True,
        'cache_ttl_minutes': 5
    }
    
    integrator = create_ml_integrator(ml_config)
    
    print("\n📊 Testing ML Decision Making for AAPL:")
    print("-" * 40)
    
    # Make a trading decision
    decision = integrator.make_trading_decision(
        symbol='AAPL',
        portfolio_value=100000,
        current_positions={'AAPL': 0.05}  # 5% existing position
    )
    
    print(f"  Symbol: {decision.symbol}")
    print(f"  Strategy: {decision.strategy}")
    print(f"  Confidence: {decision.confidence:.1%}")
    print(f"  Expected Return: {decision.expected_return:.1%}")
    print(f"  Market Regime: {decision.regime}")
    print(f"  Regime Confidence: {decision.regime_confidence:.1%}")
    print(f"  Base Position Size: {decision.position_size:.1%}")
    print(f"  Risk-Adjusted Size: {decision.risk_adjusted_size:.1%}")
    print(f"  Decision: {decision.decision.upper()}")
    print(f"  Reasoning:")
    for reason in decision.reasoning:
        print(f"    - {reason}")
    
    # Test portfolio-wide decisions
    print("\n📊 Portfolio-Wide ML Analysis:")
    print("-" * 40)
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    decisions = integrator.get_portfolio_ml_decisions(
        symbols=symbols,
        portfolio_value=100000,
        current_positions={'AAPL': 0.1}
    )
    
    print(f"  Analyzed {len(decisions)} symbols")
    print(f"  Sorted by confidence * expected_return:")
    for i, dec in enumerate(decisions[:3], 1):
        score = dec.confidence * dec.expected_return
        print(f"    {i}. {dec.symbol}: {dec.strategy} "
              f"(score: {score:.3f}, action: {dec.decision})")


if __name__ == "__main__":
    try:
        # Run main demo
        run_ml_pipeline_demo()
        
        # Run component testing
        demonstrate_ml_components()
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)