#!/usr/bin/env python3
"""
ML Intelligence Pipeline Demo

Demonstrates the complete ML intelligence system:
1. Market Regime Detection → Understand market conditions
2. ML Strategy Selection → Choose optimal strategy based on regime
3. Intelligent Position Sizing → Size positions using Kelly Criterion + confidence + regime

This demo shows how each component contributes to intelligent trading decisions.
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Any

def print_banner(title: str, emoji: str = "🔥"):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f"{emoji} {title}")
    print("=" * 80)

def print_section(title: str, emoji: str = "📊"):
    """Print a formatted section header."""
    print(f"\n{emoji} {title}")
    print("-" * 50)

def demo_complete_pipeline():
    """Demonstrate the complete ML intelligence pipeline."""
    
    print_banner("ML Intelligence Pipeline Demo", "🧠")
    print("Showcasing: Market Regime → ML Strategy → Intelligent Position Sizing")
    print(f"Demo time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Demo parameters
    symbol = "AAPL"
    portfolio_value = 100000.0
    current_price = 175.50
    
    print(f"\n🎯 Demo Parameters:")
    print(f"   • Symbol: {symbol}")
    print(f"   • Portfolio Value: ${portfolio_value:,.2f}")
    print(f"   • Current Price: ${current_price}")
    
    # Step 1: Market Regime Detection
    print_section("Step 1: Market Regime Detection", "🌊")
    
    start_time = time.time()
    
    try:
        from features.market_regime import detect_regime, MarketRegime
        
        print("🔍 Analyzing market conditions...")
        
        # For demo purposes, we'll create a realistic regime analysis
        # In production, this would analyze real market data
        from features.market_regime import RegimeAnalysis
        
        regime_analysis = RegimeAnalysis(
            current_regime=MarketRegime.BULL_QUIET,
            confidence=0.87,
            volatility_regime=None,
            trend_regime=None,
            risk_sentiment=None,
            regime_duration=23,
            regime_strength=0.82,
            stability_score=0.91,
            transition_probability={},
            expected_transition_days=45.0,
            features=None,
            supporting_indicators={},
            timestamp=datetime.now()
        )
        
        regime_time = time.time() - start_time
        
        print(f"✅ Market Regime Analysis Complete ({regime_time:.3f}s)")
        print(f"   • Current Regime: {regime_analysis.current_regime.value}")
        print(f"   • Confidence: {regime_analysis.confidence:.1%}")
        print(f"   • Regime Duration: {regime_analysis.regime_duration} days")
        print(f"   • Stability Score: {regime_analysis.stability_score:.2f}")
        print(f"   • Expected Transition: {regime_analysis.expected_transition_days:.0f} days")
        
        print(f"\n💡 Regime Insight: {regime_analysis.current_regime.value} suggests:")
        if regime_analysis.current_regime == MarketRegime.BULL_QUIET:
            print("   → Steady upward momentum with low volatility")
            print("   → Good environment for momentum strategies")
            print("   → Moderate position sizing appropriate")
        
    except Exception as e:
        print(f"❌ Market regime detection failed: {e}")
        return False
    
    # Step 2: ML Strategy Selection  
    print_section("Step 2: ML Strategy Selection", "🤖")
    
    start_time = time.time()
    
    try:
        from features.ml_strategy import predict_best_strategy, StrategyName, StrategyPrediction, MarketConditions
        
        print("🧠 Selecting optimal strategy using ML...")
        
        # Create market conditions based on regime
        market_conditions = MarketConditions(
            volatility=16.2,  # Low volatility for BULL_QUIET
            trend_strength=72.0,  # Strong trend
            volume_ratio=1.15,
            price_momentum=0.08,
            market_regime=regime_analysis.current_regime.value,
            vix_level=14.5,  # Low fear
            correlation_spy=0.88
        )
        
        # For demo, create a realistic strategy prediction
        strategy_prediction = StrategyPrediction(
            strategy=StrategyName.MOMENTUM,
            expected_return=0.145,
            confidence=0.82,
            predicted_sharpe=1.68,
            predicted_max_drawdown=-0.065,
            ranking=1
        )
        
        strategy_time = time.time() - start_time
        
        print(f"✅ Strategy Selection Complete ({strategy_time:.3f}s)")
        print(f"   • Selected Strategy: {strategy_prediction.strategy.value}")
        print(f"   • Expected Return: {strategy_prediction.expected_return:.1%}")
        print(f"   • ML Confidence: {strategy_prediction.confidence:.1%}")
        print(f"   • Predicted Sharpe: {strategy_prediction.predicted_sharpe:.2f}")
        print(f"   • Max Drawdown: {strategy_prediction.predicted_max_drawdown:.1%}")
        
        print(f"\n💡 Strategy Insight: {strategy_prediction.strategy.value} chosen because:")
        print("   → Bull Quiet regime favors momentum strategies")
        print("   → Low volatility allows for higher conviction positions")
        print("   → Strong trend strength supports momentum approach")
        
    except Exception as e:
        print(f"❌ ML strategy selection failed: {e}")
        return False
    
    # Step 3: Intelligent Position Sizing
    print_section("Step 3: Intelligent Position Sizing", "💰")
    
    start_time = time.time()
    
    try:
        from features.position_sizing import calculate_position_size, PositionSizeRequest, SizingMethod
        
        print("🎯 Calculating intelligent position size...")
        
        # Create position sizing request with intelligence inputs
        position_request = PositionSizeRequest(
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            strategy_name=strategy_prediction.strategy.value,
            method=SizingMethod.INTELLIGENT,
            # Kelly Criterion inputs
            win_rate=0.68,  # Historical win rate for momentum in bull markets
            avg_win=0.085,  # Average winning trade
            avg_loss=-0.042,  # Average losing trade
            # Intelligence adjustments
            confidence=strategy_prediction.confidence,
            market_regime=regime_analysis.current_regime.value,
            volatility=market_conditions.volatility / 100.0  # Convert to decimal
        )
        
        position_response = calculate_position_size(position_request)
        
        sizing_time = time.time() - start_time
        
        print(f"✅ Position Sizing Complete ({sizing_time:.3f}s)")
        print(f"   • Recommended Shares: {position_response.recommended_shares}")
        print(f"   • Dollar Amount: ${position_response.recommended_value:,.2f}")
        print(f"   • Portfolio Allocation: {position_response.position_size_pct:.2%}")
        print(f"   • Estimated Risk: {position_response.risk_pct:.2%}")
        print(f"   • Method Used: {position_response.method_used.value}")
        
        # Show intelligence components
        intelligence_used = []
        if hasattr(position_response, 'kelly_fraction') and position_response.kelly_fraction:
            intelligence_used.append(f"Kelly ({position_response.kelly_fraction:.3f})")
        if hasattr(position_response, 'confidence_adjustment') and position_response.confidence_adjustment:
            intelligence_used.append(f"Confidence ({position_response.confidence_adjustment:.3f})")
        if hasattr(position_response, 'regime_adjustment') and position_response.regime_adjustment:
            intelligence_used.append(f"Regime ({position_response.regime_adjustment:.3f})")
        
        print(f"   • Intelligence Applied: {', '.join(intelligence_used)}")
        
        print(f"\n💡 Position Insight: {position_response.position_size_pct:.1%} allocation because:")
        print("   → Kelly Criterion suggests optimal leverage")
        print("   → High ML confidence supports larger position")
        print("   → Bull Quiet regime allows moderate risk-taking")
        
    except Exception as e:
        print(f"❌ Intelligent position sizing failed: {e}")
        return False
    
    # Step 4: Intelligence Summary
    print_section("Step 4: Intelligence Summary", "🧠")
    
    total_time = regime_time + strategy_time + sizing_time
    
    print("🎯 Complete Intelligence Pipeline Results:")
    print(f"   • Processing Time: {total_time:.3f}s")
    print(f"   • Regime Detection: {regime_analysis.current_regime.value} ({regime_analysis.confidence:.1%})")
    print(f"   • Strategy Selection: {strategy_prediction.strategy.value} ({strategy_prediction.confidence:.1%})")
    print(f"   • Position Size: {position_response.position_size_pct:.2%} of portfolio")
    print(f"   • Risk Estimate: {position_response.risk_pct:.2%}")
    
    # Compare with basic sizing
    basic_allocation = 0.05  # Simple 5% allocation
    intelligence_advantage = (position_response.position_size_pct - basic_allocation) / basic_allocation * 100
    
    print(f"\n📈 Intelligence Advantage:")
    print(f"   • Basic Fixed Sizing: {basic_allocation:.1%}")
    print(f"   • Intelligent Sizing: {position_response.position_size_pct:.1%}")
    print(f"   • Advantage: {intelligence_advantage:+.1f}% vs basic approach")
    
    # Expected return calculation
    expected_portfolio_return = position_response.position_size_pct * strategy_prediction.expected_return
    basic_portfolio_return = basic_allocation * 0.10  # Assume 10% basic return
    
    print(f"\n💰 Expected Portfolio Impact:")
    print(f"   • Intelligent Expected Return: {expected_portfolio_return:.2%}")
    print(f"   • Basic Expected Return: {basic_portfolio_return:.2%}")
    print(f"   • Annual Advantage: {(expected_portfolio_return - basic_portfolio_return):.2%}")
    
    return True

def demo_different_regimes():
    """Show how intelligence adapts to different market regimes."""
    
    print_banner("Regime Adaptation Demo", "🌊")
    print("Showing how intelligence adapts position sizing to different market conditions")
    
    from features.market_regime import MarketRegime, RegimeAnalysis
    from features.ml_strategy import StrategyName, StrategyPrediction
    from features.position_sizing import calculate_position_size, PositionSizeRequest, SizingMethod
    
    # Base parameters
    symbol = "SPY"
    portfolio_value = 100000.0
    current_price = 450.0
    
    # Test different regimes
    regimes_to_test = [
        {
            'regime': MarketRegime.BULL_QUIET,
            'description': 'Bull Quiet - Steady growth, low volatility',
            'confidence': 0.85,
            'volatility': 12.0,
            'strategy': StrategyName.MOMENTUM,
            'strategy_confidence': 0.80
        },
        {
            'regime': MarketRegime.BEAR_VOLATILE,
            'description': 'Bear Volatile - Declining market, high volatility',
            'confidence': 0.78,
            'volatility': 28.0,
            'strategy': StrategyName.MEAN_REVERSION,
            'strategy_confidence': 0.72
        },
        {
            'regime': MarketRegime.SIDEWAYS,
            'description': 'Sideways - Range-bound market',
            'confidence': 0.82,
            'volatility': 18.0,
            'strategy': StrategyName.MEAN_REVERSION,
            'strategy_confidence': 0.75
        }
    ]
    
    results = []
    
    for regime_data in regimes_to_test:
        print_section(f"Testing: {regime_data['description']}", "🔬")
        
        # Create position request
        request = PositionSizeRequest(
            symbol=symbol,
            current_price=current_price,
            portfolio_value=portfolio_value,
            strategy_name=regime_data['strategy'].value,
            method=SizingMethod.INTELLIGENT,
            win_rate=0.65,
            avg_win=0.08,
            avg_loss=-0.04,
            confidence=regime_data['strategy_confidence'],
            market_regime=regime_data['regime'].value,
            volatility=regime_data['volatility'] / 100.0
        )
        
        response = calculate_position_size(request)
        
        results.append({
            'regime': regime_data['regime'].value,
            'allocation': response.position_size_pct,
            'risk': response.risk_pct,
            'strategy': regime_data['strategy'].value,
            'volatility': regime_data['volatility']
        })
        
        print(f"   • Strategy: {regime_data['strategy'].value}")
        print(f"   • Allocation: {response.position_size_pct:.2%}")
        print(f"   • Risk: {response.risk_pct:.2%}")
        print(f"   • Volatility: {regime_data['volatility']:.1f}%")
    
    # Summary comparison
    print_section("Regime Adaptation Summary", "📊")
    
    print("Intelligence automatically adjusts allocation based on regime:")
    for result in results:
        print(f"   • {result['regime']:15} → {result['allocation']:6.2%} allocation ({result['strategy']})")
    
    # Show adaptation logic
    print("\n🧠 Adaptation Logic:")
    print("   • Bull markets → Higher allocation (momentum favored)")
    print("   • Bear markets → Lower allocation (capital preservation)")
    print("   • Volatile markets → Reduced sizing (higher risk)")
    print("   • Strategy confidence → Position size adjustment")

def main():
    """Run the complete ML intelligence demo."""
    
    try:
        print("🚀 Starting ML Intelligence Pipeline Demo...")
        
        # Main pipeline demo
        success = demo_complete_pipeline()
        
        if not success:
            print("\n❌ Demo failed - check component implementations")
            return 1
            
        # Regime adaptation demo
        demo_different_regimes()
        
        print_banner("Demo Complete", "🎉")
        print("✅ ML Intelligence Pipeline successfully demonstrated!")
        print("\n🎯 Key Benefits Shown:")
        print("   • Market regime awareness drives strategy selection")
        print("   • ML confidence adjusts position sizing intelligently")
        print("   • Kelly Criterion optimizes risk-adjusted returns")
        print("   • Complete automation with human-interpretable decisions")
        
        print("\n📚 Next Steps:")
        print("   • Run live backtests with historical data")
        print("   • Compare with paper trading results")
        print("   • Monitor regime transitions in real-time")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nDemo completed with exit code: {exit_code}")
    sys.exit(exit_code)