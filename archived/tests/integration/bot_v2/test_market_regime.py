#!/usr/bin/env python3
"""
Test Market Regime Detection system - Week 3 of Path B implementation.
"""

import sys
from datetime import datetime, timedelta
sys.path.append('/Users/rj/PycharmProjects/GPT-Trader/src/bot_v2')


def test_regime_detection():
    """Test basic regime detection functionality."""
    print("="*80)
    print("MARKET REGIME DETECTION - BASIC TEST")
    print("="*80)
    
    from features.market_regime import detect_regime
    
    symbol = 'AAPL'
    lookback_days = 60
    
    print(f"\n🔍 Detecting regime for {symbol} ({lookback_days} days lookback)...")
    
    try:
        analysis = detect_regime(symbol, lookback_days)
        
        print(f"\n📊 Regime Analysis Results:")
        print(f"   Current Regime: {analysis.current_regime.value}")
        print(f"   Confidence: {analysis.confidence:.1%}")
        print(f"   Duration: {analysis.regime_duration} days")
        print(f"   Stability: {analysis.stability_score:.1%}")
        
        print(f"\n🧩 Component Regimes:")
        print(f"   Volatility: {analysis.volatility_regime.value}")
        print(f"   Trend: {analysis.trend_regime.value}")
        print(f"   Risk Sentiment: {analysis.risk_sentiment.value}")
        
        print(f"\n📈 Key Features:")
        print(f"   20d Return: {analysis.features.returns_20d:.2%}")
        print(f"   30d Volatility: {analysis.features.realized_vol_30d:.1f}%")
        print(f"   Volume Ratio: {analysis.features.volume_ratio:.2f}")
        print(f"   Trend Strength: {analysis.features.trend_strength:.1f}")
        
        print(f"\n🔮 Transition Probabilities:")
        for regime, prob in analysis.transition_probability.items():
            if prob > 0.05:  # Only show significant probabilities
                print(f"   {regime.value}: {prob:.1%}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Regime detection failed: {e}")
        return None


def test_regime_monitoring():
    """Test real-time regime monitoring."""
    print("\n" + "="*80)
    print("MARKET REGIME DETECTION - MONITORING TEST")
    print("="*80)
    
    from features.market_regime import monitor_regime_changes
    from features.market_regime.types import RegimeAlert
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    def alert_callback(alert: RegimeAlert):
        print(f"\n🚨 REGIME CHANGE ALERT:")
        print(f"   Symbol: {alert.symbol}")
        print(f"   Change: {alert.old_regime.value} → {alert.new_regime.value}")
        print(f"   Confidence: {alert.confidence:.1%}")
        print(f"   Severity: {alert.severity}")
        print(f"   Message: {alert.message}")
    
    print(f"\n📡 Starting monitoring for {len(symbols)} symbols...")
    
    try:
        monitor_state = monitor_regime_changes(
            symbols=symbols,
            callback=alert_callback,
            check_interval=300,  # 5 minutes
            alert_on_change=True
        )
        
        print(f"\n✅ Monitoring started successfully!")
        print(f"   Symbols: {', '.join(monitor_state.symbols)}")
        print(f"   Check interval: {monitor_state.check_interval_seconds}s")
        print(f"   Current regimes:")
        
        for symbol, regime in monitor_state.current_regimes.items():
            print(f"     {symbol}: {regime.value}")
        
        return monitor_state
        
    except Exception as e:
        print(f"❌ Monitoring setup failed: {e}")
        return None


def test_regime_history():
    """Test historical regime analysis."""
    print("\n" + "="*80)
    print("MARKET REGIME DETECTION - HISTORY TEST")
    print("="*80)
    
    from features.market_regime import get_regime_history
    
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year
    
    print(f"\n📊 Analyzing regime history for {symbol}")
    print(f"📅 Period: {start_date.date()} to {end_date.date()}")
    
    try:
        history = get_regime_history(symbol, start_date, end_date)
        
        print(f"\n📈 Historical Analysis:")
        print(f"   Total regime periods: {len(history.regimes)}")
        print(f"   Total transitions: {len(history.transitions)}")
        
        print(f"\n⏱️ Average Duration by Regime:")
        for regime, duration in history.average_duration.items():
            if duration > 0:
                print(f"   {regime.value}: {duration:.1f} days")
        
        print(f"\n📊 Returns by Regime:")
        for regime, returns in history.returns_by_regime.items():
            sharpe = history.sharpe_by_regime.get(regime, 0)
            print(f"   {regime.value}: {returns:.1%} (Sharpe: {sharpe:.2f})")
        
        print(f"\n🔄 Most Common Transitions:")
        if history.transitions:
            transition_counts = {}
            for trans in history.transitions:
                key = f"{trans.from_regime.value} → {trans.to_regime.value}"
                transition_counts[key] = transition_counts.get(key, 0) + 1
            
            sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
            for trans, count in sorted_transitions[:5]:
                print(f"   {trans}: {count} times")
        
        return history
        
    except Exception as e:
        print(f"❌ History analysis failed: {e}")
        return None


def test_regime_prediction():
    """Test regime change prediction."""
    print("\n" + "="*80)
    print("MARKET REGIME DETECTION - PREDICTION TEST")
    print("="*80)
    
    from features.market_regime import predict_regime_change
    
    symbol = 'AAPL'
    horizon_days = 5
    
    
    print(f"\n🔮 Predicting regime change for {symbol}")
    print(f"🎯 Prediction horizon: {horizon_days} days")
    
    try:
        prediction = predict_regime_change(symbol, horizon_days)
        
        print(f"\n📊 Regime Change Prediction:")
        print(f"   Current Regime: {prediction.current_regime.value}")
        print(f"   Most Likely Next: {prediction.most_likely_next.value}")
        print(f"   Change Probability: {prediction.change_probability:.1%}")
        print(f"   Confidence: {prediction.confidence:.1%}")
        
        print(f"\n🎲 All Regime Probabilities:")
        sorted_probs = sorted(prediction.regime_probabilities.items(), 
                            key=lambda x: x[1], reverse=True)
        for regime, prob in sorted_probs:
            if prob > 0.01:  # Only show > 1% probability
                print(f"   {regime.value}: {prob:.1%}")
        
        print(f"\n🔍 Leading Indicators:")
        for indicator in prediction.leading_indicators:
            print(f"   • {indicator}")
        
        print(f"\n✅ Confirming Indicators:")
        for indicator in prediction.confirming_indicators:
            print(f"   • {indicator}")
        
        return prediction
        
    except Exception as e:
        print(f"❌ Regime prediction failed: {e}")
        return None


def test_multi_symbol_analysis():
    """Test regime detection across multiple symbols."""
    print("\n" + "="*80)
    print("MARKET REGIME DETECTION - MULTI-SYMBOL TEST")
    print("="*80)
    
    from features.market_regime import detect_regime
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    print(f"\n🔍 Analyzing regimes for {len(symbols)} symbols...")
    
    results = {}
    
    for symbol in symbols:
        try:
            analysis = detect_regime(symbol, lookback_days=30)
            results[symbol] = analysis
            
            print(f"\n📊 {symbol}:")
            print(f"   Regime: {analysis.current_regime.value} ({analysis.confidence:.0%})")
            print(f"   Volatility: {analysis.volatility_regime.value}")
            print(f"   Trend: {analysis.trend_regime.value}")
            
        except Exception as e:
            print(f"❌ {symbol} failed: {e}")
            results[symbol] = None
    
    # Analyze market-wide patterns
    print(f"\n🌍 Market-Wide Analysis:")
    
    successful_analyses = [r for r in results.values() if r is not None]
    
    if successful_analyses:
        # Regime distribution
        regime_counts = {}
        for analysis in successful_analyses:
            regime = analysis.current_regime
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        print(f"   Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(successful_analyses) * 100
            print(f"     {regime.value}: {count}/{len(successful_analyses)} ({pct:.0f}%)")
        
        # Average confidence
        avg_confidence = sum(a.confidence for a in successful_analyses) / len(successful_analyses)
        print(f"   Average Confidence: {avg_confidence:.1%}")
        
        # Volatility levels
        avg_volatility = sum(a.features.realized_vol_30d for a in successful_analyses) / len(successful_analyses)
        print(f"   Average Volatility: {avg_volatility:.1f}%")
    
    return results


def test_regime_stability():
    """Test regime stability analysis."""
    print("\n" + "="*80)
    print("MARKET REGIME DETECTION - STABILITY TEST")
    print("="*80)
    
    from features.market_regime import analyze_regime_stability
    from features.market_regime.types import MarketRegime
    
    symbol = 'AAPL'
    
    print(f"\n⚖️ Analyzing regime stability for {symbol}...")
    
    # Test current regime stability
    try:
        current_stability = analyze_regime_stability(symbol)
        print(f"   Current regime stability: {current_stability:.1%}")
        
        # Test stability for different hypothetical regimes
        print(f"\n🧪 Hypothetical Regime Stability:")
        
        test_regimes = [
            MarketRegime.BULL_QUIET,
            MarketRegime.BEAR_VOLATILE,
            MarketRegime.SIDEWAYS_QUIET,
            MarketRegime.CRISIS
        ]
        
        for regime in test_regimes:
            stability = analyze_regime_stability(symbol, regime)
            print(f"   {regime.value}: {stability:.1%}")
        
        return current_stability
        
    except Exception as e:
        print(f"❌ Stability analysis failed: {e}")
        return None

