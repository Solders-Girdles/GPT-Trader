#!/usr/bin/env python3
"""
Portfolio Optimization with ML Intelligence

Practical example showing how to use the complete ML intelligence system
for multi-asset portfolio optimization:

1. Analyze market regimes across different assets
2. Select optimal strategies for each asset class
3. Calculate intelligent position sizes for the entire portfolio
4. Demonstrate regime-based asset rotation
5. Show portfolio-level risk management

This example shows real-world application of the ML intelligence pipeline.
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

def print_banner(title: str, emoji: str = "🔥"):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f"{emoji} {title}")
    print("=" * 80)

def print_section(title: str, emoji: str = "📊"):
    """Print a formatted section header."""
    print(f"\n{emoji} {title}")
    print("-" * 60)

class PortfolioOptimizer:
    """Intelligent portfolio optimizer using ML intelligence system."""
    
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.positions = {}
        self.regime_history = {}
        
    def analyze_asset_regime(self, symbol: str, asset_class: str) -> Dict:
        """Analyze market regime for a specific asset."""
        
        try:
            from features.market_regime import detect_regime, MarketRegime, RegimeAnalysis
            
            # For demo, create realistic regime analysis based on asset class
            regime_map = {
                'tech': MarketRegime.BULL_QUIET,
                'defensive': MarketRegime.SIDEWAYS,
                'growth': MarketRegime.BULL_VOLATILE,
                'value': MarketRegime.BULL_QUIET,
                'commodities': MarketRegime.BEAR_VOLATILE,
                'bonds': MarketRegime.BEAR_QUIET
            }
            
            confidence_map = {
                'tech': 0.82,
                'defensive': 0.78,
                'growth': 0.75,
                'value': 0.85,
                'commodities': 0.72,
                'bonds': 0.88
            }
            
            regime = regime_map.get(asset_class, MarketRegime.SIDEWAYS)
            confidence = confidence_map.get(asset_class, 0.75)
            
            regime_analysis = RegimeAnalysis(
                current_regime=regime,
                confidence=confidence,
                volatility_regime=None,
                trend_regime=None,
                risk_sentiment=None,
                regime_duration=25,
                regime_strength=0.8,
                stability_score=0.85,
                transition_probability={},
                expected_transition_days=40.0,
                features=None,
                supporting_indicators={},
                timestamp=datetime.now()
            )
            
            return {
                'symbol': symbol,
                'asset_class': asset_class,
                'regime': regime_analysis.current_regime,
                'confidence': regime_analysis.confidence,
                'analysis': regime_analysis
            }
            
        except Exception as e:
            print(f"⚠️  Regime analysis failed for {symbol}: {e}")
            return None
    
    def select_strategy_for_asset(self, symbol: str, regime_info: Dict) -> Dict:
        """Select optimal strategy for asset based on regime."""
        
        try:
            from features.ml_strategy import predict_best_strategy, StrategyName, StrategyPrediction, MarketConditions
            
            # Strategy mapping based on regime and asset characteristics
            strategy_map = {
                'bull_quiet': StrategyName.MOMENTUM,
                'bull_volatile': StrategyName.MOMENTUM,
                'bear_quiet': StrategyName.MEAN_REVERSION,
                'bear_volatile': StrategyName.MEAN_REVERSION,
                'sideways': StrategyName.MEAN_REVERSION
            }
            
            confidence_map = {
                'bull_quiet': 0.85,
                'bull_volatile': 0.72,
                'bear_quiet': 0.78,
                'bear_volatile': 0.68,
                'sideways': 0.75
            }
            
            expected_return_map = {
                'bull_quiet': 0.12,
                'bull_volatile': 0.15,
                'bear_quiet': 0.06,
                'bear_volatile': 0.08,
                'sideways': 0.08
            }
            
            regime_key = regime_info['regime'].value
            strategy = strategy_map.get(regime_key, StrategyName.MEAN_REVERSION)
            confidence = confidence_map.get(regime_key, 0.7)
            expected_return = expected_return_map.get(regime_key, 0.08)
            
            strategy_prediction = StrategyPrediction(
                strategy=strategy,
                expected_return=expected_return,
                confidence=confidence,
                predicted_sharpe=1.2 + confidence * 0.5,
                predicted_max_drawdown=-0.08 * (1 + (1 - confidence)),
                ranking=1
            )
            
            return {
                'symbol': symbol,
                'strategy': strategy_prediction,
                'regime_confidence': regime_info['confidence']
            }
            
        except Exception as e:
            print(f"⚠️  Strategy selection failed for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                               strategy_info: Dict, regime_info: Dict,
                               volatility: float) -> Dict:
        """Calculate intelligent position size for an asset."""
        
        try:
            from features.position_sizing import calculate_position_size, PositionSizeRequest, SizingMethod
            
            request = PositionSizeRequest(
                symbol=symbol,
                current_price=current_price,
                portfolio_value=self.portfolio_value,
                strategy_name=strategy_info['strategy'].strategy.value,
                method=SizingMethod.INTELLIGENT,
                # Kelly inputs (would come from backtesting in production)
                win_rate=0.65,
                avg_win=0.08,
                avg_loss=-0.04,
                # Intelligence inputs
                confidence=strategy_info['strategy'].confidence,
                market_regime=regime_info['regime'].value,
                volatility=volatility
            )
            
            response = calculate_position_size(request)
            
            return {
                'symbol': symbol,
                'shares': response.recommended_shares,
                'value': response.recommended_value,
                'allocation_pct': response.position_size_pct,
                'risk_pct': response.risk_pct,
                'method': response.method_used,
                'response': response
            }
            
        except Exception as e:
            print(f"⚠️  Position sizing failed for {symbol}: {e}")
            return None
    
    def optimize_portfolio(self, assets: List[Dict]) -> Dict:
        """Optimize entire portfolio using ML intelligence."""
        
        print_section("Portfolio Optimization Process", "🎯")
        
        optimization_results = {
            'assets': [],
            'total_allocation': 0.0,
            'total_risk': 0.0,
            'expected_return': 0.0,
            'diversification_score': 0.0
        }
        
        for i, asset in enumerate(assets, 1):
            print(f"\n--- Analyzing Asset {i}: {asset['symbol']} ({asset['asset_class']}) ---")
            
            # Step 1: Regime analysis
            regime_info = self.analyze_asset_regime(asset['symbol'], asset['asset_class'])
            if not regime_info:
                continue
                
            print(f"   🌊 Regime: {regime_info['regime'].value} (confidence: {regime_info['confidence']:.1%})")
            
            # Step 2: Strategy selection
            strategy_info = self.select_strategy_for_asset(asset['symbol'], regime_info)
            if not strategy_info:
                continue
                
            print(f"   🤖 Strategy: {strategy_info['strategy'].strategy.value}")
            print(f"   📈 Expected Return: {strategy_info['strategy'].expected_return:.1%}")
            print(f"   🎯 Confidence: {strategy_info['strategy'].confidence:.1%}")
            
            # Step 3: Position sizing
            position_info = self.calculate_position_size(
                asset['symbol'], asset['price'], strategy_info, 
                regime_info, asset['volatility']
            )
            if not position_info:
                continue
                
            print(f"   💰 Allocation: {position_info['allocation_pct']:.2%}")
            print(f"   ⚠️  Risk: {position_info['risk_pct']:.2%}")
            print(f"   📊 Value: ${position_info['value']:,.0f}")
            
            # Store results
            asset_result = {
                **asset,
                'regime': regime_info,
                'strategy': strategy_info,
                'position': position_info
            }
            optimization_results['assets'].append(asset_result)
            
            # Update portfolio totals
            optimization_results['total_allocation'] += position_info['allocation_pct']
            optimization_results['total_risk'] += position_info['risk_pct']
            optimization_results['expected_return'] += position_info['allocation_pct'] * strategy_info['strategy'].expected_return
        
        return optimization_results
    
    def generate_regime_rotation_signals(self, current_portfolio: Dict) -> List[Dict]:
        """Generate asset rotation signals based on regime changes."""
        
        rotation_signals = []
        
        for asset in current_portfolio['assets']:
            regime = asset['regime']['regime'].value
            confidence = asset['regime']['confidence']
            allocation = asset['position']['allocation_pct']
            
            # Generate rotation logic
            if regime in ['bear_volatile', 'bear_quiet'] and allocation > 0.05:
                rotation_signals.append({
                    'action': 'REDUCE',
                    'symbol': asset['symbol'],
                    'reason': f'Bear regime detected ({confidence:.1%} confidence)',
                    'current_allocation': allocation,
                    'recommended_allocation': allocation * 0.5,
                    'urgency': 'HIGH' if regime == 'bear_volatile' else 'MEDIUM'
                })
            
            elif regime in ['bull_quiet', 'bull_volatile'] and allocation < 0.08:
                rotation_signals.append({
                    'action': 'INCREASE',
                    'symbol': asset['symbol'],
                    'reason': f'Bull regime detected ({confidence:.1%} confidence)',
                    'current_allocation': allocation,
                    'recommended_allocation': min(allocation * 1.5, 0.12),
                    'urgency': 'MEDIUM'
                })
            
            elif regime == 'sideways' and allocation > 0.06:
                rotation_signals.append({
                    'action': 'MAINTAIN',
                    'symbol': asset['symbol'],
                    'reason': f'Sideways regime - maintain current exposure',
                    'current_allocation': allocation,
                    'recommended_allocation': allocation,
                    'urgency': 'LOW'
                })
        
        return rotation_signals

def demo_portfolio_optimization():
    """Demonstrate complete portfolio optimization process."""
    
    print_banner("Portfolio Optimization with ML Intelligence", "💼")
    
    # Portfolio setup
    portfolio_value = 1000000.0  # $1M portfolio
    optimizer = PortfolioOptimizer(portfolio_value)
    
    print(f"🎯 Portfolio Value: ${portfolio_value:,.0f}")
    print("Optimizing across multiple asset classes using ML intelligence...")
    
    # Define diversified asset universe
    assets = [
        {
            'symbol': 'AAPL',
            'asset_class': 'tech',
            'price': 175.50,
            'volatility': 0.25,
            'sector': 'Technology'
        },
        {
            'symbol': 'MSFT',
            'asset_class': 'tech',
            'price': 380.00,
            'volatility': 0.22,
            'sector': 'Technology'
        },
        {
            'symbol': 'JPM',
            'asset_class': 'value',
            'price': 150.00,
            'volatility': 0.28,
            'sector': 'Financial'
        },
        {
            'symbol': 'JNJ',
            'asset_class': 'defensive',
            'price': 160.00,
            'volatility': 0.15,
            'sector': 'Healthcare'
        },
        {
            'symbol': 'TSLA',
            'asset_class': 'growth',
            'price': 250.00,
            'volatility': 0.45,
            'sector': 'Consumer Discretionary'
        },
        {
            'symbol': 'GLD',
            'asset_class': 'commodities',
            'price': 180.00,
            'volatility': 0.20,
            'sector': 'Commodities'
        },
        {
            'symbol': 'TLT',
            'asset_class': 'bonds',
            'price': 100.00,
            'volatility': 0.12,
            'sector': 'Bonds'
        }
    ]
    
    # Run optimization
    portfolio_results = optimizer.optimize_portfolio(assets)
    
    # Display results
    print_section("Portfolio Optimization Results", "📊")
    
    print(f"Portfolio Summary:")
    print(f"   • Total Allocation: {portfolio_results['total_allocation']:.1%}")
    print(f"   • Total Risk: {portfolio_results['total_risk']:.2%}")
    print(f"   • Expected Return: {portfolio_results['expected_return']:.2%}")
    
    print(f"\nDetailed Allocation:")
    print(f"{'Symbol':<8} {'Sector':<20} {'Regime':<15} {'Strategy':<15} {'Allocation':<12} {'Risk':<8}")
    print("-" * 90)
    
    for asset in portfolio_results['assets']:
        print(f"{asset['symbol']:<8} {asset['sector']:<20} "
              f"{asset['regime']['regime'].value:<15} "
              f"{asset['strategy']['strategy'].strategy.value:<15} "
              f"{asset['position']['allocation_pct']:<11.2%} "
              f"{asset['position']['risk_pct']:<7.2%}")
    
    # Regime rotation analysis
    print_section("Regime-Based Asset Rotation", "🔄")
    
    rotation_signals = optimizer.generate_regime_rotation_signals(portfolio_results)
    
    if rotation_signals:
        print("Rotation signals based on regime analysis:")
        for signal in rotation_signals:
            urgency_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[signal['urgency']]
            print(f"   {urgency_emoji} {signal['action']} {signal['symbol']}: {signal['reason']}")
            print(f"      Current: {signal['current_allocation']:.2%} → Recommended: {signal['recommended_allocation']:.2%}")
    else:
        print("✅ No rotation signals - current allocation aligns with regime analysis")
    
    return portfolio_results

def demo_risk_scenarios():
    """Demonstrate how the system adapts to different risk scenarios."""
    
    print_section("Risk Scenario Analysis", "⚠️")
    
    scenarios = [
        {
            'name': 'Market Crash Scenario',
            'description': 'Sudden shift to bear volatile across all assets',
            'regime_override': 'bear_volatile',
            'volatility_multiplier': 2.0
        },
        {
            'name': 'Bull Run Scenario',
            'description': 'Strong bull market with low volatility',
            'regime_override': 'bull_quiet',
            'volatility_multiplier': 0.7
        },
        {
            'name': 'Uncertain Market Scenario',
            'description': 'Mixed signals and high uncertainty',
            'regime_override': 'sideways',
            'volatility_multiplier': 1.5
        }
    ]
    
    base_portfolio_value = 500000.0
    base_assets = [
        {'symbol': 'SPY', 'price': 450.0, 'asset_class': 'index', 'volatility': 0.18},
        {'symbol': 'QQQ', 'price': 380.0, 'asset_class': 'tech', 'volatility': 0.22},
        {'symbol': 'IWM', 'price': 200.0, 'asset_class': 'small_cap', 'volatility': 0.25}
    ]
    
    print("Testing portfolio allocation across different risk scenarios:")
    
    for scenario in scenarios:
        print(f"\n🎭 Scenario: {scenario['name']}")
        print(f"   {scenario['description']}")
        
        optimizer = PortfolioOptimizer(base_portfolio_value)
        
        # Override regimes for scenario
        modified_assets = []
        for asset in base_assets:
            modified_asset = asset.copy()
            modified_asset['volatility'] *= scenario['volatility_multiplier']
            modified_assets.append(modified_asset)
        
        # Quick position sizing for each asset
        total_allocation = 0.0
        for asset in modified_assets:
            try:
                from features.position_sizing import calculate_position_size, PositionSizeRequest, SizingMethod
                
                request = PositionSizeRequest(
                    symbol=asset['symbol'],
                    current_price=asset['price'],
                    portfolio_value=base_portfolio_value,
                    strategy_name='momentum',
                    method=SizingMethod.INTELLIGENT,
                    win_rate=0.6,
                    avg_win=0.07,
                    avg_loss=-0.04,
                    confidence=0.7,
                    market_regime=scenario['regime_override'],
                    volatility=asset['volatility']
                )
                
                response = calculate_position_size(request)
                total_allocation += response.position_size_pct
                
                print(f"      {asset['symbol']}: {response.position_size_pct:.2%} allocation")
                
            except Exception as e:
                print(f"      {asset['symbol']}: Error calculating position")
        
        print(f"   📊 Total Portfolio Allocation: {total_allocation:.1%}")
        print(f"   💡 Risk Adaptation: {'Conservative' if total_allocation < 0.15 else 'Moderate' if total_allocation < 0.25 else 'Aggressive'}")

def main():
    """Run the complete portfolio optimization example."""
    
    try:
        print("🚀 Starting Portfolio Optimization Example...")
        
        # Main portfolio optimization
        portfolio_results = demo_portfolio_optimization()
        
        # Risk scenario analysis
        demo_risk_scenarios()
        
        print_banner("Portfolio Optimization Complete", "🎉")
        print("✅ ML Intelligence Portfolio Optimization demonstrated!")
        
        print("\n🎯 Key Features Demonstrated:")
        print("   • Multi-asset regime analysis")
        print("   • Strategy selection per asset class")
        print("   • Intelligent position sizing across portfolio")
        print("   • Regime-based asset rotation signals")
        print("   • Risk scenario adaptation")
        
        print("\n📈 Real-World Benefits:")
        print("   • Automatic diversification across regimes")
        print("   • Dynamic rebalancing based on market conditions")
        print("   • Risk-adjusted position sizing")
        print("   • Sector rotation based on regime changes")
        print("   • Portfolio-level risk management")
        
        print("\n📚 Next Steps for Production:")
        print("   • Connect to real-time market data feeds")
        print("   • Implement automated rebalancing triggers")
        print("   • Add transaction cost analysis")
        print("   • Create portfolio monitoring dashboard")
        print("   • Backtest across multiple market cycles")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nExample completed with exit code: {exit_code}")
    sys.exit(exit_code)