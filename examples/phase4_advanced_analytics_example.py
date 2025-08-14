"""
Phase 4: Advanced Analytics Example

This example demonstrates the comprehensive analytics capabilities implemented in Phase 4:
- Strategy Decomposition Analysis
- Performance Attribution
- Risk Decomposition
- Alpha Generation Analysis

The example shows how to use these analytics to gain deep insights into strategy performance
and identify optimization opportunities.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from bot.analytics.alpha_analysis import AlphaGenerationAnalyzer
from bot.analytics.attribution import PerformanceAttributionAnalyzer

# Import analytics modules
from bot.analytics.decomposition import StrategyDecompositionAnalyzer
from bot.analytics.risk_decomposition import RiskDecompositionAnalyzer

# Import data source
from bot.dataflow.sources.enhanced_yfinance_source import EnhancedYFinanceSource

# Import strategy components
from bot.strategy.components import (
    ComponentBasedStrategy,
    ComponentConfig,
    DonchianBreakoutEntry,
    FixedTargetExit,
    PositionSizingRisk,
    RegimeFilter,
    RSIEntry,
    TrailingStopExit,
    VolatilityFilter,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_strategy() -> ComponentBasedStrategy:
    """Create a sample component-based strategy for analysis."""
    logger.info("Creating sample component-based strategy...")

    # Create component configurations
    entry_config = ComponentConfig(
        component_type="entry",
        parameters={"lookback": 55, "atr_period": 20, "atr_k": 2.0},
        enabled=True,
        priority=1,
    )

    rsi_config = ComponentConfig(
        component_type="entry",
        parameters={"rsi_period": 14, "oversold": 30, "overbought": 70},
        enabled=True,
        priority=2,
    )

    exit_config = ComponentConfig(
        component_type="exit",
        parameters={"target_pct": 0.05, "stop_pct": 0.03},
        enabled=True,
        priority=1,
    )

    trailing_config = ComponentConfig(
        component_type="exit", parameters={"trailing_pct": 0.02}, enabled=True, priority=2
    )

    risk_config = ComponentConfig(
        component_type="risk",
        parameters={"max_position_size": 0.1, "volatility_lookback": 20},
        enabled=True,
        priority=1,
    )

    regime_config = ComponentConfig(
        component_type="filter",
        parameters={"regime_type": "trending", "confidence_threshold": 0.7},
        enabled=True,
        priority=1,
    )

    volatility_config = ComponentConfig(
        component_type="filter",
        parameters={"volatility_period": 20, "high_vol_threshold": 0.03},
        enabled=True,
        priority=2,
    )

    # Create components
    components = [
        DonchianBreakoutEntry(entry_config),
        RSIEntry(rsi_config),
        FixedTargetExit(exit_config),
        TrailingStopExit(trailing_config),
        PositionSizingRisk(risk_config),
        RegimeFilter(regime_config),
        VolatilityFilter(volatility_config),
    ]

    # Create strategy
    strategy = ComponentBasedStrategy(components)

    logger.info(f"Created strategy with {len(components)} components")
    return strategy


def load_sample_data() -> pd.DataFrame:
    """Load sample market data for analysis."""
    logger.info("Loading sample market data...")

    # Use YFinance source to get real data
    source = EnhancedYFinanceSource()

    # Get SPY data for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    data = source.get_daily_bars(
        symbol="SPY", start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
    )

    logger.info(f"Loaded {len(data)} data points for SPY")
    return data


def run_decomposition_analysis(strategy: ComponentBasedStrategy, data: pd.DataFrame):
    """Run strategy decomposition analysis."""
    logger.info("Running strategy decomposition analysis...")

    analyzer = StrategyDecompositionAnalyzer()
    result = analyzer.analyze_strategy(strategy, data)

    print("\n" + "=" * 60)
    print("STRATEGY DECOMPOSITION ANALYSIS")
    print("=" * 60)
    print(analyzer.generate_report(result))

    # Get top contributors
    top_contributors = analyzer.get_top_contributors(result, n=3)
    print("\nTop Contributing Components:")
    for i, contrib in enumerate(top_contributors, 1):
        print(f"{i}. {contrib.component_name}: {contrib.contribution_score:.4f}")

    # Get improvement opportunities
    opportunities = analyzer.get_improvement_opportunities(result)
    if opportunities:
        print("\nImprovement Opportunities:")
        for opportunity in opportunities:
            print(f"  - {opportunity}")

    return result


def run_attribution_analysis(strategy: ComponentBasedStrategy, data: pd.DataFrame):
    """Run performance attribution analysis."""
    logger.info("Running performance attribution analysis...")

    analyzer = PerformanceAttributionAnalyzer()
    result = analyzer.analyze_strategy(strategy, data)

    print("\n" + "=" * 60)
    print("PERFORMANCE ATTRIBUTION ANALYSIS")
    print("=" * 60)
    print(analyzer.generate_report(result))

    # Get top factors
    top_factors = analyzer.get_top_factors(result, n=3)
    print("\nTop Contributing Factors:")
    for i, factor in enumerate(top_factors, 1):
        print(f"{i}. {factor.factor_name}: {factor.contribution:.4f}")

    # Get improvement opportunities
    opportunities = analyzer.get_improvement_opportunities(result)
    if opportunities:
        print("\nImprovement Opportunities:")
        for opportunity in opportunities:
            print(f"  - {opportunity}")

    return result


def run_risk_decomposition_analysis(strategy: ComponentBasedStrategy, data: pd.DataFrame):
    """Run risk decomposition analysis."""
    logger.info("Running risk decomposition analysis...")

    analyzer = RiskDecompositionAnalyzer()
    result = analyzer.analyze_strategy(strategy, data)

    print("\n" + "=" * 60)
    print("RISK DECOMPOSITION ANALYSIS")
    print("=" * 60)
    print(analyzer.generate_report(result))

    # Get risk breakdown
    risk_breakdown = analyzer.get_risk_breakdown(result)
    print("\nRisk Breakdown:")
    for risk_type, percentage in risk_breakdown.items():
        print(f"  {risk_type}: {percentage:.2%}")

    # Get risk insights
    insights = analyzer.get_risk_insights(result)
    if insights:
        print("\nRisk Insights:")
        for insight in insights:
            print(f"  - {insight}")

    # Run stress test
    stress_scenarios = {
        "Market Crash": 2.0,
        "High Volatility": 1.5,
        "Low Volatility": 0.5,
        "Trend Reversal": -1.0,
    }

    stress_results = analyzer.calculate_stress_test(strategy, data, stress_scenarios)
    print("\nStress Test Results:")
    for scenario, metrics in stress_results.items():
        print(f"  {scenario}:")
        print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"    VaR (95%): {metrics['var_95']:.4f}")
        print(f"    CVaR (95%): {metrics['cvar_95']:.4f}")
        print(f"    Max Drawdown: {metrics['max_drawdown']:.4f}")

    return result


def run_alpha_analysis(strategy: ComponentBasedStrategy, data: pd.DataFrame):
    """Run alpha generation analysis."""
    logger.info("Running alpha generation analysis...")

    analyzer = AlphaGenerationAnalyzer()
    result = analyzer.analyze_strategy(strategy, data)

    print("\n" + "=" * 60)
    print("ALPHA GENERATION ANALYSIS")
    print("=" * 60)
    print(analyzer.generate_report(result))

    # Get top alpha sources
    top_sources = analyzer.get_top_alpha_sources(result, n=3)
    print("\nTop Alpha Sources:")
    for i, source in enumerate(top_sources, 1):
        print(f"{i}. {source.source_name}: {source.alpha_contribution:.4f}")

    # Get alpha insights
    insights = analyzer.get_alpha_insights(result)
    if insights:
        print("\nAlpha Insights:")
        for insight in insights:
            print(f"  - {insight}")

    # Optimize alpha weights
    optimized_weights = analyzer.optimize_alpha_weights(
        result.alpha_sources, target_alpha=0.05, risk_budget=0.02
    )

    print("\nOptimized Alpha Weights:")
    for source_name, weight in optimized_weights.items():
        print(f"  {source_name}: {weight:.3f}")

    return result


def run_comprehensive_analysis():
    """Run comprehensive Phase 4 analytics."""
    logger.info("Starting comprehensive Phase 4 analytics...")

    # Create strategy and load data
    strategy = create_sample_strategy()
    data = load_sample_data()

    # Run all analyses
    decomposition_result = run_decomposition_analysis(strategy, data)
    attribution_result = run_attribution_analysis(strategy, data)
    risk_result = run_risk_decomposition_analysis(strategy, data)
    alpha_result = run_alpha_analysis(strategy, data)

    # Generate summary report
    print("\n" + "=" * 80)
    print("PHASE 4 ANALYTICS SUMMARY")
    print("=" * 80)

    print(f"\nStrategy: {strategy.__class__.__name__}")
    print(f"Analysis Period: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Data Points: {len(data)}")

    print("\nKey Metrics:")
    print(f"  - Total Alpha: {alpha_result.total_alpha:.4f}")
    print(f"  - Alpha Quality: {alpha_result.alpha_quality:.3f}")
    print(f"  - Alpha Persistence: {alpha_result.alpha_persistence:.3f}")
    print(f"  - Total Risk: {risk_result.total_risk:.4f}")
    print(f"  - Risk Quality: {risk_result.risk_quality:.3f}")
    print(f"  - Decomposition Quality: {decomposition_result.decomposition_quality:.3f}")
    print(f"  - Attribution Quality: {attribution_result.attribution_quality:.3f}")

    # Overall assessment
    print("\nOverall Assessment:")

    # Alpha assessment
    if alpha_result.alpha_quality > 0.7:
        print("  ✅ Alpha generation is strong and sustainable")
    elif alpha_result.alpha_quality > 0.5:
        print("  ⚠️  Alpha generation is moderate, consider improvements")
    else:
        print("  ❌ Alpha generation is weak, significant improvements needed")

    # Risk assessment
    if risk_result.risk_quality > 0.8:
        print("  ✅ Risk decomposition is comprehensive and accurate")
    elif risk_result.risk_quality > 0.6:
        print("  ⚠️  Risk decomposition is adequate, some gaps remain")
    else:
        print("  ❌ Risk decomposition is incomplete, additional analysis needed")

    # Component assessment
    if decomposition_result.decomposition_quality > 0.8:
        print("  ✅ Strategy components are well understood")
    elif decomposition_result.decomposition_quality > 0.6:
        print("  ⚠️  Strategy components are partially understood")
    else:
        print("  ❌ Strategy components need better analysis")

    # Attribution assessment
    if attribution_result.attribution_quality > 0.8:
        print("  ✅ Performance attribution is comprehensive")
    elif attribution_result.attribution_quality > 0.6:
        print("  ⚠️  Performance attribution is adequate")
    else:
        print("  ❌ Performance attribution needs improvement")

    print("\nPhase 4 Analytics Complete!")
    print(f"Generated {len(alpha_result.alpha_sources)} alpha sources")
    print(f"Identified {len(risk_result.risk_components)} risk components")
    print(f"Analyzed {len(decomposition_result.component_contributions)} strategy components")
    print(f"Attributed performance to {len(attribution_result.factors)} factors")


def main():
    """Main function to run the Phase 4 analytics example."""
    try:
        run_comprehensive_analysis()
    except Exception as e:
        logger.error(f"Error running Phase 4 analytics: {e}")
        raise


if __name__ == "__main__":
    main()
