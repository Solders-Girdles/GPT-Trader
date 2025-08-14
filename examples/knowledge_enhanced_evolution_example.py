#!/usr/bin/env python3
"""
Knowledge-Enhanced Strategy Evolution System - Next Steps Example

This example demonstrates the next steps for expanding the strategy discovery system:
1. Knowledge persistence and contextual storage
2. Meta-learning and strategy transfer
3. Enhanced discovery with knowledge integration
4. Strategy recommendations and insights
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime

from bot.backtest.engine_portfolio import run_backtest
from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyKnowledgeBase,
    StrategyMetadata,
    StrategyPerformance,
)
from bot.logging import get_logger
from bot.meta_learning.strategy_transfer import AssetCharacteristics, StrategyTransferEngine
from bot.optimization.config import OptimizationConfig, StrategyConfig
from bot.optimization.enhanced_evolution_with_knowledge import KnowledgeEnhancedEvolutionEngine
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.enhanced_trend_breakout import (
    EnhancedTrendBreakoutParams,
    EnhancedTrendBreakoutStrategy,
)

logger = get_logger("knowledge_enhanced_example")


def create_evaluation_function(symbols, start_date, end_date):
    """Create evaluation function for knowledge-enhanced evolution."""

    def evaluate_strategy(params):
        """Evaluate a strategy with given parameters."""
        try:
            # Convert params to EnhancedTrendBreakoutParams
            strategy_params = EnhancedTrendBreakoutParams(
                donchian_lookback=params.get("donchian_lookback", 55),
                atr_period=params.get("atr_period", 20),
                atr_k=params.get("atr_k", 2.0),
                volume_ma_period=params.get("volume_ma_period", 20),
                volume_threshold=params.get("volume_threshold", 1.5),
                use_volume_filter=params.get("use_volume_filter", True),
                rsi_period=params.get("rsi_period", 14),
                rsi_oversold=params.get("rsi_oversold", 30.0),
                rsi_overbought=params.get("rsi_overbought", 70.0),
                use_rsi_filter=params.get("use_rsi_filter", False),
                bollinger_period=params.get("bollinger_period", 20),
                bollinger_std=params.get("bollinger_std", 2.0),
                use_bollinger_filter=params.get("use_bollinger_filter", False),
                day_of_week_filter=params.get("day_of_week_filter"),
                month_filter=params.get("month_filter"),
                use_time_filter=params.get("use_time_filter", False),
                entry_confirmation_periods=params.get("entry_confirmation_periods", 1),
                exit_confirmation_periods=params.get("exit_confirmation_periods", 1),
                cooldown_periods=params.get("cooldown_periods", 0),
                max_risk_per_trade=params.get("max_risk_per_trade", 0.02),
                position_sizing_method=params.get("position_sizing_method", "atr"),
                use_regime_filter=params.get("use_regime_filter", False),
                regime_lookback=params.get("regime_lookback", 200),
                use_correlation_filter=params.get("use_correlation_filter", False),
                correlation_threshold=params.get("correlation_threshold", 0.7),
                correlation_lookback=params.get("correlation_lookback", 60),
            )

            # Create enhanced strategy
            strategy = EnhancedTrendBreakoutStrategy(strategy_params)

            # Create portfolio rules
            rules = PortfolioRules(
                per_trade_risk_pct=params.get("max_risk_per_trade", 0.02),
                atr_k=params.get("atr_k", 2.0),
                max_positions=10,
                cost_bps=5.0,
            )

            # Create a temporary CSV file with symbols
            import csv
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                writer = csv.writer(f)
                writer.writerow(["symbol"])
                for symbol in symbols:
                    writer.writerow([symbol])
                temp_csv_path = f.name

            try:
                # Run backtest
                results = run_backtest(
                    symbol=None,
                    symbol_list_csv=temp_csv_path,
                    start=datetime.strptime(start_date, "%Y-%m-%d"),
                    end=datetime.strptime(end_date, "%Y-%m-%d"),
                    strategy=strategy,
                    rules=rules,
                    regime_on=params.get("use_regime_filter", False),
                    regime_symbol="SPY",
                    regime_window=params.get("regime_lookback", 200),
                    write_trades_csv=False,
                    write_summary_csv=False,
                    quiet_mode=True,
                    return_summary=True,
                )
            finally:
                # Clean up temporary file
                import os

                os.unlink(temp_csv_path)

            if not results or not isinstance(results, dict):
                return {
                    "sharpe": float("-inf"),
                    "cagr": float("-inf"),
                    "max_drawdown": float("inf"),
                    "n_trades": 0,
                    "error": "No results or invalid results type",
                    "params": params,
                }

            # Check if results has a 'summary' key (when return_summary=True)
            if "summary" in results:
                summary = results["summary"]
                return {
                    "sharpe": summary.get("sharpe", float("-inf")),
                    "cagr": summary.get("cagr", float("-inf")),
                    "max_drawdown": summary.get("max_drawdown", float("inf")),
                    "n_trades": summary.get("n_trades", 0),
                    "win_rate": summary.get("win_rate", 0.0),
                    "consistency_score": summary.get("consistency_score", 0.0),
                    "params": params,
                }

            # If results is a list of symbol results, aggregate them
            if isinstance(results, list):
                if not results:
                    return {
                        "sharpe": float("-inf"),
                        "cagr": float("-inf"),
                        "max_drawdown": float("inf"),
                        "n_trades": 0,
                        "error": "Empty results list",
                        "params": params,
                    }

                # Aggregate results across symbols
                avg_sharpe = sum(r.get("sharpe", 0) for r in results) / len(results)
                avg_cagr = sum(r.get("cagr", 0) for r in results) / len(results)
                avg_max_dd = sum(r.get("max_drawdown", 0) for r in results) / len(results)
                total_trades = sum(r.get("n_trades", 0) for r in results)

                # Calculate additional metrics
                win_rate = sum(1 for r in results if r.get("sharpe", 0) > 0) / len(results)
                consistency_score = 1.0 - (
                    sum(abs(r.get("sharpe", 0) - avg_sharpe) for r in results) / len(results)
                )

                return {
                    "sharpe": avg_sharpe,
                    "cagr": avg_cagr,
                    "max_drawdown": avg_max_dd,
                    "n_trades": total_trades,
                    "win_rate": win_rate,
                    "consistency_score": consistency_score,
                    "n_symbols": len(results),
                    "params": params,
                    "timestamp": datetime.now().isoformat(),
                }

            # If results is a single result dictionary
            return {
                "sharpe": results.get("sharpe", float("-inf")),
                "cagr": results.get("cagr", float("-inf")),
                "max_drawdown": results.get("max_drawdown", float("inf")),
                "n_trades": results.get("n_trades", 0),
                "win_rate": results.get("win_rate", 0.0),
                "consistency_score": results.get("consistency_score", 0.0),
                "params": params,
            }

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {
                "sharpe": float("-inf"),
                "cagr": float("-inf"),
                "max_drawdown": float("inf"),
                "n_trades": 0,
                "error": str(e),
                "params": params,
            }

    return evaluate_strategy


def create_strategy_config():
    """Create enhanced strategy configuration."""
    from bot.optimization.config import ParameterDefinition

    config = StrategyConfig(
        name="enhanced_trend_breakout",
        description="Enhanced trend breakout strategy with expanded parameter space",
        parameters={
            # Core trend parameters
            "donchian_lookback": ParameterDefinition(
                name="donchian_lookback",
                type="int",
                min_value=5,
                max_value=200,
                default=55,
                description="Donchian channel lookback period",
            ),
            "atr_period": ParameterDefinition(
                name="atr_period",
                type="int",
                min_value=5,
                max_value=50,
                default=20,
                description="ATR calculation period",
            ),
            "atr_k": ParameterDefinition(
                name="atr_k",
                type="float",
                min_value=0.5,
                max_value=5.0,
                default=2.0,
                description="ATR multiplier for position sizing",
            ),
            # Volume features
            "volume_ma_period": ParameterDefinition(
                name="volume_ma_period",
                type="int",
                min_value=5,
                max_value=50,
                default=20,
                description="Volume moving average period",
            ),
            "volume_threshold": ParameterDefinition(
                name="volume_threshold",
                type="float",
                min_value=1.0,
                max_value=5.0,
                default=1.5,
                description="Volume breakout threshold",
            ),
            "use_volume_filter": ParameterDefinition(
                name="use_volume_filter",
                type="bool",
                default=True,
                description="Enable volume filter",
            ),
            # Momentum features
            "rsi_period": ParameterDefinition(
                name="rsi_period",
                type="int",
                min_value=5,
                max_value=30,
                default=14,
                description="RSI calculation period",
            ),
            "rsi_oversold": ParameterDefinition(
                name="rsi_oversold",
                type="float",
                min_value=20.0,
                max_value=40.0,
                default=30.0,
                description="RSI oversold threshold",
            ),
            "rsi_overbought": ParameterDefinition(
                name="rsi_overbought",
                type="float",
                min_value=60.0,
                max_value=80.0,
                default=70.0,
                description="RSI overbought threshold",
            ),
            "use_rsi_filter": ParameterDefinition(
                name="use_rsi_filter", type="bool", default=False, description="Enable RSI filter"
            ),
            # Volatility features
            "bollinger_period": ParameterDefinition(
                name="bollinger_period",
                type="int",
                min_value=10,
                max_value=50,
                default=20,
                description="Bollinger Bands period",
            ),
            "bollinger_std": ParameterDefinition(
                name="bollinger_std",
                type="float",
                min_value=1.0,
                max_value=4.0,
                default=2.0,
                description="Bollinger Bands standard deviation",
            ),
            "use_bollinger_filter": ParameterDefinition(
                name="use_bollinger_filter",
                type="bool",
                default=False,
                description="Enable Bollinger Bands filter",
            ),
            # Time filters
            "day_of_week_filter": ParameterDefinition(
                name="day_of_week_filter",
                type="int",
                min_value=0,
                max_value=4,
                default=None,
                description="Day of week filter (0=Monday, 4=Friday)",
            ),
            "month_filter": ParameterDefinition(
                name="month_filter",
                type="int",
                min_value=1,
                max_value=12,
                default=None,
                description="Month filter (1-12)",
            ),
            "use_time_filter": ParameterDefinition(
                name="use_time_filter",
                type="bool",
                default=False,
                description="Enable time-based filters",
            ),
            # Entry/Exit enhancements
            "entry_confirmation_periods": ParameterDefinition(
                name="entry_confirmation_periods",
                type="int",
                min_value=0,
                max_value=5,
                default=1,
                description="Entry confirmation periods",
            ),
            "exit_confirmation_periods": ParameterDefinition(
                name="exit_confirmation_periods",
                type="int",
                min_value=0,
                max_value=5,
                default=1,
                description="Exit confirmation periods",
            ),
            "cooldown_periods": ParameterDefinition(
                name="cooldown_periods",
                type="int",
                min_value=0,
                max_value=20,
                default=0,
                description="Cooldown periods between trades",
            ),
            # Risk management
            "max_risk_per_trade": ParameterDefinition(
                name="max_risk_per_trade",
                type="float",
                min_value=0.005,
                max_value=0.05,
                default=0.02,
                description="Maximum risk per trade",
            ),
            "position_sizing_method": ParameterDefinition(
                name="position_sizing_method",
                type="str",
                choices=["atr", "fixed", "kelly"],
                default="atr",
                description="Position sizing method",
            ),
            # Advanced features
            "use_regime_filter": ParameterDefinition(
                name="use_regime_filter",
                type="bool",
                default=False,
                description="Enable market regime filter",
            ),
            "regime_lookback": ParameterDefinition(
                name="regime_lookback",
                type="int",
                min_value=50,
                max_value=500,
                default=200,
                description="Regime filter lookback period",
            ),
            "use_correlation_filter": ParameterDefinition(
                name="use_correlation_filter",
                type="bool",
                default=False,
                description="Enable correlation filter",
            ),
            "correlation_threshold": ParameterDefinition(
                name="correlation_threshold",
                type="float",
                min_value=0.5,
                max_value=0.9,
                default=0.7,
                description="Correlation threshold",
            ),
            "correlation_lookback": ParameterDefinition(
                name="correlation_lookback",
                type="int",
                min_value=20,
                max_value=120,
                default=60,
                description="Correlation lookback period",
            ),
        },
    )

    return config


def demonstrate_knowledge_persistence():
    """Demonstrate knowledge persistence capabilities."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE PERSISTENCE DEMONSTRATION")
    print("=" * 60)

    # Initialize knowledge base
    knowledge_base = StrategyKnowledgeBase("data/knowledge_demo")

    # Create sample strategies with different contexts
    contexts = [
        StrategyContext(
            market_regime="trending",
            time_period="bull_market",
            asset_class="equity",
            risk_profile="moderate",
            volatility_regime="medium",
            correlation_regime="medium",
        ),
        StrategyContext(
            market_regime="volatile",
            time_period="bear_market",
            asset_class="equity",
            risk_profile="conservative",
            volatility_regime="high",
            correlation_regime="high",
        ),
        StrategyContext(
            market_regime="sideways",
            time_period="sideways_market",
            asset_class="equity",
            risk_profile="aggressive",
            volatility_regime="low",
            correlation_regime="low",
        ),
    ]

    # Add sample strategies
    for i, context in enumerate(contexts):
        strategy_metadata = StrategyMetadata(
            strategy_id=f"sample_strategy_{i+1}",
            name=f"Sample Strategy {i+1}",
            description=f"Sample strategy for {context.market_regime} market",
            strategy_type=(
                "trend_following" if i == 0 else "mean_reversion" if i == 1 else "momentum"
            ),
            parameters={
                "donchian_lookback": 55 + i * 10,
                "atr_period": 20 + i * 5,
                "atr_k": 2.0 + i * 0.5,
                "use_volume_filter": i % 2 == 0,
                "use_rsi_filter": i % 2 == 1,
            },
            context=context,
            performance=StrategyPerformance(
                sharpe_ratio=1.5 + i * 0.5,
                cagr=0.15 + i * 0.05,
                max_drawdown=0.1 - i * 0.02,
                win_rate=0.55 + i * 0.05,
                consistency_score=0.7 + i * 0.1,
                n_trades=50 + i * 20,
                avg_trade_duration=5.0,
                profit_factor=1.2 + i * 0.1,
                calmar_ratio=2.0 + i * 0.5,
                sortino_ratio=1.8 + i * 0.3,
                information_ratio=1.0 + i * 0.2,
                beta=0.8 + i * 0.1,
                alpha=0.05 + i * 0.02,
            ),
            discovery_date=datetime.now(),
            last_updated=datetime.now(),
            tags=["sample", context.market_regime],
            notes="Sample strategy for demonstration",
        )

        knowledge_base.add_strategy(strategy_metadata)

    print(f"✓ Added {len(contexts)} sample strategies to knowledge base")

    # Demonstrate strategy retrieval
    print("\n--- Strategy Retrieval Examples ---")

    # Find strategies for trending market
    trending_strategies = knowledge_base.find_strategies(
        context=contexts[0], min_sharpe=1.0, limit=5
    )
    print(f"Found {len(trending_strategies)} strategies for trending market")

    # Get strategy recommendations
    recommendations = knowledge_base.get_strategy_recommendations(contexts[1], 3)
    print(f"Got {len(recommendations)} recommendations for volatile market")

    # Analyze strategy families
    families = knowledge_base.analyze_strategy_families()
    print(f"Strategy families: {list(families.keys())}")

    return knowledge_base


def demonstrate_strategy_transfer(knowledge_base):
    """Demonstrate strategy transfer capabilities."""
    print("\n" + "=" * 60)
    print("STRATEGY TRANSFER DEMONSTRATION")
    print("=" * 60)

    # Initialize transfer engine
    transfer_engine = StrategyTransferEngine(knowledge_base)

    # Get a source strategy
    source_strategies = knowledge_base.find_strategies(min_sharpe=1.5, limit=1)
    if not source_strategies:
        print("No source strategies found")
        return

    source_strategy = source_strategies[0]
    print(
        f"Source strategy: {source_strategy.strategy_id} (Sharpe: {source_strategy.performance.sharpe_ratio:.4f})"
    )

    # Define target context and asset
    target_context = StrategyContext(
        market_regime="crisis",
        time_period="bear_market",
        asset_class="equity",
        risk_profile="conservative",
        volatility_regime="high",
        correlation_regime="high",
    )

    target_asset = AssetCharacteristics(
        volatility=0.35,  # High volatility
        correlation=0.8,  # High correlation
        volume_profile="high",
        price_range=0.05,
        liquidity="medium",
        market_cap=1000000000.0,
        sector="technology",
    )

    # Transfer strategy
    transfer_result = transfer_engine.transfer_strategy(
        source_strategy, target_context, target_asset
    )

    print("\n--- Strategy Transfer Results ---")
    print(f"Confidence Score: {transfer_result['confidence_score']:.3f}")
    print(f"Adaptation Notes: {transfer_result['adaptation_notes']}")

    # Show adapted parameters
    print("\n--- Adapted Parameters ---")
    adapted_params = transfer_result["adapted_parameters"]
    for param, value in adapted_params.items():
        if param in source_strategy.parameters:
            original = source_strategy.parameters[param]
            if original != value:
                print(f"  {param}: {original:.3f} → {value:.3f}")

    # Find similar strategies
    similar_strategies = transfer_engine.get_similar_strategies(source_strategy, 3)
    print(f"\nFound {len(similar_strategies)} similar strategies")

    return transfer_engine


def demonstrate_enhanced_evolution_with_knowledge():
    """Demonstrate knowledge-enhanced evolution."""
    print("\n" + "=" * 60)
    print("KNOWLEDGE-ENHANCED EVOLUTION DEMONSTRATION")
    print("=" * 60)

    # Setup
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    # Create evaluation function
    evaluate_func = create_evaluation_function(symbols, start_date, end_date)

    # Create strategy config
    strategy_config = create_strategy_config()

    # Create parameter space
    from bot.optimization.config import ParameterSpace

    parameter_space = ParameterSpace(
        strategy=strategy_config, grid_bounds={}, evolutionary_bounds={}
    )

    # Create optimization config
    opt_config = OptimizationConfig(
        name="knowledge_enhanced_demo",
        description="Knowledge-enhanced evolution demonstration",
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        output_dir="data/knowledge_enhanced_demo",
        generations=20,  # Reduced for demo
        population_size=15,  # Reduced for demo
        evolutionary=True,
        early_stopping=True,
        patience=5,
        parameter_space=parameter_space,
    )

    # Create context for this evolution
    context = StrategyContext(
        market_regime="trending",
        time_period="bull_market",
        asset_class="equity",
        risk_profile="moderate",
        volatility_regime="medium",
        correlation_regime="medium",
    )

    # Initialize knowledge-enhanced evolution engine
    engine = KnowledgeEnhancedEvolutionEngine(config=opt_config, strategy_config=strategy_config)

    print("Starting knowledge-enhanced evolution...")
    print(f"Knowledge base contains {len(engine.knowledge_base.strategies)} strategies")

    # Run evolution
    results = engine.evolve(evaluate_func, 20, 15, context)

    print("\n--- Evolution Results ---")
    print(f"Best Fitness: {results.get('best_fitness', 'N/A')}")
    print(f"Generations Completed: {results.get('generations_completed', 'N/A')}")
    print(f"New Strategies Discovered: {len(engine.discovered_strategies)}")

    # Show knowledge insights
    knowledge_insights = results.get("knowledge_insights", {})
    print("\n--- Knowledge Insights ---")
    print(f"Total Strategies in KB: {knowledge_insights.get('total_strategies', 0)}")
    print(f"Strategy Families: {list(knowledge_insights.get('strategy_families', {}).keys())}")

    # Get strategy recommendations
    recommendations = engine.get_strategy_recommendations(context, 3)
    print("\n--- Strategy Recommendations for Current Context ---")
    for i, rec in enumerate(recommendations):
        print(
            f"  {i+1}. {rec.strategy_id} (Sharpe: {rec.performance.sharpe_ratio:.4f}, Type: {rec.strategy_type})"
        )

    return engine


def demonstrate_next_steps_roadmap():
    """Demonstrate the roadmap for next steps."""
    print("\n" + "=" * 60)
    print("NEXT STEPS ROADMAP")
    print("=" * 60)

    roadmap = {
        "Phase 1: Knowledge Foundation": [
            "✅ Strategy Knowledge Base - Implemented",
            "✅ Contextual Strategy Storage - Implemented",
            "✅ Strategy Transfer Engine - Implemented",
            "✅ Knowledge-Enhanced Evolution - Implemented",
        ],
        "Phase 2: Advanced Discovery": [
            "🔄 Multi-Objective Optimization",
            "🔄 Hierarchical Strategy Evolution",
            "🔄 Component-Based Strategy Building",
            "🔄 Strategy Composition Framework",
        ],
        "Phase 3: Meta-Learning": [
            "🔄 Cross-Asset Strategy Transfer",
            "🔄 Temporal Strategy Adaptation",
            "🔄 Regime Detection & Switching",
            "🔄 Continuous Learning Pipeline",
        ],
        "Phase 4: Advanced Analytics": [
            "🔄 Strategy Decomposition Analysis",
            "🔄 Performance Attribution",
            "🔄 Risk Decomposition",
            "🔄 Alpha Generation Analysis",
        ],
        "Phase 5: Production Integration": [
            "🔄 Real-Time Strategy Selection",
            "🔄 Portfolio Optimization",
            "🔄 Risk Management Integration",
            "🔄 Performance Monitoring",
        ],
    }

    for phase, items in roadmap.items():
        print(f"\n{phase}:")
        for item in items:
            print(f"  {item}")

    print("\n" + "=" * 60)
    print("IMPLEMENTATION PRIORITIES")
    print("=" * 60)

    priorities = [
        "1. Multi-Objective Optimization: Balance Sharpe ratio, drawdown, consistency, and novelty",
        "2. Hierarchical Evolution: Evolve strategy components separately then compose",
        "3. Regime Detection: Automatic market regime identification and strategy switching",
        "4. Real-Time Adaptation: Continuous strategy parameter adjustment based on market conditions",
        "5. Portfolio Integration: Combine multiple strategies into optimal portfolios",
        "6. Advanced Analytics: Deep analysis of strategy performance drivers",
        "7. Production Pipeline: End-to-end strategy discovery to deployment workflow",
    ]

    for priority in priorities:
        print(f"  {priority}")


def main():
    """Main demonstration function."""
    print("Knowledge-Enhanced Strategy Evolution System - Next Steps")
    print("=" * 60)
    print("This example demonstrates the next steps for expanding the strategy discovery system")
    print("with knowledge persistence, contextual learning, and meta-learning capabilities.")

    try:
        # Demonstrate knowledge persistence
        knowledge_base = demonstrate_knowledge_persistence()

        # Demonstrate strategy transfer
        transfer_engine = demonstrate_strategy_transfer(knowledge_base)

        # Demonstrate knowledge-enhanced evolution
        evolution_engine = demonstrate_enhanced_evolution_with_knowledge()

        # Show roadmap
        demonstrate_next_steps_roadmap()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("✅ Knowledge persistence system working")
        print("✅ Strategy transfer capabilities demonstrated")
        print("✅ Knowledge-enhanced evolution operational")
        print("✅ Next steps roadmap defined")
        print("\nThe system is ready for the next phase of development!")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
