"""
Hierarchical Evolution Example
Demonstrates component-based strategy evolution and composition.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.backtest.engine_portfolio import run_backtest
from bot.optimization.config import OptimizationConfig, StrategyConfig
from bot.optimization.hierarchical_evolution import (
    ComponentParameters,
    HierarchicalEvolutionEngine,
    StrategyComposition,
)
from bot.strategy.enhanced_trend_breakout import EnhancedTrendBreakoutStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_component_evaluator(
    component_type: str, symbols: list[str], start_date: datetime, end_date: datetime
):
    """Create evaluator for a specific component type."""

    def evaluate_component(component: ComponentParameters) -> float:
        """Evaluate a component by testing it in a simplified strategy context."""
        try:
            # Create a basic strategy with the component parameters
            strategy = EnhancedTrendBreakoutStrategy()

            # Apply component parameters to strategy
            params = component.parameters

            if component_type == "entry":
                strategy.params.donchian_lookback = params.get("donchian_lookback", 55)
                strategy.params.atr_period = params.get("atr_period", 20)
                strategy.params.atr_k = params.get("atr_k", 2.0)
                strategy.params.entry_confirmation_periods = params.get(
                    "entry_confirmation_periods", 1
                )
                strategy.params.use_volume_filter = params.get("use_volume_filter", True)
                strategy.params.volume_threshold = params.get("volume_threshold", 1.5)
                strategy.params.use_rsi_filter = params.get("use_rsi_filter", False)
                strategy.params.rsi_oversold = params.get("rsi_oversold", 30.0)
                strategy.params.rsi_overbought = params.get("rsi_overbought", 70.0)

            elif component_type == "exit":
                strategy.params.exit_confirmation_periods = params.get(
                    "exit_confirmation_periods", 1
                )
                strategy.params.cooldown_periods = params.get("cooldown_periods", 0)
                # Note: trailing stop and time-based exit would need strategy modifications

            elif component_type == "risk":
                strategy.params.max_risk_per_trade = params.get("max_risk_per_trade", 0.02)
                strategy.params.position_sizing_method = params.get("position_sizing_method", "atr")
                strategy.params.use_correlation_filter = params.get("use_correlation_filter", False)
                strategy.params.correlation_threshold = params.get("correlation_threshold", 0.7)

            elif component_type == "filter":
                strategy.params.use_regime_filter = params.get("use_regime_filter", False)
                strategy.params.regime_lookback = params.get("regime_lookback", 200)
                strategy.params.use_bollinger_filter = params.get("use_bollinger_filter", False)
                strategy.params.bollinger_period = params.get("bollinger_period", 20)
                strategy.params.bollinger_std = params.get("bollinger_std", 2.0)
                strategy.params.use_time_filter = params.get("use_time_filter", False)
                strategy.params.day_of_week_filter = params.get("day_of_week_filter")
                strategy.params.month_filter = params.get("month_filter")

            # Create portfolio rules
            from bot.portfolio.allocator import PortfolioRules

            rules = PortfolioRules(
                risk_pct=params.get("max_risk_per_trade", 0.02),
                max_positions=params.get("max_positions", 10),
                rebalance_freq="daily",
            )

            # Run backtest with limited symbols for faster evaluation
            test_symbols = symbols[:5]  # Use fewer symbols for component evaluation

            results = run_backtest(
                symbol_list_csv=",".join(test_symbols),
                start=start_date,
                end=end_date,
                strategy=strategy,
                rules=rules,
                debug=False,
                make_plot=False,
                write_portfolio_csv=False,
                write_trades_csv=False,
                write_summary_csv=False,
                return_summary=True,
            )

            if results and "summary" in results:
                summary = results["summary"]

                # Component-specific scoring
                if component_type == "entry":
                    # Entry components: focus on signal quality and trade frequency
                    sharpe = summary.get("sharpe_ratio", 0.0)
                    n_trades = summary.get("num_trades", 0)
                    win_rate = summary.get("win_rate", 0.5)

                    # Score based on Sharpe ratio and reasonable trade frequency
                    trade_score = min(1.0, n_trades / 50.0)  # Normalize trade frequency
                    return sharpe * 0.7 + trade_score * 0.2 + win_rate * 0.1

                elif component_type == "exit":
                    # Exit components: focus on drawdown control and profit preservation
                    max_dd = summary.get("max_drawdown", 1.0)
                    calmar = summary.get("calmar_ratio", 0.0)
                    profit_factor = summary.get("profit_factor", 1.0)

                    # Score based on drawdown control and profit factor
                    dd_score = max(0.0, 1.0 - max_dd)  # Lower drawdown is better
                    return dd_score * 0.5 + calmar * 0.3 + profit_factor * 0.2

                elif component_type == "risk":
                    # Risk components: focus on risk-adjusted returns and consistency
                    sharpe = summary.get("sharpe_ratio", 0.0)
                    sortino = summary.get("sortino_ratio", 0.0)
                    max_dd = summary.get("max_drawdown", 1.0)

                    # Score based on risk-adjusted returns
                    dd_score = max(0.0, 1.0 - max_dd)
                    return sharpe * 0.4 + sortino * 0.4 + dd_score * 0.2

                elif component_type == "filter":
                    # Filter components: focus on consistency and reduced volatility
                    sharpe = summary.get("sharpe_ratio", 0.0)
                    consistency = summary.get("consistency_score", 0.5)
                    max_dd = summary.get("max_drawdown", 1.0)

                    # Score based on consistency and volatility reduction
                    dd_score = max(0.0, 1.0 - max_dd)
                    return sharpe * 0.4 + consistency * 0.4 + dd_score * 0.2

            return 0.0

        except Exception as e:
            logger.warning(f"Component evaluation failed: {e}")
            return 0.0

    return evaluate_component


def create_composition_evaluator(symbols: list[str], start_date: datetime, end_date: datetime):
    """Create evaluator for strategy compositions."""

    def evaluate_composition(composition: StrategyComposition) -> float:
        """Evaluate a complete strategy composition."""
        try:
            # Create strategy with all component parameters
            strategy = EnhancedTrendBreakoutStrategy()

            # Apply entry component parameters
            entry_params = composition.entry_component.parameters
            strategy.params.donchian_lookback = entry_params.get("donchian_lookback", 55)
            strategy.params.atr_period = entry_params.get("atr_period", 20)
            strategy.params.atr_k = entry_params.get("atr_k", 2.0)
            strategy.params.entry_confirmation_periods = entry_params.get(
                "entry_confirmation_periods", 1
            )
            strategy.params.use_volume_filter = entry_params.get("use_volume_filter", True)
            strategy.params.volume_threshold = entry_params.get("volume_threshold", 1.5)
            strategy.params.use_rsi_filter = entry_params.get("use_rsi_filter", False)
            strategy.params.rsi_oversold = entry_params.get("rsi_oversold", 30.0)
            strategy.params.rsi_overbought = entry_params.get("rsi_overbought", 70.0)

            # Apply exit component parameters
            exit_params = composition.exit_component.parameters
            strategy.params.exit_confirmation_periods = exit_params.get(
                "exit_confirmation_periods", 1
            )
            strategy.params.cooldown_periods = exit_params.get("cooldown_periods", 0)

            # Apply risk component parameters
            risk_params = composition.risk_component.parameters
            strategy.params.max_risk_per_trade = risk_params.get("max_risk_per_trade", 0.02)
            strategy.params.position_sizing_method = risk_params.get(
                "position_sizing_method", "atr"
            )
            strategy.params.use_correlation_filter = risk_params.get(
                "use_correlation_filter", False
            )
            strategy.params.correlation_threshold = risk_params.get("correlation_threshold", 0.7)

            # Apply filter component parameters
            for filter_comp in composition.filter_components:
                filter_params = filter_comp.parameters
                strategy.params.use_regime_filter = filter_params.get("use_regime_filter", False)
                strategy.params.regime_lookback = filter_params.get("regime_lookback", 200)
                strategy.params.use_bollinger_filter = filter_params.get(
                    "use_bollinger_filter", False
                )
                strategy.params.bollinger_period = filter_params.get("bollinger_period", 20)
                strategy.params.bollinger_std = filter_params.get("bollinger_std", 2.0)
                strategy.params.use_time_filter = filter_params.get("use_time_filter", False)
                strategy.params.day_of_week_filter = filter_params.get("day_of_week_filter")
                strategy.params.month_filter = filter_params.get("month_filter")

            # Create portfolio rules
            from bot.portfolio.allocator import PortfolioRules

            rules = PortfolioRules(
                risk_pct=risk_params.get("max_risk_per_trade", 0.02),
                max_positions=risk_params.get("max_positions", 10),
                rebalance_freq="daily",
            )

            # Run full backtest
            results = run_backtest(
                symbol_list_csv=",".join(symbols),
                start=start_date,
                end=end_date,
                strategy=strategy,
                rules=rules,
                debug=False,
                make_plot=False,
                write_portfolio_csv=False,
                write_trades_csv=False,
                write_summary_csv=False,
                return_summary=True,
            )

            if results and "summary" in results:
                summary = results["summary"]

                # Overall performance score
                sharpe = summary.get("sharpe_ratio", 0.0)
                max_dd = summary.get("max_drawdown", 1.0)
                calmar = summary.get("calmar_ratio", 0.0)
                consistency = summary.get("consistency_score", 0.5)

                # Weighted score
                dd_score = max(0.0, 1.0 - max_dd)
                return sharpe * 0.4 + calmar * 0.3 + consistency * 0.2 + dd_score * 0.1

            return 0.0

        except Exception as e:
            logger.warning(f"Composition evaluation failed: {e}")
            return 0.0

    return evaluate_composition


def main():
    """Run hierarchical evolution example."""
    logger.info("Starting Hierarchical Evolution Example")

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC"]
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Create optimization config
    config = OptimizationConfig(
        name="hierarchical_evolution_example",
        output_dir="outputs/hierarchical_evolution_example",
        generations=30,  # Reduced for faster execution
        population_size=20,
        evolutionary=True,
        grid_search=False,
        create_plots=True,
    )

    # Create strategy config
    strategy_config = StrategyConfig(
        name="enhanced_trend_breakout",
        description="Enhanced trend breakout strategy for hierarchical evolution",
    )

    # Create component evaluators
    component_evaluators = {
        "entry": create_component_evaluator("entry", symbols, start_date, end_date),
        "exit": create_component_evaluator("exit", symbols, start_date, end_date),
        "risk": create_component_evaluator("risk", symbols, start_date, end_date),
        "filter": create_component_evaluator("filter", symbols, start_date, end_date),
    }

    # Create composition evaluator
    composition_evaluator = create_composition_evaluator(symbols, start_date, end_date)

    # Initialize hierarchical evolution engine
    logger.info("Initializing Hierarchical Evolution Engine")
    hierarchical_engine = HierarchicalEvolutionEngine(config, strategy_config)

    # Run hierarchical evolution
    logger.info("Starting hierarchical evolution...")
    results = hierarchical_engine.evolve_hierarchically(
        component_evaluators=component_evaluators,
        composition_evaluator=composition_evaluator,
        generations=config.generations,
        population_size=config.population_size,
    )

    # Display results
    logger.info("Hierarchical evolution completed!")

    # Component performance analysis
    component_performance = results.get("component_performance", {})
    logger.info("\nComponent Performance Analysis:")
    logger.info("-" * 35)

    for component_type, performance in component_performance.items():
        logger.info(f"\n{component_type.upper()} Components:")
        logger.info(f"  Count: {performance['count']}")
        logger.info(f"  Best Score: {performance['best_score']:.4f}")
        logger.info(f"  Average Score: {performance['avg_score']:.4f}")
        logger.info(f"  Standard Deviation: {performance['std_score']:.4f}")

        # Show top component
        if performance["top_components"]:
            top_comp = performance["top_components"][0]
            logger.info(f"  Top Component Score: {top_comp['performance_score']:.4f}")

    # Composition analysis
    composition_analysis = results.get("composition_analysis", {})
    logger.info("\nComposition Analysis:")
    logger.info("-" * 20)
    logger.info(f"Total Compositions: {composition_analysis.get('total_compositions', 0)}")
    logger.info(f"Best Performance: {composition_analysis.get('best_performance', 0):.4f}")
    logger.info(f"Average Performance: {composition_analysis.get('avg_performance', 0):.4f}")
    logger.info(
        f"Best Composition Score: {composition_analysis.get('best_composition_score', 0):.4f}"
    )

    # Show best strategy
    best_strategy = results.get("best_strategy")
    if best_strategy:
        logger.info("\nBest Composed Strategy:")
        logger.info("-" * 25)
        logger.info(f"Overall Performance: {best_strategy.overall_performance:.4f}")
        logger.info(f"Composition Score: {best_strategy.composition_score:.4f}")
        logger.info(f"Entry Component Score: {best_strategy.entry_component.performance_score:.4f}")
        logger.info(f"Exit Component Score: {best_strategy.exit_component.performance_score:.4f}")
        logger.info(f"Risk Component Score: {best_strategy.risk_component.performance_score:.4f}")

        for i, filter_comp in enumerate(best_strategy.filter_components):
            logger.info(f"Filter Component {i+1} Score: {filter_comp.performance_score:.4f}")

    # Save detailed results
    import json

    # Convert results to serializable format
    serializable_results = {
        "component_performance": component_performance,
        "composition_analysis": composition_analysis,
        "best_strategy": None,
        "top_strategies": composition_analysis.get("top_strategies", []),
    }

    if best_strategy:
        serializable_results["best_strategy"] = {
            "entry_component": {
                "parameters": best_strategy.entry_component.parameters,
                "performance_score": best_strategy.entry_component.performance_score,
            },
            "exit_component": {
                "parameters": best_strategy.exit_component.parameters,
                "performance_score": best_strategy.exit_component.performance_score,
            },
            "risk_component": {
                "parameters": best_strategy.risk_component.parameters,
                "performance_score": best_strategy.risk_component.performance_score,
            },
            "filter_components": [
                {"parameters": f.parameters, "performance_score": f.performance_score}
                for f in best_strategy.filter_components
            ],
            "overall_performance": best_strategy.overall_performance,
            "composition_score": best_strategy.composition_score,
        }

    # Save to JSON
    results_file = f"{config.output_dir}/hierarchical_results.json"
    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info(f"All results saved to: {config.output_dir}")

    return results


if __name__ == "__main__":
    try:
        results = main()
        logger.info("Hierarchical evolution example completed successfully!")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
