"""
Multi-Objective Optimization Example
Demonstrates Pareto front identification, non-dominated sorting, and multi-objective selection.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.optimization.multi_objective import MultiObjectiveEvolution
from bot.optimization.multi_objective_visualizer import MultiObjectiveVisualizer
from bot.optimization.config import OptimizationConfig, StrategyConfig
from bot.strategy.enhanced_trend_breakout import EnhancedTrendBreakoutStrategy
from bot.backtest.engine_portfolio import run_backtest
from bot.dataflow.sources.enhanced_yfinance_source import EnhancedYFinanceSource

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_evaluation_function(symbols: list[str], start_date: datetime, end_date: datetime):
    """Create evaluation function for multi-objective optimization."""

    def evaluate_strategy(parameters: dict) -> dict:
        """Evaluate a strategy with given parameters."""
        try:
            # Create strategy
            strategy = EnhancedTrendBreakoutStrategy()
            strategy.params.donchian_lookback = parameters.get("donchian_lookback", 55)
            strategy.params.atr_period = parameters.get("atr_period", 20)
            strategy.params.atr_k = parameters.get("atr_k", 2.0)
            strategy.params.volume_ma_period = parameters.get("volume_ma_period", 20)
            strategy.params.volume_threshold = parameters.get("volume_threshold", 1.5)
            strategy.params.use_volume_filter = parameters.get("use_volume_filter", True)
            strategy.params.rsi_period = parameters.get("rsi_period", 14)
            strategy.params.rsi_oversold = parameters.get("rsi_oversold", 30.0)
            strategy.params.rsi_overbought = parameters.get("rsi_overbought", 70.0)
            strategy.params.use_rsi_filter = parameters.get("use_rsi_filter", False)
            strategy.params.bollinger_period = parameters.get("bollinger_period", 20)
            strategy.params.bollinger_std = parameters.get("bollinger_std", 2.0)
            strategy.params.use_bollinger_filter = parameters.get("use_bollinger_filter", False)
            strategy.params.entry_confirmation_periods = parameters.get(
                "entry_confirmation_periods", 1
            )
            strategy.params.exit_confirmation_periods = parameters.get(
                "exit_confirmation_periods", 1
            )
            strategy.params.cooldown_periods = parameters.get("cooldown_periods", 0)
            strategy.params.max_risk_per_trade = parameters.get("max_risk_per_trade", 0.02)
            strategy.params.position_sizing_method = parameters.get("position_sizing_method", "atr")
            strategy.params.use_regime_filter = parameters.get("use_regime_filter", False)
            strategy.params.regime_lookback = parameters.get("regime_lookback", 200)
            strategy.params.use_correlation_filter = parameters.get("use_correlation_filter", False)
            strategy.params.correlation_threshold = parameters.get("correlation_threshold", 0.7)
            strategy.params.correlation_lookback = parameters.get("correlation_lookback", 60)

            # Create portfolio rules
            from bot.portfolio.allocator import PortfolioRules

            rules = PortfolioRules(
                risk_pct=parameters.get("max_risk_per_trade", 0.02),
                max_positions=10,
                rebalance_freq="daily",
            )

            # Run backtest
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
                return {
                    "sharpe": summary.get("sharpe_ratio", 0.0),
                    "max_drawdown": summary.get("max_drawdown", 1.0),
                    "cagr": summary.get("cagr", 0.0),
                    "win_rate": summary.get("win_rate", 0.5),
                    "n_trades": summary.get("num_trades", 0),
                    "profit_factor": summary.get("profit_factor", 1.0),
                    "calmar_ratio": summary.get("calmar_ratio", 0.0),
                    "sortino_ratio": summary.get("sortino_ratio", 0.0),
                    "information_ratio": summary.get("information_ratio", 0.0),
                    "beta": summary.get("beta", 1.0),
                    "alpha": summary.get("alpha", 0.0),
                }
            else:
                return {
                    "sharpe": 0.0,
                    "max_drawdown": 1.0,
                    "cagr": 0.0,
                    "win_rate": 0.5,
                    "n_trades": 0,
                    "profit_factor": 1.0,
                    "calmar_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "information_ratio": 0.0,
                    "beta": 1.0,
                    "alpha": 0.0,
                }

        except Exception as e:
            logger.warning(f"Strategy evaluation failed: {e}")
            return {
                "sharpe": 0.0,
                "max_drawdown": 1.0,
                "cagr": 0.0,
                "win_rate": 0.5,
                "n_trades": 0,
                "profit_factor": 1.0,
                "calmar_ratio": 0.0,
                "sortino_ratio": 0.0,
                "information_ratio": 0.0,
                "beta": 1.0,
                "alpha": 0.0,
            }

    return evaluate_strategy


def main():
    """Run multi-objective optimization example."""
    logger.info("Starting Multi-Objective Optimization Example")

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC"]
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Create optimization config
    config = OptimizationConfig(
        name="multi_objective_example",
        output_dir="outputs/multi_objective_example",
        generations=50,  # Reduced for faster execution
        population_size=30,
        evolutionary=True,
        grid_search=False,
        create_plots=True,
    )

    # Create strategy config
    strategy_config = StrategyConfig(
        name="enhanced_trend_breakout",
        description="Enhanced trend breakout strategy for multi-objective optimization",
    )

    # Create evaluation function
    evaluate_func = create_evaluation_function(symbols, start_date, end_date)

    # Initialize multi-objective evolution
    logger.info("Initializing Multi-Objective Evolution Engine")
    mo_evolution = MultiObjectiveEvolution(config, strategy_config)

    # Run optimization
    logger.info("Starting multi-objective optimization...")
    results = mo_evolution.evolve(
        evaluate_func=evaluate_func,
        generations=config.generations,
        population_size=config.population_size,
    )

    # Display results
    logger.info("Optimization completed!")
    logger.info(f"Pareto front size: {results['pareto_front_size']}")
    logger.info(f"Generations completed: {results['generations_completed']}")

    # Analyze Pareto front
    pareto_front = results.get("pareto_front", [])
    if pareto_front:
        logger.info("\nPareto Front Analysis:")
        logger.info("-" * 30)

        # Find best solutions for each objective
        best_solutions = results.get("best_solutions", {})
        for objective, solution in best_solutions.items():
            logger.info(f"\nBest {objective}:")
            logger.info(f"  Sharpe Ratio: {solution.fitness.sharpe_ratio:.4f}")
            logger.info(f"  Max Drawdown: {solution.fitness.max_drawdown:.4f}")
            logger.info(f"  Consistency: {solution.fitness.consistency:.4f}")
            logger.info(f"  Novelty: {solution.fitness.novelty:.4f}")
            logger.info(f"  Robustness: {solution.fitness.robustness:.4f}")

        # Diversity analysis
        diversity_analysis = results.get("diversity_analysis", {})
        if diversity_analysis:
            logger.info(f"\nDiversity Score: {diversity_analysis.get('diversity_score', 0):.4f}")

            solution_types = diversity_analysis.get("solution_types", {})
            if solution_types:
                logger.info("Solution Types:")
                for sol_type, count in solution_types.items():
                    logger.info(f"  {sol_type}: {count}")

    # Create visualizations
    logger.info("\nCreating visualizations...")
    visualizer = MultiObjectiveVisualizer(output_dir=config.output_dir)

    # Generate comprehensive report
    visualizer.create_comprehensive_report(results)

    # Additional specific plots
    if pareto_front:
        # 2D Pareto front
        visualizer.plot_pareto_front(
            pareto_front, save_path=f"{config.output_dir}/pareto_front_2d.png"
        )

        # 3D Pareto front
        visualizer.plot_3d_pareto_front(
            pareto_front, save_path=f"{config.output_dir}/pareto_front_3d.png"
        )

        # Evolution progress
        visualizer.plot_evolution_progress(
            results.get("performance_history", {}),
            save_path=f"{config.output_dir}/evolution_progress.png",
        )

        # Objective correlations
        visualizer.plot_objective_correlations(
            pareto_front, save_path=f"{config.output_dir}/correlations.png"
        )

        # Solution diversity
        visualizer.plot_solution_diversity(
            pareto_front, save_path=f"{config.output_dir}/diversity.png"
        )

        # Best solutions comparison
        if results.get("best_solutions"):
            visualizer.plot_best_solutions_comparison(
                results["best_solutions"], save_path=f"{config.output_dir}/best_solutions.png"
            )

    logger.info(f"\nAll results and visualizations saved to: {config.output_dir}")

    # Save detailed results
    import json
    from datetime import datetime

    # Convert results to serializable format
    serializable_results = {
        "pareto_front_size": results["pareto_front_size"],
        "generations_completed": results["generations_completed"],
        "performance_history": results["performance_history"],
        "diversity_analysis": results["diversity_analysis"],
        "best_solutions": {},
    }

    # Convert Pareto solutions to serializable format
    if pareto_front:
        serializable_results["pareto_front"] = []
        for solution in pareto_front:
            serializable_results["pareto_front"].append(
                {
                    "parameters": solution.parameters,
                    "fitness": {
                        "sharpe_ratio": solution.fitness.sharpe_ratio,
                        "max_drawdown": solution.fitness.max_drawdown,
                        "consistency": solution.fitness.consistency,
                        "novelty": solution.fitness.novelty,
                        "robustness": solution.fitness.robustness,
                    },
                    "rank": solution.rank,
                    "crowding_distance": solution.crowding_distance,
                }
            )

    # Convert best solutions
    best_solutions = results.get("best_solutions", {})
    for obj, solution in best_solutions.items():
        serializable_results["best_solutions"][obj] = {
            "parameters": solution.parameters,
            "fitness": {
                "sharpe_ratio": solution.fitness.sharpe_ratio,
                "max_drawdown": solution.fitness.max_drawdown,
                "consistency": solution.fitness.consistency,
                "novelty": solution.fitness.novelty,
                "robustness": solution.fitness.robustness,
            },
        }

    # Save to JSON
    results_file = f"{config.output_dir}/detailed_results.json"
    with open(results_file, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    logger.info(f"Detailed results saved to: {results_file}")

    return results


if __name__ == "__main__":
    try:
        results = main()
        logger.info("Multi-objective optimization example completed successfully!")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
