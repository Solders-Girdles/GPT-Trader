#!/usr/bin/env python3
"""
Example script demonstrating the new optimization framework.

This script shows how to:
1. Create a configuration for parameter optimization
2. Run grid search optimization
3. Run evolutionary optimization
4. Analyze and visualize results
"""


from bot.optimization.config import OptimizationConfig, ParameterSpace, get_trend_breakout_config
from bot.optimization.engine import OptimizationEngine


def main():
    """Run optimization example."""
    # Setup logging
    from bot.logging import get_logger

    logger = get_logger(__name__)

    # Create strategy configuration
    strategy_config = get_trend_breakout_config()

    # Create parameter space
    parameter_space = ParameterSpace(
        strategy=strategy_config,
        # Grid search ranges (smaller for example)
        grid_ranges={
            "donchian_lookback": [40, 55, 70],
            "atr_k": [1.5, 2.0, 2.5],
            "entry_confirm": [1, 2],
            "cooldown": [0, 2],
            "regime_window": [150, 200],
            "risk_pct": [0.3, 0.5],
        },
        # Evolutionary bounds
        evolutionary_bounds={
            "donchian_lookback": {"min": 20, "max": 200},
            "atr_k": {"min": 0.5, "max": 5.0},
            "atr_period": {"min": 10, "max": 50},
            "entry_confirm": {"min": 1, "max": 3},
            "cooldown": {"min": 0, "max": 10},
            "regime_window": {"min": 100, "max": 300},
            "risk_pct": {"min": 0.1, "max": 2.0},
        },
    )

    # Create optimization configuration
    config = OptimizationConfig(
        name="example_optimization",
        description="Example optimization run for trend breakout strategy",
        symbols=["AAPL", "MSFT", "GOOGL"],  # Small set for example
        start_date="2022-01-01",
        end_date="2022-12-31",
        walk_forward=False,  # Disable for simplicity
        method="grid",  # Start with grid search
        max_workers=1,  # Single worker for example
        grid_search=True,
        evolutionary=False,  # Disable for this example
        primary_metric="sharpe",
        min_trades=5,
        min_sharpe=0.0,  # Lower threshold for example
        max_drawdown=0.5,  # Higher threshold for example
        output_dir="data/optimization/example",
        save_intermediate=True,
        create_plots=True,
        parameter_space=parameter_space,
    )

    print("Starting optimization example...")
    print(f"Configuration: {config.name}")
    print(f"Symbols: {config.symbols}")
    print(f"Date range: {config.start_date} to {config.end_date}")
    print(f"Method: {config.method}")

    # Create and run optimization engine
    engine = OptimizationEngine(config)

    try:
        summary = engine.run()

        print("\n" + "=" * 50)
        print("OPTIMIZATION COMPLETE")
        print("=" * 50)
        print(f"Total evaluations: {summary['total_evaluations']}")

        if summary["best_result"]:
            best = summary["best_result"]
            print("\nBest result:")
            print(f"  Sharpe ratio: {best['sharpe']:.4f}")
            print(f"  CAGR: {best['cagr']:.4f}")
            print(f"  Max drawdown: {best['max_drawdown']:.4f}")
            print(f"  Total trades: {best['n_trades']}")
            print(f"  Parameters: {best['params']}")

        if "statistics" in summary:
            stats = summary["statistics"]
            if "sharpe" in stats:
                print("\nSharpe ratio statistics:")
                print(f"  Mean: {stats['sharpe']['mean']:.4f}")
                print(f"  Max: {stats['sharpe']['max']:.4f}")
                print(f"  Min: {stats['sharpe']['min']:.4f}")
                print(f"  Std: {stats['sharpe']['std']:.4f}")

        print(f"\nResults saved to: {config.output_dir}")
        print("Check the output directory for:")
        print("  - CSV files with all results")
        print("  - Analysis plots and visualizations")
        print("  - HTML dashboard")
        print("=" * 50)

    except Exception as e:
        print(f"Optimization failed: {e}")
        raise


def run_evolutionary_example():
    """Run evolutionary optimization example."""
    print("\n" + "=" * 50)
    print("EVOLUTIONARY OPTIMIZATION EXAMPLE")
    print("=" * 50)

    # Create strategy configuration
    strategy_config = get_trend_breakout_config()

    # Create parameter space
    parameter_space = ParameterSpace(
        strategy=strategy_config,
        # No grid ranges for evolutionary only
        grid_ranges={},
        # Evolutionary bounds
        evolutionary_bounds={
            "donchian_lookback": {"min": 20, "max": 200},
            "atr_k": {"min": 0.5, "max": 5.0},
            "atr_period": {"min": 10, "max": 50},
            "entry_confirm": {"min": 1, "max": 3},
            "cooldown": {"min": 0, "max": 10},
            "regime_window": {"min": 100, "max": 300},
            "risk_pct": {"min": 0.1, "max": 2.0},
        },
    )

    # Create optimization configuration for evolutionary search
    config = OptimizationConfig(
        name="evolutionary_example",
        description="Evolutionary optimization example",
        symbols=["AAPL", "MSFT"],  # Small set for example
        start_date="2022-01-01",
        end_date="2022-12-31",
        walk_forward=False,
        method="evolutionary",
        max_workers=1,
        grid_search=False,
        evolutionary=True,
        generations=20,  # Small number for example
        population_size=12,  # Small population for example
        elite_size=2,
        mutation_rate=0.3,
        crossover_rate=0.7,
        early_stopping=True,
        patience=5,
        min_improvement=0.001,
        primary_metric="sharpe",
        min_trades=5,
        min_sharpe=0.0,
        max_drawdown=0.5,
        output_dir="data/optimization/evolutionary_example",
        save_intermediate=True,
        create_plots=True,
        parameter_space=parameter_space,
    )

    print("Starting evolutionary optimization...")
    print(f"Generations: {config.generations}")
    print(f"Population size: {config.population_size}")
    print(f"Elite size: {config.elite_size}")

    # Create and run optimization engine
    engine = OptimizationEngine(config)

    try:
        summary = engine.run()

        print("\nEvolutionary optimization complete!")
        print(f"Total evaluations: {summary['total_evaluations']}")

        if summary["best_result"]:
            best = summary["best_result"]
            print("\nBest evolved result:")
            print(f"  Sharpe ratio: {best['sharpe']:.4f}")
            print(f"  Parameters: {best['params']}")

        print(f"\nResults saved to: {config.output_dir}")

    except Exception as e:
        print(f"Evolutionary optimization failed: {e}")
        raise


if __name__ == "__main__":
    # Run grid search example
    main()

    # Uncomment to run evolutionary example
    # run_evolutionary_example()
