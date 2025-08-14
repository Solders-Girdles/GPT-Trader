"""
CLI interface for the optimization framework.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import OptimizationConfig, ParameterSpace, get_trend_breakout_config
from .engine import OptimizationEngine

logger = logging.getLogger(__name__)


def create_optimization_config(args: argparse.Namespace) -> OptimizationConfig:
    """Create optimization configuration from CLI arguments."""

    # Get strategy configuration
    if args.strategy == "trend_breakout":
        strategy_config = get_trend_breakout_config()
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    # Create parameter space with expanded ranges for maximum exploration
    parameter_space = ParameterSpace(
        strategy=strategy_config,
        grid_ranges={
            # Core parameters with wider ranges
            "donchian_lookback": [10, 25, 50, 100, 200, 300],
            "atr_k": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
            "entry_confirm": [0, 1, 3, 5],
            "cooldown": [0, 5, 15, 30],
            "regime_window": [50, 100, 200, 400, 600],
            "risk_pct": [0.1, 0.3, 0.5, 1.0, 2.0, 3.0],
            # New parameters for diversity
            "trend_strength_threshold": [0.0, 0.5, 1.0, 1.5],
            "momentum_lookback": [5, 10, 20, 50],
            "profit_target_multiplier": [1.0, 2.0, 3.0, 4.0],
        },
        evolutionary_bounds={
            # Core parameters with maximum exploration ranges
            "donchian_lookback": {"min": 5, "max": 500},
            "atr_period": {"min": 2, "max": 100},
            "atr_k": {"min": 0.1, "max": 10.0},
            "entry_confirm": {"min": 0, "max": 10},
            "cooldown": {"min": 0, "max": 50},
            "regime_window": {"min": 10, "max": 1000},
            "risk_pct": {"min": 0.01, "max": 5.0},
            # New parameters for strategy diversity
            "trend_strength_threshold": {"min": 0.0, "max": 2.0},
            "volume_filter": {"choices": [True, False]},
            "volatility_filter": {"choices": [True, False]},
            "momentum_lookback": {"min": 1, "max": 100},
            "profit_target_multiplier": {"min": 0.5, "max": 5.0},
        },
    )

    # Create main configuration
    config = OptimizationConfig(
        name=args.name,
        description=args.description,
        symbols=args.symbols.split(",") if args.symbols else [],
        symbol_list_path=args.symbol_list,
        start_date=args.start_date,
        end_date=args.end_date,
        walk_forward=args.walk_forward,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        method=args.method,
        max_workers=args.max_workers,
        grid_search=args.grid_search,
        grid_sample_size=args.grid_sample_size,
        evolutionary=args.evolutionary,
        generations=args.generations,
        population_size=args.population_size,
        elite_size=args.elite_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        early_stopping=args.early_stopping,
        patience=args.patience,
        min_improvement=args.min_improvement,
        primary_metric=args.primary_metric,
        min_trades=args.min_trades,
        min_sharpe=args.min_sharpe,
        max_drawdown=args.max_drawdown,
        output_dir=args.output_dir,
        save_intermediate=args.save_intermediate,
        create_plots=args.create_plots,
        parameter_space=parameter_space,
        quiet_bars=args.quiet_bars,
        coarse_then_refine=args.coarse_then_refine,
        coarse_months=args.coarse_months,
        coarse_symbols=args.coarse_symbols,
        refine_top_pct=args.refine_top_pct,
        vectorized_phase1=args.vectorized_phase1,
        entry_confirm_phase1=args.entry_confirm_phase1,
        min_rebalance_pct_phase1=args.min_rebalance_pct_phase1,
    )

    return config


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optimization framework for trading strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Basic settings
    parser.add_argument("--name", required=True, help="Optimization run name")
    parser.add_argument("--description", default="", help="Run description")
    parser.add_argument(
        "--strategy",
        default="trend_breakout",
        choices=["trend_breakout"],
        help="Strategy to optimize",
    )

    # Data settings
    parser.add_argument("--symbols", help="Comma-separated list of symbols")
    parser.add_argument("--symbol-list", help="Path to CSV file with symbols")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD")

    # Walk-forward settings
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward testing")
    parser.add_argument("--train-months", type=int, default=12, help="Training window in months")
    parser.add_argument("--test-months", type=int, default=6, help="Test window in months")
    parser.add_argument("--step-months", type=int, default=6, help="Step between windows in months")

    # Optimization settings
    parser.add_argument(
        "--method",
        default="grid",
        choices=["grid", "evolutionary", "both"],
        help="Optimization method",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Number of parallel workers")

    # Grid search settings
    parser.add_argument(
        "--grid-search", action="store_true", default=True, help="Enable grid search"
    )
    parser.add_argument("--grid-sample-size", type=int, help="Random sample size for grid search")

    # Evolutionary settings
    parser.add_argument("--evolutionary", action="store_true", help="Enable evolutionary search")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--population-size", type=int, default=24, help="Population size")
    parser.add_argument("--elite-size", type=int, default=4, help="Elite population size")
    parser.add_argument("--mutation-rate", type=float, default=0.3, help="Mutation probability")
    parser.add_argument("--crossover-rate", type=float, default=0.7, help="Crossover probability")

    # Early stopping
    parser.add_argument(
        "--early-stopping", action="store_true", default=True, help="Enable early stopping"
    )
    parser.add_argument("--patience", type=int, default=10, help="Generations without improvement")
    parser.add_argument(
        "--min-improvement", type=float, default=0.001, help="Minimum improvement threshold"
    )

    # Evaluation settings
    parser.add_argument(
        "--primary-metric",
        default="sharpe",
        choices=["sharpe", "cagr", "sortino", "calmar", "max_drawdown"],
        help="Primary optimization metric",
    )
    parser.add_argument("--min-trades", type=int, default=10, help="Minimum trades required")
    parser.add_argument("--min-sharpe", type=float, default=0.5, help="Minimum Sharpe ratio")
    parser.add_argument("--max-drawdown", type=float, default=0.25, help="Maximum drawdown")

    # Output settings
    parser.add_argument("--output-dir", default="data/optimization", help="Output directory")
    parser.add_argument(
        "--save-intermediate", action="store_true", default=True, help="Save intermediate results"
    )
    parser.add_argument(
        "--create-plots", action="store_true", default=True, help="Create visualization plots"
    )
    parser.add_argument(
        "--quiet-bars", action="store_true", default=True, help="Disable progress bars/log noise"
    )

    # Seeding convenience
    parser.add_argument(
        "--seed-latest",
        action="store_true",
        help="Load seeds.json from most recent run in output_dir",
    )
    parser.add_argument("--seed-from", help="Path to seeds.json or a run directory containing it")
    parser.add_argument(
        "--seed-mode",
        choices=["merge", "replace"],
        default="merge",
        help="How to apply seeds to initial search",
    )
    parser.add_argument(
        "--seed-topk", type=int, default=5, help="Top-k to write to seeds.json at end of run"
    )

    # Coarse-then-refine wrapper
    parser.add_argument(
        "--coarse-then-refine", action="store_true", help="Run fast coarse stage then refine top"
    )
    parser.add_argument("--coarse-months", type=int, default=18, help="Months for coarse stage")
    parser.add_argument(
        "--coarse-symbols", type=int, default=10, help="Number of symbols in coarse subset"
    )
    parser.add_argument(
        "--refine-top-pct", type=float, default=0.02, help="Top fraction to refine (0-1)"
    )
    parser.add_argument(
        "--vectorized-phase1",
        action="store_true",
        help="Use simplified close-to-close model phase 1",
    )
    parser.add_argument(
        "--entry-confirm-phase1", type=int, default=3, help="Entry confirm periods in coarse stage"
    )
    parser.add_argument(
        "--min-rebalance-pct-phase1",
        type=float,
        default=0.01,
        help="Min rebalance pct coarse stage",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    from bot.logging import get_logger

    logger = get_logger(__name__)
    logger.setLevel(getattr(logging, args.log_level))

    try:
        # Create configuration
        config = create_optimization_config(args)

        # Optional: pass research features from env/profile to engine context
        import os

        features_env = os.getenv("GPT_TRADER_FEATURES", "").strip()
        if features_env:
            try:
                config.extra = getattr(config, "extra", {})  # type: ignore[attr-defined]
                config.extra["features"] = [s.strip() for s in features_env.split(",") if s.strip()]  # type: ignore[index]
            except Exception:
                pass

        # Create and run optimization engine
        engine = OptimizationEngine(config)
        summary = engine.run()

        # Print summary
        print("\n" + "=" * 50)
        print("OPTIMIZATION COMPLETE")
        print("=" * 50)
        print(f"Name: {summary['config']['name']}")
        print(f"Total evaluations: {summary['total_evaluations']}")

        if summary["best_result"]:
            best = summary["best_result"]
            print("\nBest result:")
            print(f"  Sharpe ratio: {best['sharpe']:.4f}")
            print(f"  CAGR: {best['cagr']:.4f}")
            print(f"  Max drawdown: {best['max_drawdown']:.4f}")
            print(f"  Parameters: {best['params']}")

        if "statistics" in summary:
            stats = summary["statistics"]
            if "sharpe" in stats:
                print("\nSharpe ratio statistics:")
                print(f"  Mean: {stats['sharpe']['mean']:.4f}")
                print(f"  Max: {stats['sharpe']['max']:.4f}")
                print(f"  Min: {stats['sharpe']['min']:.4f}")

        print(f"\nResults saved to: {config.output_dir}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
