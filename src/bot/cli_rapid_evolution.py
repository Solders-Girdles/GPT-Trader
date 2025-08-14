#!/usr/bin/env python3
"""
Rapid evolutionary optimization CLI for fast iteration.
"""

import argparse
import logging

from bot.backtest.engine_portfolio import run_backtest
from bot.optimization.config import get_trend_breakout_config
from bot.optimization.rapid_evolution import RapidEvolutionEngine
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    from bot.logging import get_logger

    logger = get_logger(__name__)
    logger.setLevel(getattr(logging, level.upper()))


def create_evaluation_function(symbols: list, start_date: str, end_date: str):
    """Create evaluation function for the evolutionary algorithm."""

    def evaluate_strategy(params: dict) -> dict:
        """Evaluate a strategy with given parameters."""
        try:
            # Create strategy
            strategy_params = TrendBreakoutParams(
                donchian_lookback=params.get("donchian_lookback", 55),
                atr_period=params.get("atr_period", 20),
                atr_k=params.get("atr_k", 2.0),
            )
            strategy = TrendBreakoutStrategy(strategy_params)

            # Create portfolio rules
            rules = PortfolioRules(
                per_trade_risk_pct=params.get("risk_pct", 0.5) / 100.0,
                atr_k=params.get("atr_k", 2.0),
                max_positions=10,
                cost_bps=5.0,
            )

            # Run backtest
            results = run_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                rules=rules,
                write_trades_csv=False,
                write_summary_csv=False,
                quiet_mode=True,
                return_summary=True,
            )

            if not results:
                return {
                    "sharpe": float("-inf"),
                    "cagr": float("-inf"),
                    "max_drawdown": float("inf"),
                    "n_trades": 0,
                    "error": "No results",
                }

            # Aggregate results
            avg_sharpe = sum(r.get("sharpe", 0) for r in results) / len(results)
            avg_cagr = sum(r.get("cagr", 0) for r in results) / len(results)
            avg_max_dd = sum(r.get("max_drawdown", 0) for r in results) / len(results)
            total_trades = sum(r.get("n_trades", 0) for r in results)

            return {
                "sharpe": avg_sharpe,
                "cagr": avg_cagr,
                "max_drawdown": avg_max_dd,
                "n_trades": total_trades,
            }

        except Exception as e:
            return {
                "sharpe": float("-inf"),
                "cagr": float("-inf"),
                "max_drawdown": float("inf"),
                "n_trades": 0,
                "error": str(e),
            }

    return evaluate_strategy


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Rapid Evolutionary Optimization")

    # Data settings
    parser.add_argument(
        "--symbols", default="AAPL,MSFT,GOOGL", help="Comma-separated list of symbols"
    )
    parser.add_argument("--start-date", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2022-12-31", help="End date YYYY-MM-DD")

    # Evolution settings
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument("--population-size", type=int, default=20, help="Population size")
    parser.add_argument("--mutation-rate", type=float, default=0.2, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Crossover rate")

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]

    logger.info(f"Starting rapid evolution with {len(symbols)} symbols: {symbols}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(
        f"Evolution settings: {args.generations} generations, {args.population_size} population"
    )

    # Create strategy config
    strategy_config = get_trend_breakout_config()

    # Create evaluation function
    evaluate_func = create_evaluation_function(symbols, args.start_date, args.end_date)

    # Create rapid evolution engine
    engine = RapidEvolutionEngine(
        config=None,
        strategy_config=strategy_config,  # Not needed for rapid evolution
    )

    # Run evolution
    logger.info("Starting evolution...")
    engine.evolve(
        evaluate_func=evaluate_func,
        generations=args.generations,
        population_size=args.population_size,
    )

    # Print summary
    engine.print_summary()

    logger.info("Evolution complete!")


if __name__ == "__main__":
    main()
