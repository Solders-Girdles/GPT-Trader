"""
CLI interface for enhanced evolutionary strategy optimization.
Provides access to the expanded parameter space and novel genetic operators.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from bot.backtest.engine_portfolio import run_backtest
from bot.cli.cli_utils import setup_logging
from bot.dataflow.sources.enhanced_yfinance_source import EnhancedYFinanceSource
from bot.logging import get_logger
from bot.optimization.config import OptimizationConfig, StrategyConfig
from bot.optimization.enhanced_evolution import EnhancedEvolutionEngine
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.enhanced_trend_breakout import (
    EnhancedTrendBreakoutParams,
    EnhancedTrendBreakoutStrategy,
)

logger = get_logger("enhanced_evolution")


def create_enhanced_evaluation_function(
    symbols: list[str], start_date: str, end_date: str, market_symbol: str = "SPY"
) -> callable:
    """Create evaluation function for enhanced evolutionary algorithm."""

    def evaluate_enhanced_strategy(params: dict[str, Any]) -> dict[str, Any]:
        """Evaluate an enhanced strategy with given parameters."""
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

            # Get market data for correlation filter if needed
            if params.get("use_correlation_filter", False):
                try:
                    enhanced_source = EnhancedYFinanceSource()
                    enhanced_source.get_daily_bars(market_symbol, start_date, end_date)
                except Exception as e:
                    logger.warning(f"Failed to get market data for correlation filter: {e}")

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
                    regime_symbol=market_symbol,
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

    return evaluate_enhanced_strategy


def create_enhanced_strategy_config() -> StrategyConfig:
    """Create enhanced strategy configuration with expanded parameter space."""
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


def save_enhanced_results(results: dict[str, Any], output_dir: str) -> None:
    """Save enhanced evolution results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save best individual
    if results.get("best_individual"):
        best_params = results["best_individual"]
        best_params_df = pd.DataFrame([best_params])
        best_params_df.to_csv(output_path / f"best_individual_{timestamp}.csv", index=False)

    # Save performance history
    if hasattr(results, "performance_history"):
        perf_df = pd.DataFrame(results.performance_history)
        perf_df.to_csv(output_path / f"performance_history_{timestamp}.csv", index=False)

    # Save strategy types
    if results.get("strategy_types"):
        strategy_types_df = pd.DataFrame([results["strategy_types"]])
        strategy_types_df.to_csv(output_path / f"strategy_types_{timestamp}.csv", index=False)

    # Save summary
    summary = {
        "timestamp": timestamp,
        "best_fitness": results.get("best_fitness", 0),
        "generations_completed": results.get("generations_completed", 0),
        "final_population_size": results.get("final_population_size", 0),
        "diverse_strategies_found": results.get("diverse_strategies_found", 0),
        "novel_strategies_found": results.get("novel_strategies_found", 0),
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path / f"summary_{timestamp}.csv", index=False)

    logger.info(f"Results saved to {output_path}")


def main():
    """Main CLI function for enhanced evolution."""
    parser = argparse.ArgumentParser(description="Enhanced Evolutionary Strategy Optimization")

    # Data settings
    parser.add_argument(
        "--symbols", default="AAPL,MSFT,GOOGL,AMZN,TSLA", help="Comma-separated list of symbols"
    )
    parser.add_argument("--start-date", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", default="2023-12-31", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--market-symbol", default="SPY", help="Market symbol for regime/correlation filters"
    )
    parser.add_argument(
        "--features",
        default="",
        help="Comma-separated feature sets to compose for research surrogate (e.g., returns,volatility,trend)",
    )

    # Evolution settings
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--population-size", type=int, default=50, help="Population size")
    parser.add_argument(
        "--novelty-weight",
        type=float,
        default=0.3,
        help="Weight for novelty in fitness calculation",
    )
    parser.add_argument(
        "--novelty-threshold", type=float, default=0.3, help="Threshold for novelty archive"
    )

    # Output settings
    parser.add_argument(
        "--output-dir", default="data/enhanced_evolution", help="Output directory for results"
    )
    parser.add_argument("--save-results", action="store_true", help="Save detailed results")

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    # Respect --log-level flag
    try:
        logger.setLevel(getattr(logging, args.log_level.upper()))
        logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    except Exception:
        pass

    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(",")]

    logger.info(f"Starting enhanced evolution with {len(symbols)} symbols: {symbols}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(
        f"Evolution settings: {args.generations} generations, {args.population_size} population"
    )
    logger.info(
        f"Novelty settings: weight={args.novelty_weight}, threshold={args.novelty_threshold}"
    )

    # Create enhanced strategy config
    strategy_config = create_enhanced_strategy_config()

    # Create optimization config
    opt_config = OptimizationConfig(
        name="enhanced_evolution_run",
        description="Enhanced evolutionary optimization with expanded parameter space",
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        evolutionary=True,
        generations=args.generations,
        population_size=args.population_size,
        parameter_space={"strategy": strategy_config},
    )

    # Create evaluation function
    evaluate_func = create_enhanced_evaluation_function(
        symbols, args.start_date, args.end_date, args.market_symbol
    )

    # Create enhanced evolution engine
    engine = EnhancedEvolutionEngine(opt_config, strategy_config)

    # Set novelty parameters
    engine.novelty_threshold = args.novelty_threshold

    # Optional: build research surrogate using selected features (first symbol)
    if args.features:
        try:
            import numpy as np
            from bot.dataflow.sources.yfinance_source import YFinanceSource
            from bot.intelligence.facade import ResearchDatasetBuilder

            feats = [s.strip() for s in args.features.split(",") if s.strip()]
            src = YFinanceSource()
            sym0 = symbols[0] if symbols else "AAPL"
            df = src.get_daily_bars(sym0, start=args.start_date, end=args.end_date)
            # Simple next-day return target
            tgt = df["Close"].pct_change().shift(-1).fillna(0.0)
            builder = ResearchDatasetBuilder()
            X, y = builder.build(df, tgt, features=feats)
            # Tiny ridge to stabilize pseudo-inverse
            try:
                lam = 1e-6
                XtX = X.T @ X + lam * np.eye(X.shape[1])
                Xty = X.T @ y
                w = np.linalg.solve(XtX, Xty)
                engine.set_surrogate({"symbol": sym0, "features": feats, "weights": w.tolist()})
                logger.info(
                    f"Surrogate model fitted on {sym0} with features={feats} (X shape={X.shape})"
                )
            except Exception as e:
                logger.warning(f"Surrogate training failed (falling back to no-op): {e}")
        except Exception as e:
            logger.warning(f"Feature-based surrogate setup skipped: {e}")

    # Run enhanced evolution
    logger.info("Starting enhanced evolution...")
    results = engine.evolve(
        evaluate_func=evaluate_func,
        generations=args.generations,
        population_size=args.population_size,
    )

    # Display results
    logger.info("Enhanced evolution completed!")
    logger.info(f"Best fitness: {results.get('best_fitness', 0):.4f}")
    logger.info(f"Generations completed: {results.get('generations_completed', 0)}")
    logger.info(f"Diverse strategies found: {results.get('diverse_strategies_found', 0)}")
    logger.info(f"Novel strategies found: {results.get('novel_strategies_found', 0)}")

    if results.get("strategy_types"):
        logger.info("Strategy types found:")
        for strategy_type, count in results["strategy_types"].items():
            logger.info(f"  {strategy_type}: {count}")

    # Save results if requested
    if args.save_results:
        save_enhanced_results(results, args.output_dir)

    return results


if __name__ == "__main__":
    main()
