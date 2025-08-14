#!/usr/bin/env python3
"""
Example script demonstrating the Enhanced Strategy Evolution System.

This example shows how to:
1. Set up enhanced evolution with expanded parameter space
2. Run evolution with novelty search
3. Analyze results and discovered strategies
4. Compare with basic evolution
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime
import pandas as pd
import numpy as np

from bot.optimization.enhanced_evolution import EnhancedEvolutionEngine, EnhancedStrategyParams
from bot.optimization.config import OptimizationConfig, StrategyConfig
from bot.strategy.enhanced_trend_breakout import (
    EnhancedTrendBreakoutStrategy,
    EnhancedTrendBreakoutParams,
)
from bot.backtest.engine_portfolio import run_backtest
from bot.portfolio.allocator import PortfolioRules
from bot.dataflow.sources.enhanced_yfinance_source import EnhancedYFinanceSource
from bot.logging import get_logger

logger = get_logger("enhanced_example")


def create_evaluation_function(symbols, start_date, end_date):
    """Create evaluation function for enhanced evolution."""

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
            import tempfile
            import csv

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


def create_enhanced_strategy_config():
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


def analyze_results(results):
    """Analyze and display evolution results."""
    print("\n" + "=" * 60)
    print("ENHANCED EVOLUTION RESULTS")
    print("=" * 60)

    print(f"Best Fitness: {results.get('best_fitness', 0):.4f}")
    print(f"Generations Completed: {results.get('generations_completed', 0)}")
    print(f"Final Population Size: {results.get('final_population_size', 0)}")
    print(f"Diverse Strategies Found: {results.get('diverse_strategies_found', 0)}")
    print(f"Novel Strategies Found: {results.get('novel_strategies_found', 0)}")

    if results.get("strategy_types"):
        print("\nStrategy Types Discovered:")
        for strategy_type, count in results["strategy_types"].items():
            print(f"  {strategy_type}: {count}")

    if results.get("best_individual"):
        print("\nBest Strategy Parameters:")
        best_params = results["best_individual"]
        for key, value in best_params.items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)


def compare_with_basic_evolution():
    """Compare enhanced evolution with basic evolution."""
    print("\n" + "=" * 60)
    print("COMPARISON: ENHANCED vs BASIC EVOLUTION")
    print("=" * 60)

    print("Enhanced Evolution Advantages:")
    print("1. Expanded Parameter Space: 25+ parameters vs 3-4 parameters")
    print("2. Novel Genetic Operators: Enhanced crossover, mutation, novelty search")
    print("3. Diversity Mechanisms: Strategy archetypes, adaptive phase switching")
    print("4. Enhanced Data Usage: Volume, time, regime, correlation filters")
    print("5. Novelty Search: Discovers surprising strategies")
    print("6. Strategy Classification: Automatically categorizes strategies")

    print("\nExpected Improvements:")
    print("1. More Diverse Strategies: Different strategy types discovered")
    print("2. Better Performance: Higher Sharpe ratios and consistency")
    print("3. Surprising Discoveries: Novel strategies not found by basic evolution")
    print("4. Robustness: Strategies that work in different market conditions")
    print("5. Interpretability: Clear strategy categorization and analysis")


def main():
    """Main example function."""
    print("Enhanced Strategy Evolution System - Example")
    print("=" * 60)

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = "2022-01-01"
    end_date = "2023-12-31"
    generations = 50  # Reduced for example
    population_size = 20  # Reduced for example

    print(f"Symbols: {symbols}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Evolution Settings: {generations} generations, {population_size} population")

    # Create strategy config
    strategy_config = create_enhanced_strategy_config()

    # Create optimization config
    opt_config = OptimizationConfig(
        name="enhanced_evolution_example",
        description="Enhanced evolutionary optimization example",
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        output_dir="data/enhanced_evolution_example",
        evolutionary=True,
        generations=generations,
        population_size=population_size,
        parameter_space={"strategy": strategy_config},
    )

    # Create evaluation function
    evaluate_func = create_evaluation_function(symbols, start_date, end_date)

    # Create enhanced evolution engine
    engine = EnhancedEvolutionEngine(opt_config, strategy_config)

    # Set novelty parameters
    engine.novelty_threshold = 0.3

    print("\nStarting enhanced evolution...")
    print("This may take several minutes...")

    # Run enhanced evolution
    results = engine.evolve(
        evaluate_func=evaluate_func, generations=generations, population_size=population_size
    )

    # Analyze results
    analyze_results(results)

    # Compare with basic evolution
    compare_with_basic_evolution()

    print("\nExample completed!")
    print("Check the output directory for detailed results.")
    print("You can run this with more generations and larger population for better results.")


if __name__ == "__main__":
    main()
