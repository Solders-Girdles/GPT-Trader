"""
Component-Based Strategy Building Example
Demonstrates building strategies from reusable, tested components.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.strategy.components import (
    ComponentBasedStrategy,
    ComponentRegistry,
    ComponentConfig,
    DonchianBreakoutEntry,
    RSIEntry,
    VolumeBreakoutEntry,
    FixedTargetExit,
    TrailingStopExit,
    TimeBasedExit,
    PositionSizingRisk,
    CorrelationFilterRisk,
    RegimeFilter,
    VolatilityFilter,
    BollingerFilter,
    TimeFilter,
)
from bot.backtest.engine_portfolio import run_backtest
from bot.dataflow.sources.enhanced_yfinance_source import EnhancedYFinanceSource

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_basic_strategy() -> ComponentBasedStrategy:
    """Create a basic component-based strategy."""
    logger.info("Creating basic component-based strategy...")

    # Create components
    components = [
        # Entry component
        DonchianBreakoutEntry(
            ComponentConfig(
                "entry", {"lookback": 55, "atr_period": 20, "atr_k": 2.0}, enabled=True, priority=3
            )
        ),
        # Exit component
        FixedTargetExit(
            ComponentConfig(
                "exit", {"profit_target": 0.05, "stop_loss": 0.03}, enabled=True, priority=2
            )
        ),
        # Risk component
        PositionSizingRisk(
            ComponentConfig(
                "risk",
                {"risk_per_trade": 0.02, "method": "atr", "atr_period": 20},
                enabled=True,
                priority=1,
            )
        ),
        # Filter component
        RegimeFilter(ComponentConfig("filter", {"lookback": 200}, enabled=True, priority=4)),
    ]

    return ComponentBasedStrategy(components)


def create_advanced_strategy() -> ComponentBasedStrategy:
    """Create an advanced component-based strategy with multiple components."""
    logger.info("Creating advanced component-based strategy...")

    # Create components
    components = [
        # Multiple entry components
        DonchianBreakoutEntry(
            ComponentConfig(
                "donchian_entry",
                {"lookback": 55, "atr_period": 20, "atr_k": 2.0},
                enabled=True,
                priority=4,
            )
        ),
        RSIEntry(
            ComponentConfig(
                "rsi_entry",
                {"period": 14, "oversold": 30.0, "overbought": 70.0},
                enabled=True,
                priority=3,
            )
        ),
        VolumeBreakoutEntry(
            ComponentConfig(
                "volume_entry", {"period": 20, "threshold": 1.5}, enabled=True, priority=2
            )
        ),
        # Multiple exit components
        FixedTargetExit(
            ComponentConfig(
                "fixed_exit", {"profit_target": 0.05, "stop_loss": 0.03}, enabled=True, priority=3
            )
        ),
        TrailingStopExit(
            ComponentConfig(
                "trailing_exit", {"atr_period": 20, "atr_multiplier": 2.0}, enabled=True, priority=2
            )
        ),
        TimeBasedExit(
            ComponentConfig("time_exit", {"max_hold_days": 30}, enabled=True, priority=1)
        ),
        # Risk components
        PositionSizingRisk(
            ComponentConfig(
                "position_sizing",
                {"risk_per_trade": 0.02, "method": "atr", "atr_period": 20},
                enabled=True,
                priority=2,
            )
        ),
        CorrelationFilterRisk(
            ComponentConfig(
                "correlation_filter", {"threshold": 0.7, "lookback": 60}, enabled=True, priority=1
            )
        ),
        # Multiple filter components
        RegimeFilter(ComponentConfig("regime_filter", {"lookback": 200}, enabled=True, priority=5)),
        VolatilityFilter(
            ComponentConfig(
                "volatility_filter",
                {"short_period": 20, "long_period": 100, "threshold": 1.2},
                enabled=True,
                priority=4,
            )
        ),
        BollingerFilter(
            ComponentConfig(
                "bollinger_filter", {"period": 20, "std_dev": 2.0}, enabled=True, priority=3
            )
        ),
        TimeFilter(
            ComponentConfig(
                "time_filter", {"day_of_week": None, "month": None}, enabled=True, priority=2
            )
        ),
    ]

    return ComponentBasedStrategy(components)


def create_strategy_from_config() -> ComponentBasedStrategy:
    """Create a strategy from configuration."""
    logger.info("Creating strategy from configuration...")

    config = {
        "strategy_type": "component_based",
        "components": [
            {
                "name": "donchian_breakout",
                "parameters": {"lookback": 40, "atr_period": 15, "atr_k": 1.5},
                "enabled": True,
                "priority": 3,
            },
            {
                "name": "rsi_entry",
                "parameters": {"period": 10, "oversold": 25.0, "overbought": 75.0},
                "enabled": True,
                "priority": 2,
            },
            {
                "name": "fixed_target",
                "parameters": {"profit_target": 0.03, "stop_loss": 0.02},
                "enabled": True,
                "priority": 2,
            },
            {
                "name": "position_sizing",
                "parameters": {"risk_per_trade": 0.015, "method": "atr", "atr_period": 15},
                "enabled": True,
                "priority": 1,
            },
            {
                "name": "regime_filter",
                "parameters": {"lookback": 150},
                "enabled": True,
                "priority": 4,
            },
        ],
    }

    return ComponentBasedStrategy.from_config(config)


def demonstrate_component_registry():
    """Demonstrate the component registry functionality."""
    logger.info("Demonstrating component registry...")

    registry = ComponentRegistry()

    # List all components
    all_components = registry.list_components()
    logger.info(f"Available components: {all_components}")

    # List components by type
    entry_components = registry.list_components("entry")
    exit_components = registry.list_components("exit")
    risk_components = registry.list_components("risk")
    filter_components = registry.list_components("filter")

    logger.info(f"Entry components: {entry_components}")
    logger.info(f"Exit components: {exit_components}")
    logger.info(f"Risk components: {risk_components}")
    logger.info(f"Filter components: {filter_components}")

    # Get component information
    for component_name in ["donchian_breakout", "rsi_entry", "regime_filter"]:
        try:
            info = registry.get_component_info(component_name)
            logger.info(f"\nComponent: {component_name}")
            logger.info(f"  Type: {info['type']}")
            logger.info(f"  Description: {info['description']}")
            logger.info(f"  Parameter bounds: {info['parameter_bounds']}")
        except Exception as e:
            logger.warning(f"Failed to get info for {component_name}: {e}")


def test_strategy_performance(
    strategy: ComponentBasedStrategy, symbols: list[str], start_date: datetime, end_date: datetime
) -> dict:
    """Test strategy performance."""
    logger.info(f"Testing strategy performance...")

    try:
        # Create portfolio rules
        from bot.portfolio.allocator import PortfolioRules

        rules = PortfolioRules(risk_pct=0.02, max_positions=10, rebalance_freq="daily")

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
                "sharpe_ratio": summary.get("sharpe_ratio", 0.0),
                "cagr": summary.get("cagr", 0.0),
                "max_drawdown": summary.get("max_drawdown", 1.0),
                "win_rate": summary.get("win_rate", 0.5),
                "num_trades": summary.get("num_trades", 0),
                "profit_factor": summary.get("profit_factor", 1.0),
                "calmar_ratio": summary.get("calmar_ratio", 0.0),
                "sortino_ratio": summary.get("sortino_ratio", 0.0),
                "information_ratio": summary.get("information_ratio", 0.0),
                "beta": summary.get("beta", 1.0),
                "alpha": summary.get("alpha", 0.0),
            }
        else:
            return {
                "sharpe_ratio": 0.0,
                "cagr": 0.0,
                "max_drawdown": 1.0,
                "win_rate": 0.5,
                "num_trades": 0,
                "profit_factor": 1.0,
                "calmar_ratio": 0.0,
                "sortino_ratio": 0.0,
                "information_ratio": 0.0,
                "beta": 1.0,
                "alpha": 0.0,
            }

    except Exception as e:
        logger.error(f"Strategy testing failed: {e}")
        return {
            "sharpe_ratio": 0.0,
            "cagr": 0.0,
            "max_drawdown": 1.0,
            "win_rate": 0.5,
            "num_trades": 0,
            "profit_factor": 1.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "information_ratio": 0.0,
            "beta": 1.0,
            "alpha": 0.0,
            "error": str(e),
        }


def main():
    """Run component-based strategy example."""
    logger.info("Starting Component-Based Strategy Example")

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC"]
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Demonstrate component registry
    demonstrate_component_registry()

    # Create different strategies
    strategies = {
        "Basic Strategy": create_basic_strategy(),
        "Advanced Strategy": create_advanced_strategy(),
        "Config Strategy": create_strategy_from_config(),
    }

    # Test each strategy
    results = {}

    for strategy_name, strategy in strategies.items():
        logger.info(f"\nTesting {strategy_name}...")

        # Show strategy configuration
        config = strategy.to_config()
        logger.info(f"Strategy configuration: {config}")

        # Test performance
        performance = test_strategy_performance(strategy, symbols, start_date, end_date)
        results[strategy_name] = performance

        # Display results
        logger.info(f"\n{strategy_name} Performance:")
        logger.info(f"  Sharpe Ratio: {performance['sharpe_ratio']:.4f}")
        logger.info(f"  CAGR: {performance['cagr']:.4f}")
        logger.info(f"  Max Drawdown: {performance['max_drawdown']:.4f}")
        logger.info(f"  Win Rate: {performance['win_rate']:.4f}")
        logger.info(f"  Number of Trades: {performance['num_trades']}")
        logger.info(f"  Profit Factor: {performance['profit_factor']:.4f}")
        logger.info(f"  Calmar Ratio: {performance['calmar_ratio']:.4f}")

        if "error" in performance:
            logger.error(f"  Error: {performance['error']}")

    # Compare strategies
    logger.info("\n" + "=" * 50)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 50)

    comparison_data = []
    for strategy_name, performance in results.items():
        comparison_data.append(
            {
                "Strategy": strategy_name,
                "Sharpe": performance["sharpe_ratio"],
                "CAGR": performance["cagr"],
                "MaxDD": performance["max_drawdown"],
                "WinRate": performance["win_rate"],
                "Trades": performance["num_trades"],
                "ProfitFactor": performance["profit_factor"],
            }
        )

    # Sort by Sharpe ratio
    comparison_data.sort(key=lambda x: x["Sharpe"], reverse=True)

    for data in comparison_data:
        logger.info(
            f"{data['Strategy']:20} | "
            f"Sharpe: {data['Sharpe']:6.3f} | "
            f"CAGR: {data['CAGR']:6.3f} | "
            f"MaxDD: {data['MaxDD']:6.3f} | "
            f"WinRate: {data['WinRate']:6.3f} | "
            f"Trades: {data['Trades']:4d} | "
            f"PF: {data['ProfitFactor']:6.3f}"
        )

    # Demonstrate component modification
    logger.info("\n" + "=" * 50)
    logger.info("COMPONENT MODIFICATION EXAMPLE")
    logger.info("=" * 50)

    # Create a strategy and modify its components
    base_strategy = create_basic_strategy()

    # Modify a component parameter
    for component in base_strategy.components:
        if isinstance(component, DonchianBreakoutEntry):
            component.parameters["lookback"] = 30  # Change from 55 to 30
            logger.info(f"Modified Donchian lookback from 55 to 30")
            break

    # Test modified strategy
    modified_performance = test_strategy_performance(base_strategy, symbols, start_date, end_date)
    logger.info(f"Modified Strategy Sharpe Ratio: {modified_performance['sharpe_ratio']:.4f}")

    # Save results
    import json
    from datetime import datetime

    output_dir = "outputs/component_based_strategy_example"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_results = {
        "strategies": results,
        "modified_strategy": modified_performance,
        "comparison_data": comparison_data,
        "timestamp": datetime.now().isoformat(),
    }

    results_file = f"{output_dir}/component_strategy_results.json"
    with open(results_file, "w") as f:
        json.dump(detailed_results, f, indent=2, default=str)

    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info(f"All results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    try:
        results = main()
        logger.info("Component-based strategy example completed successfully!")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
