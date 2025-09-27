#!/usr/bin/env python3
"""
Integrated Backtest Demo for GPT-Trader

This demo shows the complete end-to-end integration of all GPT-Trader components:
- Data Pipeline (INT-002)
- Strategy-Allocator Bridge (INT-001)
- Risk Management Integration (INT-003)
- Orchestrator (INT-004)

The demo runs a complete backtest using the trend breakout strategy on a small
universe of stocks, demonstrating the full data → strategy → risk → execution flow.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.integration.orchestrator import (
    BacktestConfig,
    IntegratedOrchestrator,
    run_integrated_backtest,
)
from bot.logging import get_logger
from bot.portfolio.allocator import PortfolioRules
from bot.risk.integration import RiskConfig
from bot.strategy.demo_ma import DemoMAStrategy

logger = get_logger("demo")


def run_basic_integration_demo():
    """Run a basic integration demo with minimal configuration."""
    print("\n=== GPT-Trader Integrated Backtest Demo ===")
    print("Testing complete integration: Data → Strategy → Risk → Execution → Results\n")

    # Define test universe
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    # Define date range (last 6 months)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=180)

    print(f"Universe: {', '.join(symbols)}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("Initial Capital: $1,000,000\n")

    # Create strategy with reasonable parameters
    strategy = DemoMAStrategy(fast=10, slow=20, atr_period=14)

    print(f"Strategy: {strategy.name}")
    print(
        f"Parameters: Fast MA={strategy.fast}, Slow MA={strategy.slow}, ATR={strategy.atr_period}\n"
    )

    # Run integrated backtest using convenience function
    try:
        print("Starting integrated backtest...")

        results = run_integrated_backtest(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=1_000_000.0,
            show_progress=True,
            quiet_mode=False,
        )

        # Display results
        print("\n=== BACKTEST RESULTS ===")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"CAGR: {results.cagr:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Volatility: {results.volatility:.2%}")
        print(f"Total Trades: {results.total_trades}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Max Concurrent Positions: {results.max_positions}")
        print(f"Average Positions: {results.avg_positions:.1f}")
        print(f"Execution Time: {results.execution_time_seconds:.2f}s")

        if results.warnings:
            print(f"\nWarnings ({len(results.warnings)}):")
            for warning in results.warnings[:5]:  # Show first 5
                print(f"  - {warning}")

        if results.errors:
            print(f"\nErrors ({len(results.errors)}):")
            for error in results.errors:
                print(f"  - {error}")

        print("\n✅ Integration demo completed successfully!")
        print("Output files saved to data/backtests/")

        return results

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)
        return None


def run_advanced_integration_demo():
    """Run an advanced integration demo with custom configuration."""
    print("\n=== Advanced Integration Demo ===")
    print("Testing with custom risk settings and portfolio rules\n")

    # Larger universe for more complex test
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "CRM"]

    # Date range
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365)  # 1 year

    print(f"Universe: {len(symbols)} symbols")
    print(f"Period: {start_date.date()} to {end_date.date()}")

    # Custom risk configuration
    risk_config = RiskConfig(
        max_position_size=0.15,  # 15% max per position
        max_portfolio_exposure=0.90,  # 90% max total exposure
        max_risk_per_trade=0.02,  # 2% risk per trade
        max_daily_loss=0.05,  # 5% max daily loss
        default_stop_loss_pct=0.08,  # 8% stop loss
        use_dynamic_sizing=True,
        stress_test_enabled=True,
    )

    # Custom portfolio rules
    portfolio_rules = PortfolioRules(
        per_trade_risk_pct=0.015,  # 1.5% risk per trade
        max_positions=8,  # Max 8 positions
        max_gross_exposure_pct=0.90,  # 90% max exposure
        atr_k=2.5,  # 2.5x ATR for stops
        cost_bps=8.0,  # 8 bps transaction costs
    )

    # Strategy with different parameters
    strategy = DemoMAStrategy(fast=5, slow=15, atr_period=10)

    # Custom backtest configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=2_000_000.0,  # $2M
        risk_config=risk_config,
        portfolio_rules=portfolio_rules,
        use_cache=True,
        strict_validation=True,
        show_progress=True,
        quiet_mode=False,
        save_trades=True,
        save_portfolio=True,
        save_metrics=True,
        generate_plot=True,
    )

    print("\nAdvanced Configuration:")
    print(f"  - Initial Capital: ${config.initial_capital:,.0f}")
    print(f"  - Max Position Size: {risk_config.max_position_size:.1%}")
    print(f"  - Max Positions: {portfolio_rules.max_positions}")
    print(f"  - Risk Per Trade: {risk_config.max_risk_per_trade:.1%}")
    print(f"  - Transaction Costs: {portfolio_rules.cost_bps} bps")

    try:
        print("\nStarting advanced integrated backtest...")

        # Create orchestrator with custom config
        orchestrator = IntegratedOrchestrator(config)

        # Run health check first
        health = orchestrator.health_check()
        print(f"System Health: {health['status']}")
        if health["warnings"]:
            print(f"Health Warnings: {len(health['warnings'])}")

        # Run backtest
        results = orchestrator.run_backtest(strategy, symbols)

        # Enhanced results display
        print("\n=== ADVANCED BACKTEST RESULTS ===")

        # Performance metrics
        perf_data = [
            ("Total Return", f"{results.total_return:.2%}"),
            ("CAGR", f"{results.cagr:.2%}"),
            ("Sharpe Ratio", f"{results.sharpe_ratio:.3f}"),
            ("Sortino Ratio", f"{results.sortino_ratio:.3f}"),
            ("Calmar Ratio", f"{results.calmar_ratio:.3f}"),
            ("Max Drawdown", f"{results.max_drawdown:.2%}"),
            ("Volatility", f"{results.volatility:.2%}"),
        ]

        print("\nPerformance Metrics:")
        for metric, value in perf_data:
            print(f"  {metric:<15}: {value}")

        # Trading statistics
        trading_data = [
            ("Total Trades", f"{results.total_trades}"),
            ("Winning Trades", f"{results.winning_trades}"),
            ("Losing Trades", f"{results.losing_trades}"),
            ("Win Rate", f"{results.win_rate:.2%}"),
            ("Avg Win", f"${results.avg_win:,.2f}" if results.avg_win else "N/A"),
            ("Avg Loss", f"${results.avg_loss:,.2f}" if results.avg_loss else "N/A"),
            ("Profit Factor", f"{results.profit_factor:.2f}" if results.profit_factor else "N/A"),
        ]

        print("\nTrading Statistics:")
        for metric, value in trading_data:
            print(f"  {metric:<15}: {value}")

        # Risk metrics
        print("\nRisk Metrics:")
        print(f"  Max Positions   : {results.max_positions}")
        print(f"  Avg Positions   : {results.avg_positions:.1f}")
        print(f"  Total Costs     : ${results.total_costs:,.2f}")

        # Additional info
        print("\nExecution Info:")
        print(f"  Symbols Traded  : {len(results.symbols_traded)}")
        print(f"  Duration        : {results.duration_days} days")
        print(f"  Execution Time  : {results.execution_time_seconds:.2f}s")

        if results.warnings:
            print(f"\n⚠️  Warnings ({len(results.warnings)}):")
            for i, warning in enumerate(results.warnings[:3], 1):
                print(f"  {i}. {warning}")
            if len(results.warnings) > 3:
                print(f"  ... and {len(results.warnings) - 3} more")

        if results.errors:
            print(f"\n❌ Errors ({len(results.errors)}):")
            for error in results.errors:
                print(f"  - {error}")
        else:
            print("\n✅ Advanced integration demo completed successfully!")
            print("All components integrated properly with enhanced risk management.")

        return results

    except Exception as e:
        print(f"\n❌ Advanced demo failed: {e}")
        logger.error(f"Advanced demo failed: {e}", exc_info=True)
        return None


def run_component_validation():
    """Validate that all integration components are working."""
    print("\n=== Component Integration Validation ===")

    try:
        from bot.dataflow.pipeline import DataPipeline
        from bot.integration.strategy_allocator_bridge import StrategyAllocatorBridge
        from bot.portfolio.allocator import PortfolioRules
        from bot.risk.integration import RiskIntegration

        # Test data pipeline
        print("✓ Data Pipeline import OK")
        pipeline = DataPipeline()
        pipeline_health = pipeline.health_check()
        print(f"✓ Data Pipeline health: {pipeline_health['status']}")

        # Test strategy-allocator bridge
        print("✓ Strategy-Allocator Bridge import OK")
        strategy = DemoMAStrategy(fast=10, slow=20)
        rules = PortfolioRules()
        bridge = StrategyAllocatorBridge(strategy, rules)
        bridge_valid = bridge.validate_configuration()
        print(f"✓ Strategy-Allocator Bridge validation: {bridge_valid}")

        # Test risk integration
        print("✓ Risk Integration import OK")
        risk_integration = RiskIntegration()
        risk_report = risk_integration.generate_risk_report()
        print(f"✓ Risk Integration report generated: {len(risk_report)} fields")

        # Test orchestrator
        print("✓ Orchestrator import OK")
        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30), end_date=datetime.now()
        )
        orchestrator = IntegratedOrchestrator(config)
        orch_health = orchestrator.health_check()
        print(f"✓ Orchestrator health: {orch_health['status']}")

        print("\n🎉 All integration components validated successfully!")
        print("The system is ready for end-to-end backtesting.")

        return True

    except Exception as e:
        print(f"\n❌ Component validation failed: {e}")
        logger.error(f"Component validation failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT-Trader Integration Demo")
    parser.add_argument(
        "--mode",
        choices=["basic", "advanced", "validate", "all"],
        default="basic",
        help="Demo mode to run",
    )

    args = parser.parse_args()

    print("GPT-Trader Integration Demo")
    print("===========================")
    print("This demo shows the complete integration of all components:")
    print("• Data Pipeline (fetch & validate market data)")
    print("• Strategy Execution (generate trading signals)")
    print("• Risk Management (validate & adjust allocations)")
    print("• Portfolio Management (execute trades & track performance)")
    print("• Performance Reporting (comprehensive metrics)")

    success = True

    if args.mode in ["validate", "all"]:
        success &= run_component_validation()

    if args.mode in ["basic", "all"]:
        result = run_basic_integration_demo()
        success &= result is not None

    if args.mode in ["advanced", "all"]:
        result = run_advanced_integration_demo()
        success &= result is not None

    print("\n" + "=" * 50)
    if success:
        print("🎉 Integration demo completed successfully!")
        print("All components are working together properly.")
        print("\nNext steps:")
        print("• Review output files in data/backtests/")
        print("• Experiment with different strategies and parameters")
        print("• Add more symbols to the universe")
        print("• Customize risk management settings")
    else:
        print("❌ Demo encountered errors.")
        print("Please check the logs for details.")
        sys.exit(1)
