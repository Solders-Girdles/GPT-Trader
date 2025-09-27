#!/usr/bin/env python3
"""
Complete Pipeline Example for GPT-Trader

This example demonstrates the complete pipeline from optimization to monitoring.
Note: This requires Alpaca credentials to be set up for the monitoring stage.
"""

import os


def run_optimization_example():
    """Run a simple optimization example."""
    print("🚀 Stage 1: Running Optimization Example")
    print("-" * 50)

    try:
        from bot.optimization.config import (
            OptimizationConfig,
            ParameterSpace,
            get_trend_breakout_config,
        )

        # Create a simple optimization config
        strategy_config = get_trend_breakout_config()
        parameter_space = ParameterSpace(
            strategy=strategy_config,
            grid_ranges={
                "donchian_lookback": [40, 55, 70],
                "atr_k": [1.5, 2.0, 2.5],
            },
        )

        config = OptimizationConfig(
            name="pipeline_example",
            symbols=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            method="grid",
            parameter_space=parameter_space,
            max_workers=1,
            create_plots=False,
            save_intermediate=False,
        )

        print(f"✅ Optimization config created: {config.name}")
        print(f"   Symbols: {config.symbols}")
        print(f"   Date range: {config.start_date} to {config.end_date}")
        print(f"   Method: {config.method}")

        return config

    except Exception as e:
        print(f"❌ Optimization setup failed: {e}")
        return None


def run_walk_forward_example():
    """Run a simple walk-forward validation example."""
    print("\n🔍 Stage 2: Walk-Forward Validation Example")
    print("-" * 50)

    try:
        from bot.optimization.walk_forward_validator import WalkForwardConfig

        config = WalkForwardConfig(
            symbols=["AAPL", "MSFT"],
            train_months=6,
            test_months=3,
            step_months=3,
            min_windows=2,
            min_mean_sharpe=0.3,
            max_sharpe_std=1.0,
            max_mean_drawdown=0.2,
        )

        print("✅ Walk-forward config created")
        print(f"   Train months: {config.train_months}")
        print(f"   Test months: {config.test_months}")
        print(f"   Step months: {config.step_months}")
        print(f"   Min windows: {config.min_windows}")

        return config

    except Exception as e:
        print(f"❌ Walk-forward setup failed: {e}")
        return None


def run_deployment_example():
    """Run a simple deployment example."""
    print("\n🚀 Stage 3: Deployment Example")
    print("-" * 50)

    try:
        from bot.optimization.deployment_pipeline import DeploymentConfig

        config = DeploymentConfig(
            symbols=["AAPL", "MSFT"],
            min_sharpe=0.8,
            max_drawdown=0.15,
            min_trades=10,
            max_concurrent_strategies=2,
            validation_period_days=30,
            deployment_budget=5000.0,
            risk_per_strategy=0.02,
        )

        print("✅ Deployment config created")
        print(f"   Min Sharpe: {config.min_sharpe}")
        print(f"   Max drawdown: {config.max_drawdown}")
        print(f"   Max strategies: {config.max_concurrent_strategies}")
        print(f"   Budget: ${config.deployment_budget:,.0f}")

        return config

    except Exception as e:
        print(f"❌ Deployment setup failed: {e}")
        return None


def run_monitoring_example():
    """Run a simple monitoring example."""
    print("\n📈 Stage 4: Monitoring Example")
    print("-" * 50)

    try:
        from bot.monitor.performance_monitor import AlertConfig, PerformanceThresholds

        thresholds = PerformanceThresholds(
            min_sharpe=0.7,
            max_drawdown=0.12,
            min_cagr=0.05,
            max_position_concentration=0.3,
            min_diversification=2,
        )

        alert_config = AlertConfig(
            webhook_enabled=False,
            email_enabled=False,
            slack_enabled=False,
            alert_cooldown_hours=24,
        )

        print("✅ Monitoring config created")
        print(f"   Min Sharpe: {thresholds.min_sharpe}")
        print(f"   Max drawdown: {thresholds.max_drawdown}")
        print(f"   Min CAGR: {thresholds.min_cagr}")
        print(f"   Max position concentration: {thresholds.max_position_concentration}")

        return thresholds, alert_config

    except Exception as e:
        print(f"❌ Monitoring setup failed: {e}")
        return None, None


def check_alpaca_credentials():
    """Check if Alpaca credentials are available."""
    print("\n🔑 Checking Alpaca Credentials")
    print("-" * 30)

    api_key = os.getenv("ALPACA_API_KEY_ID")
    secret_key = os.getenv("ALPACA_API_SECRET_KEY")

    if api_key and secret_key:
        print("✅ Alpaca credentials found")
        print(f"   API Key: {api_key[:8]}...")
        print(f"   Secret Key: {secret_key[:8]}...")
        return True
    else:
        print("❌ Alpaca credentials not found")
        print("   Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY environment variables")
        print("   These are required for paper trading and monitoring")
        return False


def main():
    """Run the complete pipeline example."""
    print("🎯 GPT-Trader Complete Pipeline Example")
    print("=" * 60)

    # Stage 1: Optimization
    opt_config = run_optimization_example()

    # Stage 2: Walk-Forward
    wf_config = run_walk_forward_example()

    # Stage 3: Deployment
    deploy_config = run_deployment_example()

    # Stage 4: Monitoring
    monitor_thresholds, monitor_alerts = run_monitoring_example()

    # Check Alpaca credentials
    alpaca_ready = check_alpaca_credentials()

    # Summary
    print("\n📊 Pipeline Summary")
    print("=" * 30)
    print(f"✅ Optimization: {'Ready' if opt_config else 'Failed'}")
    print(f"✅ Walk-Forward: {'Ready' if wf_config else 'Failed'}")
    print(f"✅ Deployment: {'Ready' if deploy_config else 'Failed'}")
    print(f"✅ Monitoring: {'Ready' if monitor_thresholds else 'Failed'}")
    print(f"✅ Alpaca: {'Ready' if alpaca_ready else 'Not Configured'}")

    if all([opt_config, wf_config, deploy_config, monitor_thresholds]):
        print("\n🎉 All pipeline components are ready!")
        print("\n📝 To run the complete pipeline:")
        print("1. Set up Alpaca credentials (if not already done)")
        print(
            "2. Run: poetry run gpt-trader optimize-new --name example --strategy trend_breakout --symbols AAPL,MSFT --start-date 2023-01-01 --end-date 2023-12-31"
        )
        print(
            "3. Run: poetry run gpt-trader walk-forward --results data/optimization/example/all_results.csv"
        )
        print(
            "4. Run: poetry run gpt-trader deploy --results data/optimization/example/wf_validated.csv"
        )
        print("5. Run: poetry run gpt-trader monitor --min-sharpe 0.7 --max-drawdown 0.12")
    else:
        print("\n⚠️  Some components failed to initialize")
        print("   Check the error messages above and fix any issues")

    return True


if __name__ == "__main__":
    main()
