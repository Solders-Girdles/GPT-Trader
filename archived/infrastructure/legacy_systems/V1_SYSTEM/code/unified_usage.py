"""
Unified Orchestrator Usage Examples

Demonstrates how to use the UnifiedOrchestrator as a single entry point
for all GPT-Trader functionality.
"""

import asyncio
from datetime import datetime, timedelta

from bot.orchestrator import (
    UnifiedOrchestrator,
    UnifiedConfig,
    SystemMode,
    create_unified_orchestrator,
    setup_graceful_shutdown,
)


async def example_paper_trading():
    """Example: Start paper trading with default configuration"""
    print("=== Paper Trading Example ===")
    
    # Create orchestrator with simple configuration
    orchestrator = create_unified_orchestrator(
        mode=SystemMode.PAPER,
        symbols=["AAPL", "MSFT", "GOOGL"],
        initial_capital=100_000.0,
        enable_ml=True,
        enable_dashboard=True,
    )
    
    try:
        # Start the system
        await orchestrator.start(strategy="trend_breakout")
        
        # Check status
        status = orchestrator.get_status()
        print(f"System Status: {status['system']['state']}")
        print(f"Strategy: {status['trading']['current_strategy']}")
        print(f"Active Components: {sum(status['components'].values())}/6")
        
        # Simulate running for a short time
        print("Running for 10 seconds...")
        await asyncio.sleep(10)
        
        # Get updated status
        status = orchestrator.get_status()
        print(f"Uptime: {status['system']['uptime_hours']:.3f} hours")
        print(f"Total Value: ${status['trading']['total_value']:,.2f}")
        
    finally:
        # Always shutdown gracefully
        await orchestrator.stop(graceful=True)
        print("System shutdown complete")


async def example_backtest():
    """Example: Run a backtest using the unified orchestrator"""
    print("\n=== Backtest Example ===")
    
    # Create orchestrator for backtesting
    config = UnifiedConfig(
        mode=SystemMode.BACKTEST,
        initial_capital=1_000_000.0,
        symbols=["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"],
        enable_ml=False,  # Disable ML for simple backtest
        enable_dashboard=False,
        save_trades=True,
        output_dir="data/backtest_results",
    )
    
    orchestrator = UnifiedOrchestrator(config)
    
    try:
        # Initialize system
        await orchestrator.start()
        
        # Run backtest
        results = await orchestrator.run_backtest(
            strategy="demo_ma",
            symbols=["AAPL", "MSFT", "TSLA"],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
        )
        
        # Display results
        print("Backtest Results:")
        print(f"  Total Return: {results['performance']['total_return']:.2%}")
        print(f"  CAGR: {results['performance']['cagr']:.2%}")
        print(f"  Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['performance']['max_drawdown']:.2%}")
        print(f"  Total Trades: {results['trading']['total_trades']}")
        print(f"  Win Rate: {results['trading']['win_rate']:.2%}")
        
    finally:
        await orchestrator.stop()


async def example_ml_enhanced_trading():
    """Example: ML-enhanced trading with strategy selection"""
    print("\n=== ML-Enhanced Trading Example ===")
    
    # Create orchestrator with ML features enabled
    config = UnifiedConfig(
        mode=SystemMode.PAPER,
        symbols=["AAPL", "MSFT", "TSLA", "NVDA"],
        initial_capital=500_000.0,
        enable_ml=True,
        enable_ml_strategy_selection=True,
        confidence_threshold=0.7,
        enable_auto_retraining=True,
        max_daily_loss=0.01,  # 1% daily loss limit
        enable_circuit_breakers=True,
        enable_dashboard=True,
        dashboard_port=8502,
    )
    
    orchestrator = UnifiedOrchestrator(config)
    
    try:
        # Setup graceful shutdown
        setup_graceful_shutdown(orchestrator)
        
        # Start with ML-driven strategy selection
        await orchestrator.start()
        
        # Monitor for a longer period
        print("ML-enhanced trading started...")
        print("Check dashboard at http://localhost:8502")
        
        for i in range(6):  # Monitor for 1 minute
            await asyncio.sleep(10)
            
            status = orchestrator.get_status()
            print(f"[{i*10}s] Strategy: {status['trading']['current_strategy']}, "
                  f"Value: ${status['trading']['total_value']:,.2f}, "
                  f"P&L: ${status['trading']['daily_pnl']:,.2f}")
            
            # Example: Execute a manual trade signal
            if i == 2:
                signal = {
                    'symbol': 'AAPL',
                    'action': 'buy',
                    'quantity': 100,
                    'reason': 'manual_signal',
                }
                success = await orchestrator.execute_trade(signal)
                print(f"Manual trade executed: {success}")
        
    finally:
        await orchestrator.stop()


async def example_health_monitoring():
    """Example: System health monitoring and diagnostics"""
    print("\n=== Health Monitoring Example ===")
    
    orchestrator = create_unified_orchestrator(
        mode=SystemMode.PAPER,
        health_check_interval=5,  # Check every 5 seconds
    )
    
    try:
        await orchestrator.start()
        
        # Monitor health for 30 seconds
        for i in range(6):
            await asyncio.sleep(5)
            
            status = orchestrator.get_status()
            system = status['system']
            components = status['components']
            
            print(f"[{i*5}s] Health Check:")
            print(f"  State: {system['state']}")
            print(f"  Uptime: {system['uptime_hours']:.3f}h")
            print(f"  Components Healthy: {sum(components.values())}/{len(components)}")
            
            if system['warnings']:
                print(f"  Recent Warnings: {system['warnings']}")
            
            if system['error_count'] > 0:
                print(f"  Error Count: {system['error_count']}")
        
    finally:
        await orchestrator.stop()


async def example_configuration_variations():
    """Example: Different configuration patterns"""
    print("\n=== Configuration Variations ===")
    
    # Minimal configuration
    print("1. Minimal Configuration:")
    minimal = create_unified_orchestrator()
    await minimal.start()
    status = minimal.get_status()
    print(f"   Mode: {status['system']['mode']}")
    print(f"   Strategy: {status['trading']['current_strategy']}")
    await minimal.stop()
    
    # High-performance configuration
    print("2. High-Performance Configuration:")
    high_perf_config = UnifiedConfig(
        mode=SystemMode.PAPER,
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
        initial_capital=10_000_000.0,
        enable_ml=True,
        confidence_threshold=0.8,
        max_position_size=0.05,  # 5% max per position
        enable_circuit_breakers=True,
        health_check_interval=60,  # Check every minute
        save_trades=True,
        save_positions=True,
    )
    
    high_perf = UnifiedOrchestrator(high_perf_config)
    await high_perf.start(strategy="trend_breakout")
    status = high_perf.get_status()
    print(f"   Capital: ${status['trading']['total_value']:,.0f}")
    print(f"   ML Enabled: {high_perf_config.enable_ml}")
    await high_perf.stop()
    
    # Conservative configuration
    print("3. Conservative Configuration:")
    conservative_config = UnifiedConfig(
        mode=SystemMode.PAPER,
        symbols=["SPY", "QQQ", "IWM"],  # ETFs only
        initial_capital=100_000.0,
        enable_ml=False,  # Simple strategies only
        max_daily_loss=0.005,  # 0.5% daily loss limit
        max_position_size=0.2,  # 20% max per position
        enable_circuit_breakers=True,
        trading_hours_only=True,
    )
    
    conservative = UnifiedOrchestrator(conservative_config)
    await conservative.start(strategy="demo_ma")
    status = conservative.get_status()
    print(f"   Risk Level: Conservative")
    print(f"   Daily Loss Limit: {conservative_config.max_daily_loss:.1%}")
    await conservative.stop()


async def main():
    """Run all examples"""
    print("GPT-Trader Unified Orchestrator Examples")
    print("=" * 50)
    
    try:
        # Run examples in sequence
        await example_paper_trading()
        await example_backtest()
        await example_ml_enhanced_trading()
        await example_health_monitoring()
        await example_configuration_variations()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())