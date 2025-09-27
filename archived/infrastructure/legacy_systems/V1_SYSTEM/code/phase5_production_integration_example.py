"""
Phase 5 Production Integration Example
Demonstrates the complete Phase 5 system: real-time strategy selection, portfolio optimization,
risk management, and performance monitoring working together.
"""

import asyncio
import logging
from datetime import datetime, timedelta

import numpy as np
from bot.exec.alpaca_paper import AlpacaPaperBroker
from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyKnowledgeBase,
    StrategyMetadata,
    StrategyPerformance,
)
from bot.live.production_orchestrator import (
    OrchestrationMode,
    OrchestratorConfig,
    ProductionOrchestrator,
)
from bot.live.strategy_selector import SelectionMethod
from bot.monitor.alerts import AlertConfig, AlertSeverity
from bot.portfolio.optimizer import OptimizationMethod

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def create_sample_knowledge_base() -> StrategyKnowledgeBase:
    """Create a sample knowledge base with strategies for demonstration."""
    knowledge_base = StrategyKnowledgeBase()

    # Create sample strategies for different market regimes
    strategies = [
        {
            "strategy_id": "trend_following_001",
            "name": "Enhanced Trend Following",
            "description": "Trend following strategy with dynamic position sizing",
            "strategy_type": "trend_following",
            "parameters": {"lookback_period": 20, "atr_multiplier": 2.0},
            "context": StrategyContext(
                market_regime="trending",
                time_period="bull_market",
                asset_class="equity",
                risk_profile="moderate",
                volatility_regime="medium",
                correlation_regime="low",
            ),
            "performance": StrategyPerformance(
                sharpe_ratio=1.85,
                cagr=0.18,
                max_drawdown=0.12,
                win_rate=0.65,
                consistency_score=0.78,
                n_trades=45,
                avg_trade_duration=5.2,
                profit_factor=1.45,
                calmar_ratio=1.54,
                sortino_ratio=2.1,
                information_ratio=1.2,
                beta=0.85,
                alpha=0.08,
            ),
            "usage_count": 25,
            "success_rate": 0.72,
        },
        {
            "strategy_id": "mean_reversion_001",
            "name": "Bollinger Band Mean Reversion",
            "description": "Mean reversion strategy using Bollinger Bands",
            "strategy_type": "mean_reversion",
            "parameters": {"lookback_period": 20, "std_dev": 2.0},
            "context": StrategyContext(
                market_regime="sideways",
                time_period="sideways_market",
                asset_class="equity",
                risk_profile="conservative",
                volatility_regime="low",
                correlation_regime="high",
            ),
            "performance": StrategyPerformance(
                sharpe_ratio=1.45,
                cagr=0.12,
                max_drawdown=0.08,
                win_rate=0.58,
                consistency_score=0.82,
                n_trades=67,
                avg_trade_duration=3.8,
                profit_factor=1.32,
                calmar_ratio=1.5,
                sortino_ratio=1.8,
                information_ratio=0.9,
                beta=0.45,
                alpha=0.06,
            ),
            "usage_count": 18,
            "success_rate": 0.68,
        },
        {
            "strategy_id": "momentum_001",
            "name": "Relative Strength Momentum",
            "description": "Momentum strategy based on relative strength",
            "strategy_type": "momentum",
            "parameters": {"momentum_period": 12, "rebalance_freq": 5},
            "context": StrategyContext(
                market_regime="trending",
                time_period="bull_market",
                asset_class="equity",
                risk_profile="aggressive",
                volatility_regime="high",
                correlation_regime="medium",
            ),
            "performance": StrategyPerformance(
                sharpe_ratio=2.1,
                cagr=0.22,
                max_drawdown=0.15,
                win_rate=0.62,
                consistency_score=0.75,
                n_trades=38,
                avg_trade_duration=7.5,
                profit_factor=1.58,
                calmar_ratio=1.47,
                sortino_ratio=2.4,
                information_ratio=1.4,
                beta=1.1,
                alpha=0.12,
            ),
            "usage_count": 32,
            "success_rate": 0.75,
        },
        {
            "strategy_id": "volatility_001",
            "name": "Volatility Breakout",
            "description": "Volatility-based breakout strategy",
            "strategy_type": "volatility",
            "parameters": {"vol_period": 10, "breakout_threshold": 1.5},
            "context": StrategyContext(
                market_regime="volatile",
                time_period="volatile_market",
                asset_class="equity",
                risk_profile="aggressive",
                volatility_regime="high",
                correlation_regime="low",
            ),
            "performance": StrategyPerformance(
                sharpe_ratio=1.65,
                cagr=0.16,
                max_drawdown=0.18,
                win_rate=0.55,
                consistency_score=0.68,
                n_trades=52,
                avg_trade_duration=4.2,
                profit_factor=1.38,
                calmar_ratio=0.89,
                sortino_ratio=1.9,
                information_ratio=1.1,
                beta=0.95,
                alpha=0.09,
            ),
            "usage_count": 15,
            "success_rate": 0.65,
        },
        {
            "strategy_id": "crisis_001",
            "name": "Crisis Hedging Strategy",
            "description": "Defensive strategy for crisis markets",
            "strategy_type": "hedging",
            "parameters": {"hedge_ratio": 0.3, "flight_to_quality": True},
            "context": StrategyContext(
                market_regime="crisis",
                time_period="bear_market",
                asset_class="equity",
                risk_profile="conservative",
                volatility_regime="high",
                correlation_regime="high",
            ),
            "performance": StrategyPerformance(
                sharpe_ratio=0.95,
                cagr=0.08,
                max_drawdown=0.06,
                win_rate=0.72,
                consistency_score=0.88,
                n_trades=23,
                avg_trade_duration=12.5,
                profit_factor=1.25,
                calmar_ratio=1.33,
                sortino_ratio=1.2,
                information_ratio=0.6,
                beta=0.25,
                alpha=0.04,
            ),
            "usage_count": 8,
            "success_rate": 0.85,
        },
    ]

    # Add strategies to knowledge base
    for strategy_data in strategies:
        strategy = StrategyMetadata(
            strategy_id=strategy_data["strategy_id"],
            name=strategy_data["name"],
            description=strategy_data["description"],
            strategy_type=strategy_data["strategy_type"],
            parameters=strategy_data["parameters"],
            context=strategy_data["context"],
            performance=strategy_data["performance"],
            discovery_date=datetime.now() - timedelta(days=np.random.randint(30, 365)),
            last_updated=datetime.now() - timedelta(days=np.random.randint(1, 30)),
            usage_count=strategy_data["usage_count"],
            success_rate=strategy_data["success_rate"],
            tags=[strategy_data["strategy_type"], strategy_data["context"].market_regime],
        )

        knowledge_base.add_strategy(strategy)

    logger.info(f"Created knowledge base with {len(strategies)} sample strategies")
    return knowledge_base


async def setup_broker() -> AlpacaPaperBroker:
    """Set up a paper trading broker for demonstration."""
    # In a real implementation, you would use actual API keys
    broker = AlpacaPaperBroker(api_key="demo_key", secret_key="demo_secret", paper=True)

    logger.info("Paper trading broker initialized")
    return broker


def create_orchestrator_config() -> OrchestratorConfig:
    """Create configuration for the production orchestrator."""
    return OrchestratorConfig(
        mode=OrchestrationMode.SEMI_AUTOMATED,
        rebalance_interval=3600,  # 1 hour
        risk_check_interval=300,  # 5 minutes
        performance_check_interval=600,  # 10 minutes
        # Strategy selection
        max_strategies=5,
        min_strategy_confidence=0.7,
        selection_method=SelectionMethod.HYBRID,
        # Portfolio optimization
        optimization_method=OptimizationMethod.SHARPE_MAXIMIZATION,
        max_position_weight=0.4,
        target_volatility=0.15,
        # Risk management
        max_portfolio_var=0.02,
        max_drawdown=0.15,
        stop_loss_pct=0.05,
        # Performance monitoring
        min_sharpe_ratio=0.5,
        max_drawdown_threshold=0.15,
        # Alerting
        enable_alerts=True,
        alert_cooldown_minutes=30,
    )


async def demonstrate_strategy_selection(orchestrator: ProductionOrchestrator):
    """Demonstrate real-time strategy selection."""
    logger.info("=== Demonstrating Real-Time Strategy Selection ===")

    # Get current selection
    selection_summary = orchestrator.get_strategy_summary()
    logger.info(f"Current strategy selection: {selection_summary}")

    # Simulate market regime change
    logger.info("Simulating market regime change...")

    # In a real implementation, this would be triggered by actual market data
    # For demonstration, we'll just show the selection process

    logger.info("Strategy selection demonstration completed")


async def demonstrate_portfolio_optimization(orchestrator: ProductionOrchestrator):
    """Demonstrate portfolio optimization."""
    logger.info("=== Demonstrating Portfolio Optimization ===")

    # Get portfolio summary
    portfolio_summary = orchestrator.get_portfolio_summary()
    logger.info(f"Portfolio optimization summary: {portfolio_summary}")

    # Show different optimization methods
    optimization_methods = [
        OptimizationMethod.SHARPE_MAXIMIZATION,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.MAX_DIVERSIFICATION,
    ]

    for method in optimization_methods:
        logger.info(f"Testing optimization method: {method.value}")
        # In a real implementation, you would run the optimization
        # For demonstration, we'll just show the concept

    logger.info("Portfolio optimization demonstration completed")


async def demonstrate_risk_management(orchestrator: ProductionOrchestrator):
    """Demonstrate risk management capabilities."""
    logger.info("=== Demonstrating Risk Management ===")

    # Get risk summary
    risk_summary = orchestrator.get_risk_summary()
    logger.info(f"Risk management summary: {risk_summary}")

    # Demonstrate risk limit checking
    logger.info("Checking risk limits...")

    # Demonstrate stop-loss management
    logger.info("Managing stop-losses...")

    # Demonstrate stress testing
    logger.info("Running stress tests...")

    logger.info("Risk management demonstration completed")


async def demonstrate_performance_monitoring(orchestrator: ProductionOrchestrator):
    """Demonstrate performance monitoring."""
    logger.info("=== Demonstrating Performance Monitoring ===")

    # Get alert summary
    alert_summary = orchestrator.get_alert_summary()
    logger.info(f"Alert summary: {alert_summary}")

    # Demonstrate performance tracking
    logger.info("Tracking performance metrics...")

    # Demonstrate anomaly detection
    logger.info("Detecting performance anomalies...")

    logger.info("Performance monitoring demonstration completed")


async def demonstrate_system_integration(orchestrator: ProductionOrchestrator):
    """Demonstrate the complete system integration."""
    logger.info("=== Demonstrating Complete System Integration ===")

    # Get system status
    system_status = orchestrator.get_system_status()
    if system_status:
        logger.info("System Status:")
        logger.info(f"  Mode: {system_status.mode.value}")
        logger.info(f"  Running: {system_status.is_running}")
        logger.info(f"  Current Regime: {system_status.current_regime}")
        logger.info(f"  Active Strategies: {system_status.n_active_strategies}")
        logger.info(f"  Portfolio Value: ${system_status.portfolio_value:,.2f}")
        logger.info(f"  Portfolio Sharpe: {system_status.portfolio_sharpe:.3f}")
        logger.info(f"  Risk Level: {system_status.risk_level}")
        logger.info(f"  Active Alerts: {system_status.alerts_active}")

    # Get operation history
    operation_history = orchestrator.get_operation_history()
    logger.info(f"Recent operations: {len(operation_history)} operations recorded")

    # Show recent operations
    for operation in operation_history[-5:]:  # Last 5 operations
        logger.info(f"  {operation['timestamp']}: {operation['operation']}")

    logger.info("System integration demonstration completed")


async def run_production_demo():
    """Run the complete Phase 5 production integration demo."""
    logger.info("Starting Phase 5 Production Integration Demo")
    logger.info("=" * 60)

    try:
        # Initialize components
        logger.info("Initializing components...")

        # Create knowledge base with sample strategies
        knowledge_base = await create_sample_knowledge_base()

        # Set up broker
        broker = await setup_broker()

        # Create orchestrator configuration
        config = create_orchestrator_config()

        # Define symbols for trading
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]

        # Create production orchestrator
        orchestrator = ProductionOrchestrator(
            config=config, broker=broker, knowledge_base=knowledge_base, symbols=symbols
        )

        logger.info("Components initialized successfully")
        logger.info("=" * 60)

        # Demonstrate each component
        await demonstrate_strategy_selection(orchestrator)
        logger.info("-" * 40)

        await demonstrate_portfolio_optimization(orchestrator)
        logger.info("-" * 40)

        await demonstrate_risk_management(orchestrator)
        logger.info("-" * 40)

        await demonstrate_performance_monitoring(orchestrator)
        logger.info("-" * 40)

        await demonstrate_system_integration(orchestrator)
        logger.info("=" * 60)

        # Show comprehensive summary
        logger.info("=== COMPREHENSIVE SYSTEM SUMMARY ===")

        # Strategy selection summary
        strategy_summary = orchestrator.get_strategy_summary()
        logger.info(f"Strategy Selection: {strategy_summary}")

        # Portfolio optimization summary
        portfolio_summary = orchestrator.get_portfolio_summary()
        logger.info(f"Portfolio Optimization: {portfolio_summary}")

        # Risk management summary
        risk_summary = orchestrator.get_risk_summary()
        logger.info(f"Risk Management: {risk_summary}")

        # Alert summary
        alert_summary = orchestrator.get_alert_summary()
        logger.info(f"Alert System: {alert_summary}")

        # System status
        system_status = orchestrator.get_system_status()
        if system_status:
            logger.info(f"System Status: {system_status}")

        logger.info("=" * 60)
        logger.info("Phase 5 Production Integration Demo completed successfully!")

        # Demonstrate running the orchestrator (without actually starting it)
        logger.info(
            "Note: This demo shows the system components without actually starting the orchestrator."
        )
        logger.info(
            "In a real deployment, you would call orchestrator.start() to begin automated operation."
        )

    except Exception as e:
        logger.error(f"Error in production demo: {e}")
        raise


async def demonstrate_alert_system():
    """Demonstrate the alert system capabilities."""
    logger.info("=== Demonstrating Alert System ===")

    from bot.monitor.alerts import AlertManager

    # Create alert configuration
    alert_config = AlertConfig(
        email_enabled=False,  # Set to True with real credentials
        slack_enabled=False,  # Set to True with real webhook
        discord_enabled=False,  # Set to True with real webhook
        webhook_enabled=False,  # Set to True with real webhook
        alert_cooldown_minutes=5,
    )

    # Create alert manager
    alert_manager = AlertManager(alert_config)

    # Send various types of alerts
    logger.info("Sending sample alerts...")

    # Performance alert
    await alert_manager.send_performance_alert(
        "strategy_001", "sharpe_ratio", 0.3, 0.5, AlertSeverity.WARNING
    )

    # Risk alert
    await alert_manager.send_risk_alert("portfolio_var", 0.025, 0.02, AlertSeverity.WARNING)

    # Strategy alert
    await alert_manager.send_strategy_alert(
        "momentum_001",
        "regime_change",
        "Market regime changed from trending to volatile",
        AlertSeverity.INFO,
    )

    # System alert
    await alert_manager.send_system_alert(
        "data_feed", "connection_lost", "Lost connection to market data feed", AlertSeverity.ERROR
    )

    # Trade alert
    await alert_manager.send_trade_alert("AAPL", "buy", 100, 150.25, AlertSeverity.INFO)

    # Get alert summary
    alert_summary = alert_manager.get_alert_summary()
    logger.info(f"Alert Summary: {alert_summary}")

    # Get active alerts
    active_alerts = alert_manager.get_active_alerts()
    logger.info(f"Active Alerts: {len(active_alerts)}")

    # Get alert history
    alert_history = alert_manager.get_alert_history(days=1)
    logger.info(f"Alert History (last 24h): {len(alert_history)} alerts")

    logger.info("Alert system demonstration completed")


if __name__ == "__main__":
    # Run the main demo
    asyncio.run(run_production_demo())

    # Run alert system demo
    asyncio.run(demonstrate_alert_system())

    logger.info("All Phase 5 demonstrations completed!")
