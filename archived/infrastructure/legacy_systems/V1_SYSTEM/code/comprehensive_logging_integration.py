#!/usr/bin/env python3
"""Comprehensive logging integration example for GPT-Trader.

This example demonstrates:
- Setting up structured logging across all components
- Real-time monitoring and alerting
- Database persistence and querying
- Correlation tracking for distributed operations
- Performance metrics collection
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from bot.config import get_config, set_config, TradingConfig
from bot.logging import (
    TradingLogger,
    StrategyLogger, 
    MLLogger,
    RiskLogger,
    SystemLogger,
    LogQueryInterface,
    get_log_monitor,
    Alert
)


def setup_comprehensive_logging():
    """Setup comprehensive logging for the entire system."""
    
    # Configure logging in config
    config = get_config()
    config.logging.structured_logging = True
    config.logging.log_trades = True
    config.logging.log_performance = True
    config.logging.file_path = Path("logs/gpt_trader.log")
    
    # Start the global log monitor
    monitor = get_log_monitor()
    monitor.start()
    
    print("✅ Comprehensive logging system initialized")
    return monitor


def demonstrate_trading_logging():
    """Demonstrate trading-specific logging."""
    print("\n🔄 Demonstrating Trading Logging...")
    
    trading_logger = TradingLogger()
    
    # Log signal generation
    trading_logger.log_signal_generated(
        strategy_name="demo_ma",
        symbol="AAPL",
        signal_type="buy",
        confidence=0.75,
        signal_data={
            "sma_20": 150.25,
            "sma_50": 148.50,
            "current_price": 152.30,
            "volume": 1500000
        }
    )
    
    # Log order submission
    order_id = f"ORD-{uuid.uuid4().hex[:8]}"
    trading_logger.log_order_submitted(
        order_id=order_id,
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type="market",
        strategy_name="demo_ma"
    )
    
    # Simulate order execution
    time.sleep(0.1)
    execution_id = f"EXEC-{uuid.uuid4().hex[:8]}"
    trading_logger.log_order_execution(
        order_id=order_id,
        execution_id=execution_id,
        symbol="AAPL",
        side="buy",
        quantity=100,
        execution_price=152.35,
        commission=1.00,
        slippage=0.05
    )
    
    # Log position update
    trading_logger.log_position_update(
        symbol="AAPL",
        position_change=100,
        new_position=100,
        avg_cost=152.35,
        unrealized_pnl=-5.00,
        market_price=152.30
    )
    
    # Simulate closing position later
    time.sleep(0.1)
    trading_logger.log_pnl_realization(
        symbol="AAPL",
        quantity_closed=100,
        entry_price=152.35,
        exit_price=154.80,
        realized_pnl=245.00,  # 100 * (154.80 - 152.35)
        holding_period_days=2.5
    )
    
    print("✅ Trading events logged successfully")


def demonstrate_strategy_logging():
    """Demonstrate strategy-specific logging."""
    print("\n🧠 Demonstrating Strategy Logging...")
    
    strategy_logger = StrategyLogger("trend_breakout")
    
    # Log strategy start
    strategy_logger.log_strategy_start({
        "fast_period": 10,
        "slow_period": 20,
        "atr_period": 14,
        "breakout_multiplier": 2.0,
        "risk_per_trade": 0.01
    })
    
    # Log data processing
    start_time = time.time()
    # Simulate data processing
    time.sleep(0.05)
    processing_time = (time.time() - start_time) * 1000
    
    strategy_logger.log_data_processing(
        symbol="TSLA",
        data_points=500,
        processing_time_ms=processing_time,
        indicators_calculated=["SMA", "ATR", "RSI", "Bollinger_Bands"]
    )
    
    # Log decision process
    strategy_logger.log_decision_process(
        symbol="TSLA",
        decision="buy",
        confidence=0.82,
        decision_factors={
            "price_above_sma": True,
            "atr_breakout": True,
            "rsi_oversold": False,
            "volume_surge": True,
            "trend_strength": 0.75
        }
    )
    
    # Log performance update
    strategy_logger.log_performance_update(
        total_return=12.5,
        sharpe_ratio=1.85,
        max_drawdown=8.2,
        win_rate=68.5,
        total_trades=47
    )
    
    print("✅ Strategy events logged successfully")


def demonstrate_ml_logging():
    """Demonstrate ML-specific logging."""
    print("\n🤖 Demonstrating ML Logging...")
    
    ml_logger = MLLogger()
    
    # Log model training start
    ml_logger.log_model_training_start(
        model_name="lstm_price_predictor",
        training_samples=50000,
        features=["price", "volume", "rsi", "macd", "bb_position", "atr"],
        hyperparameters={
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32
        }
    )
    
    # Simulate training time
    time.sleep(0.1)
    
    # Log training completion
    ml_logger.log_model_training_complete(
        model_name="lstm_price_predictor",
        training_time_seconds=3600,
        final_loss=0.0025,
        validation_score=0.78,
        model_size_mb=12.5
    )
    
    # Log predictions
    for i in range(3):
        ml_logger.log_prediction(
            model_name="lstm_price_predictor",
            input_features={
                "price": 152.30 + i,
                "volume": 1500000,
                "rsi": 65.2,
                "macd": 0.85
            },
            prediction=0.75 + i * 0.1,
            confidence=0.82,
            prediction_time_ms=2.5
        )
        time.sleep(0.01)
    
    # Log model performance
    ml_logger.log_model_performance(
        model_name="lstm_price_predictor",
        metric_name="accuracy",
        metric_value=0.78,
        data_period="2024-01-01_to_2024-08-01",
        sample_size=15000
    )
    
    # Log drift detection
    ml_logger.log_model_drift_detection(
        model_name="lstm_price_predictor",
        drift_score=0.15,
        drift_threshold=0.20,
        drift_detected=False,
        features_drifted=[]
    )
    
    print("✅ ML events logged successfully")


def demonstrate_risk_logging():
    """Demonstrate risk management logging."""
    print("\n⚠️ Demonstrating Risk Logging...")
    
    risk_logger = RiskLogger()
    
    # Log risk checks
    risk_logger.log_risk_check(
        check_type="position_size",
        passed=True,
        risk_metric="position_value",
        current_value=15000.0,
        limit_value=25000.0,
        symbol="AAPL"
    )
    
    risk_logger.log_risk_check(
        check_type="portfolio_concentration",
        passed=False,
        risk_metric="single_position_percent",
        current_value=0.35,
        limit_value=0.30,
        symbol="TSLA"
    )
    
    # Log circuit breaker
    risk_logger.log_circuit_breaker_trigger(
        breaker_type="daily_loss_limit",
        trigger_reason="max_daily_loss_exceeded",
        trigger_value=5500.0,
        threshold_value=5000.0,
        action_taken="halt_all_trading"
    )
    
    # Log VaR calculation
    risk_logger.log_var_calculation(
        portfolio_value=100000.0,
        var_1d=2500.0,
        var_5d=5600.0,
        confidence_level=0.95,
        calculation_method="historical_simulation"
    )
    
    print("✅ Risk events logged successfully")


def demonstrate_system_logging():
    """Demonstrate system health logging."""
    print("\n🖥️ Demonstrating System Logging...")
    
    system_logger = SystemLogger()
    
    # Log system startup
    system_logger.log_system_startup(
        components=["data_pipeline", "strategy_engine", "risk_manager", "execution_engine"],
        startup_time_seconds=12.5
    )
    
    # Log resource usage
    system_logger.log_resource_usage(
        cpu_percent=45.2,
        memory_percent=62.8,
        disk_percent=78.5,
        network_bytes_sent=1024*1024*15,  # 15MB
        network_bytes_recv=1024*1024*8    # 8MB
    )
    
    # Log health checks
    system_logger.log_health_check(
        component="database",
        status="healthy",
        response_time_ms=15.2,
        details={"connections": 5, "query_time_avg": 12.5}
    )
    
    system_logger.log_health_check(
        component="market_data_feed",
        status="degraded",
        response_time_ms=450.0,
        details={"latency_high": True, "connection_drops": 2}
    )
    
    # Log error rates
    system_logger.log_error_rate(
        component="execution_engine",
        error_count=3,
        total_requests=150,
        time_window_minutes=60
    )
    
    print("✅ System events logged successfully")


def demonstrate_query_interface():
    """Demonstrate log querying and analysis."""
    print("\n🔍 Demonstrating Log Query Interface...")
    
    # Wait a moment for logs to be persisted
    time.sleep(0.5)
    
    query_interface = LogQueryInterface()
    
    # Query recent logs
    recent_logs = query_interface.query_logs(
        start_time=datetime.now() - timedelta(minutes=5),
        limit=10
    )
    print(f"Recent logs: {len(recent_logs)} entries")
    
    # Get error summary
    error_summary = query_interface.get_error_summary(hours_back=1)
    print(f"Error summary: {error_summary['total_errors']} errors in last hour")
    
    # Get trading summary
    trading_summary = query_interface.get_trading_summary(hours_back=1)
    print(f"Trading summary: {len(trading_summary['trading_events'])} events")
    
    # Get performance metrics
    performance_metrics = query_interface.get_performance_metrics(hours_back=1)
    print(f"Performance metrics collected for {len(performance_metrics.get('latency_metrics', {}))} operations")
    
    # Detect anomalies
    anomalies = query_interface.detect_anomalies(hours_back=1)
    print(f"Anomalies detected: {anomalies['anomalies_detected']}")
    
    print("✅ Query interface demonstrated successfully")


def demonstrate_monitoring_and_alerts():
    """Demonstrate real-time monitoring and alerting."""
    print("\n📊 Demonstrating Monitoring and Alerts...")
    
    monitor = get_log_monitor()
    
    # Add custom alert rule
    monitor.alert_manager.add_rule(
        name="high_slippage",
        condition=lambda data: data.get("slippage", 0) > 0.1,  # 10 bps slippage
        level="warning",
        title="High Slippage Detected",
        message="Trade execution slippage exceeded 10 basis points"
    )
    
    # Subscribe to alerts
    def alert_handler(alert: Alert):
        print(f"🚨 ALERT: {alert.level.upper()} - {alert.title}")
        print(f"   Component: {alert.component}")
        print(f"   Message: {alert.message}")
        print(f"   Time: {alert.timestamp}")
    
    monitor.alert_manager.subscribe(alert_handler)
    
    # Get monitoring summary
    summary = monitor.get_monitoring_summary()
    print(f"System health: {summary['overall_health']}")
    print(f"Active alerts: {summary['active_alerts']}")
    print(f"Monitor status: {summary['system_status']}")
    
    # Register health check
    def database_health_check():
        # Simulate health check
        return "healthy", 25.0, {"connections": 8, "avg_query_time": 22.5}
    
    monitor.health_monitor.register_component("database", database_health_check)
    
    print("✅ Monitoring and alerts demonstrated successfully")


def generate_log_analysis_report():
    """Generate comprehensive log analysis report."""
    print("\n📈 Generating Log Analysis Report...")
    
    query_interface = LogQueryInterface()
    
    # Generate comprehensive report
    print("\n" + "="*60)
    print("COMPREHENSIVE LOG ANALYSIS REPORT")
    print("="*60)
    print(f"Generated at: {datetime.now().isoformat()}")
    
    # Error analysis
    error_summary = query_interface.get_error_summary(hours_back=24)
    print(f"\n📊 ERROR ANALYSIS (Last 24 hours)")
    print(f"Total Errors: {error_summary['total_errors']}")
    if error_summary['component_errors']:
        print("Errors by Component:")
        for error in error_summary['component_errors'][:5]:
            print(f"  - {error['logger']}: {error['count']}")
    
    # Trading analysis
    trading_summary = query_interface.get_trading_summary(hours_back=24)
    print(f"\n💰 TRADING ANALYSIS (Last 24 hours)")
    print(f"Total P&L: ${trading_summary['total_realized_pnl']:.2f}")
    print(f"Completed Trades: {trading_summary['completed_trades']}")
    if trading_summary['symbols_traded']:
        print("Most Active Symbols:")
        for symbol in trading_summary['symbols_traded'][:5]:
            print(f"  - {symbol['symbol']}: {symbol['activity_count']} events")
    
    # Performance analysis
    performance = query_interface.get_performance_metrics(hours_back=24)
    print(f"\n⚡ PERFORMANCE ANALYSIS (Last 24 hours)")
    if performance['latency_metrics']:
        print("Latency by Operation:")
        for op, stats in list(performance['latency_metrics'].items())[:5]:
            print(f"  - {op}: {stats['mean']:.1f}ms avg (max: {stats['max']:.1f}ms)")
    
    # Anomaly detection
    anomalies = query_interface.detect_anomalies(hours_back=24)
    print(f"\n🔍 ANOMALY DETECTION (Last 24 hours)")
    print(f"Anomalies Detected: {anomalies['anomalies_detected']}")
    if anomalies['anomalies']:
        print("Recent Anomalies:")
        for anomaly in anomalies['anomalies'][:3]:
            print(f"  - {anomaly['type']}: {anomaly['description']}")
    
    print("\n" + "="*60)
    print("✅ Log analysis report generated successfully")


async def main():
    """Main demonstration function."""
    print("🚀 GPT-Trader Comprehensive Logging Demo")
    print("="*60)
    
    try:
        # Setup logging system
        monitor = setup_comprehensive_logging()
        
        # Demonstrate different logging categories
        demonstrate_trading_logging()
        demonstrate_strategy_logging()
        demonstrate_ml_logging()
        demonstrate_risk_logging()
        demonstrate_system_logging()
        
        # Demonstrate querying and analysis
        demonstrate_query_interface()
        demonstrate_monitoring_and_alerts()
        
        # Generate comprehensive report
        generate_log_analysis_report()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📁 Logs persisted to database and files")
        print(f"🔍 Query interface ready for analysis")
        print(f"📊 Real-time monitoring active")
        
        # Keep monitoring running for a bit
        print(f"\n⏱️ Monitoring system for 10 seconds...")
        await asyncio.sleep(10)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise
    finally:
        # Cleanup
        monitor = get_log_monitor()
        monitor.stop()
        print("🛑 Monitoring stopped")


if __name__ == "__main__":
    asyncio.run(main())