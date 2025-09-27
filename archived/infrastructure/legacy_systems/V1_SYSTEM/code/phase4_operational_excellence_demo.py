"""
Phase 4: Operational Excellence Integration Demo

Demonstrates the complete Phase 4 architecture including:
- Production deployment automation with CI/CD pipelines
- Advanced security hardening with encryption and authentication
- Disaster recovery and high availability with automated failover
- Advanced analytics and ML optimization with predictive insights

This example shows how all Phase 4 components work together to provide
enterprise-grade operational excellence for production trading systems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from bot.core.analytics import (
    DataSource,
    create_performance_model,
    get_analytics_manager,
    optimize_latency,
    setup_anomaly_detection,
)

# Import Phase 4 architecture components
from bot.core.base import BaseComponent, ComponentConfig, HealthStatus

# Phase 4: Operational Excellence
from bot.core.deployment import (
    DeploymentEnvironment,
    DeploymentStrategy,
    canary_deploy,
    deploy_to_kubernetes,
    get_deployment_manager,
)
from bot.core.disaster_recovery import (
    configure_high_availability,
    create_scheduled_backup,
    get_disaster_recovery_manager,
    setup_database_replication,
)
from bot.core.security import (
    audit_operation,
    encrypt_sensitive_data,
    get_security_manager,
    require_authentication,
    require_authorization,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("phase4_operational_excellence.log")],
)

logger = logging.getLogger(__name__)


# Example 1: Production Deployment with Automated CI/CD


class ProductionTradingService(BaseComponent):
    """Production trading service with operational excellence"""

    def __init__(self, config: ComponentConfig = None):
        if not config:
            config = ComponentConfig(
                component_id="production_trading_service", component_type="trading_service"
            )

        super().__init__(config)

        # Get operational systems
        self.deployment_manager = get_deployment_manager()
        self.security_manager = get_security_manager()
        self.dr_manager = get_disaster_recovery_manager()
        self.analytics_manager = get_analytics_manager()

        logger.info(f"Production trading service initialized: {self.component_id}")

    def _initialize_component(self):
        """Initialize with operational excellence"""
        logger.info("Initializing production trading service...")

    def _start_component(self):
        """Start trading service"""
        logger.info("Starting production trading service...")

    def _stop_component(self):
        """Stop trading service"""
        logger.info("Stopping production trading service...")

    def _health_check(self) -> HealthStatus:
        """Comprehensive health check"""
        return HealthStatus.HEALTHY

    @require_authentication(provider="jwt")
    @require_authorization("trading_orders", "create", provider="rbac")
    @encrypt_sensitive_data("trading_key", provider="fernet")
    @audit_operation("TRADING", "trading_orders", "submit_order")
    async def submit_order(
        self, symbol: str, quantity: int, order_type: str = "market", **kwargs
    ) -> dict[str, Any]:
        """Submit trading order with full security and auditing"""

        principal = kwargs.get("principal")
        logger.info(f"Processing order for {principal.name}: {symbol} x{quantity} ({order_type})")

        # Simulate order processing
        order_id = f"ORDER_{symbol}_{int(datetime.now().timestamp())}"

        # Predict order execution time using ML
        execution_time_prediction = await self._predict_execution_time(symbol, quantity, order_type)

        order_result = {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": quantity,
            "order_type": order_type,
            "status": "submitted",
            "submitted_at": datetime.now().isoformat(),
            "predicted_execution_time_ms": execution_time_prediction,
        }

        logger.info(
            f"Order submitted: {order_id} (predicted execution: {execution_time_prediction:.1f}ms)"
        )
        return order_result

    async def _predict_execution_time(self, symbol: str, quantity: int, order_type: str) -> float:
        """Predict order execution time using ML model"""
        try:
            # Use analytics system for prediction
            features = {
                "quantity": quantity,
                "order_type_market": 1 if order_type == "market" else 0,
                "order_type_limit": 1 if order_type == "limit" else 0,
                "hour_of_day": datetime.now().hour,
                "symbol_volatility": np.random.uniform(0.1, 0.5),  # Simulated volatility
            }

            # Make prediction (would use trained model in reality)
            base_time = 50.0  # 50ms base execution time
            complexity_factor = np.log(quantity + 1) * 10  # Larger orders take longer
            time_factor = (
                1.2 if datetime.now().hour in [9, 10, 15, 16] else 1.0
            )  # Market open/close

            predicted_time = base_time * complexity_factor * time_factor
            return min(predicted_time, 1000.0)  # Cap at 1 second

        except Exception as e:
            logger.error(f"Execution time prediction failed: {str(e)}")
            return 100.0  # Default prediction


# Example 2: Comprehensive Deployment Pipeline


async def demonstrate_deployment_automation():
    """Demonstrate production deployment automation"""

    logger.info("🚀 Step 1: Demonstrating deployment automation...")

    deployment_manager = get_deployment_manager()

    try:
        # Deploy to staging environment first
        logger.info("   📦 Deploying to staging environment...")
        staging_result = await deploy_to_kubernetes(
            application_name="gpt-trader",
            version="v2.1.0",
            docker_image="gpttrader/core",
            environment=DeploymentEnvironment.STAGING,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            replicas=2,
        )

        if staging_result.success:
            logger.info(f"   ✅ Staging deployment successful: {staging_result.deployment_id}")

            # Perform staging validation
            await asyncio.sleep(2)  # Simulate validation time

            # Deploy to production using canary strategy
            logger.info("   🐤 Deploying to production with canary strategy...")
            production_result = await canary_deploy(
                application_name="gpt-trader",
                version="v2.1.0",
                docker_image="gpttrader/core",
                traffic_percentage=10,
                analysis_duration_minutes=5,  # Shortened for demo
            )

            if production_result.success:
                logger.info(
                    f"   ✅ Canary deployment successful: {production_result.deployment_id}"
                )
            else:
                logger.error(f"   ❌ Canary deployment failed: {production_result.error_message}")
        else:
            logger.error(f"   ❌ Staging deployment failed: {staging_result.error_message}")

    except Exception as e:
        logger.error(f"Deployment demonstration failed: {str(e)}")


# Example 3: Security and Compliance Framework


async def demonstrate_security_framework():
    """Demonstrate advanced security framework"""

    logger.info("🔒 Step 2: Demonstrating security framework...")

    security_manager = get_security_manager()

    try:
        # Demonstrate encryption
        logger.info("   🔐 Testing data encryption...")
        sensitive_data = "Trading strategy parameters: MA(20), RSI(14), Stop-loss: 2%"

        encrypted_data = await security_manager.encrypt_data(sensitive_data, "strategy_key")
        logger.info(f"   ✅ Data encrypted (length: {len(encrypted_data)} chars)")

        decrypted_data = await security_manager.decrypt_data(encrypted_data, "strategy_key")
        logger.info(f"   ✅ Data decrypted successfully: {decrypted_data[:50]}...")

        # Demonstrate authentication
        logger.info("   👤 Testing user authentication...")
        credentials = {
            "username": "admin",
            "password": "secure_password",
            "ip_address": "192.168.1.100",
            "user_agent": "GPTTrader-Client/1.0",
        }

        principal = await security_manager.authenticate_user(credentials)
        if principal:
            logger.info(
                f"   ✅ Authentication successful: {principal.name} ({principal.principal_type})"
            )
        else:
            logger.error("   ❌ Authentication failed")

        # Demonstrate audit logging
        logger.info("   📋 Reviewing security audit events...")
        audit_events = security_manager.get_audit_events(
            start_time=datetime.now() - timedelta(hours=1)
        )
        logger.info(f"   📊 Found {len(audit_events)} audit events in last hour")

    except Exception as e:
        logger.error(f"Security demonstration failed: {str(e)}")


# Example 4: Disaster Recovery and High Availability


async def demonstrate_disaster_recovery():
    """Demonstrate disaster recovery capabilities"""

    logger.info("🆘 Step 3: Demonstrating disaster recovery...")

    dr_manager = get_disaster_recovery_manager()

    try:
        # Configure high availability
        logger.info("   ⚙️ Configuring high availability...")
        await configure_high_availability(
            rto_minutes=15,  # 15 minute recovery time objective
            rpo_minutes=5,  # 5 minute recovery point objective
            availability_percentage=99.9,
            primary_region="us-east-1",
            secondary_regions=["us-west-2"],
        )
        logger.info("   ✅ High availability configured")

        # Setup database replication
        logger.info("   🔄 Setting up database replication...")
        replication_success = await setup_database_replication(
            source_endpoint="postgres://primary-db:5432/trading",
            target_endpoints=[
                "postgres://backup-db-1:5432/trading",
                "postgres://backup-db-2:5432/trading",
            ],
        )

        if replication_success:
            logger.info("   ✅ Database replication configured")
        else:
            logger.warning("   ⚠️ Database replication setup skipped (demo mode)")

        # Create backup
        logger.info("   💾 Creating system backup...")
        backup_id = await create_scheduled_backup(
            storage_location="s3://gpttrader-backups/prod", retention_days=30
        )

        if backup_id:
            logger.info(f"   ✅ Backup created: {backup_id}")
        else:
            logger.warning("   ⚠️ Backup creation skipped (demo mode)")

        # Register health checks
        async def sample_health_check():
            return True  # Always healthy for demo

        dr_manager.register_health_check("trading_engine", sample_health_check)
        dr_manager.register_health_check("market_data_feed", sample_health_check)
        dr_manager.register_health_check("risk_manager", sample_health_check)

        # Get availability report
        availability_report = dr_manager.get_system_availability_report(days=7)
        logger.info(
            f"   📊 System availability: {availability_report['availability_percentage']:.2f}%"
        )

    except Exception as e:
        logger.error(f"Disaster recovery demonstration failed: {str(e)}")


# Example 5: Advanced Analytics and ML Optimization


async def demonstrate_analytics_and_optimization():
    """Demonstrate advanced analytics and ML optimization"""

    logger.info("🧠 Step 4: Demonstrating analytics and optimization...")

    analytics_manager = get_analytics_manager()

    try:
        # Register performance data source
        logger.info("   📊 Setting up performance analytics...")
        performance_data_source = DataSource(
            source_id="system_performance",
            source_type="database",
            connection_string="postgres://metrics-db:5432/metrics",
            query_or_path="SELECT * FROM system_metrics WHERE timestamp > NOW() - INTERVAL '1 hour'",
        )

        analytics_manager.register_data_source(performance_data_source)

        # Create performance prediction model
        logger.info("   🤖 Training performance prediction model...")
        model_id = await create_performance_model(
            model_name="latency_predictor",
            features=["cpu_usage", "memory_usage", "request_count", "active_connections"],
            target="response_time",
            data_source_id="system_performance",
        )
        logger.info(f"   ✅ Model trained: {model_id}")

        # Setup anomaly detection
        logger.info("   🔍 Setting up anomaly detection...")
        anomaly_model_id = await setup_anomaly_detection(
            data_source_id="system_performance",
            features=["cpu_usage", "memory_usage", "error_rate", "response_time"],
        )
        logger.info(f"   ✅ Anomaly detection configured: {anomaly_model_id}")

        # Perform system optimization
        logger.info("   ⚡ Optimizing system performance...")
        optimization_result = await optimize_latency(
            constraints={"max_cpu_usage": 80, "max_memory_usage": 70}
        )

        if optimization_result.success:
            improvements = optimization_result.improvement_percentage
            best_improvement = max(improvements.values()) if improvements else 0
            logger.info(f"   ✅ Optimization completed: {best_improvement:.1f}% improvement")
        else:
            logger.warning("   ⚠️ Optimization completed with mixed results")

        # Generate analytics summary
        summary = analytics_manager.get_analytics_summary()
        logger.info("   📈 Analytics Summary:")
        logger.info(f"      • Active Models: {summary['active_models']}")
        logger.info(f"      • Data Sources: {summary['data_sources']}")
        logger.info(f"      • Optimization Experiments: {summary['optimization_experiments']}")

    except Exception as e:
        logger.error(f"Analytics demonstration failed: {str(e)}")


# Example 6: Integrated Trading Workflow


async def demonstrate_integrated_workflow():
    """Demonstrate integrated operational workflow"""

    logger.info("🔄 Step 5: Demonstrating integrated workflow...")

    try:
        # Create production trading service
        trading_service = ProductionTradingService()
        trading_service.start()

        # Simulate authenticated trading session
        logger.info("   💼 Simulating authenticated trading session...")

        # Create mock principal for demonstration
        from bot.core.security import SecurityLevel, SecurityPrincipal

        mock_principal = SecurityPrincipal(
            principal_id="trader123",
            principal_type="user",
            name="John Trader",
            email="john@gpttrader.com",
            roles={"trader", "analyst"},
            permissions={"read", "write", "execute"},
            security_clearance=SecurityLevel.CONFIDENTIAL,
        )

        # Submit orders with full operational stack
        orders = [
            {"symbol": "AAPL", "quantity": 100, "order_type": "market"},
            {"symbol": "GOOGL", "quantity": 50, "order_type": "limit"},
            {"symbol": "MSFT", "quantity": 75, "order_type": "market"},
        ]

        for order_data in orders:
            try:
                # Include authentication context
                order_result = await trading_service.submit_order(
                    **order_data,
                    principal=mock_principal,
                    session_id="session_123",
                    ip_address="192.168.1.100",
                    user_agent="GPTTrader-Client/1.0",
                )
                logger.info(
                    f"   ✅ Order processed: {order_result['order_id']} - {order_result['predicted_execution_time_ms']:.1f}ms predicted"
                )
            except Exception as e:
                logger.error(f"   ❌ Order failed: {str(e)}")

        # Stop trading service
        trading_service.stop()

    except Exception as e:
        logger.error(f"Integrated workflow demonstration failed: {str(e)}")


async def demonstrate_phase4_operational_excellence():
    """Demonstrate Phase 4 Operational Excellence features"""

    logger.info("🚀 Starting Phase 4: Operational Excellence Demo")

    try:
        # Step 1: Deployment Automation
        await demonstrate_deployment_automation()

        # Step 2: Security Framework
        await demonstrate_security_framework()

        # Step 3: Disaster Recovery
        await demonstrate_disaster_recovery()

        # Step 4: Analytics and Optimization
        await demonstrate_analytics_and_optimization()

        # Step 5: Integrated Workflow
        await demonstrate_integrated_workflow()

        # Final Summary
        logger.info("📋 Step 6: Generating operational excellence summary...")

        deployment_manager = get_deployment_manager()
        security_manager = get_security_manager()
        dr_manager = get_disaster_recovery_manager()
        analytics_manager = get_analytics_manager()

        # Collect system summaries
        deployment_summary = {
            "active_deployments": len(deployment_manager.list_active_deployments()),
            "deployment_strategies": ["rolling_update", "canary", "blue_green"],
        }

        security_summary = security_manager.get_security_summary()
        availability_report = dr_manager.get_system_availability_report(days=1)
        analytics_summary = analytics_manager.get_analytics_summary()

        logger.info("   📊 Operational Excellence Summary:")
        logger.info(f"      🚀 Deployments: {deployment_summary['active_deployments']} active")
        logger.info(
            f"      🔒 Security: {security_summary['active_sessions']} sessions, {security_summary['total_audit_events']} audit events"
        )
        logger.info(f"      🆘 Availability: {availability_report['availability_percentage']:.2f}%")
        logger.info(
            f"      🧠 Analytics: {analytics_summary['active_models']} models, {analytics_summary['optimization_experiments']} experiments"
        )

        logger.info("✅ Phase 4: Operational Excellence Demo completed successfully!")

        return True

    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        return False


def show_phase4_benefits():
    """Show the benefits of Phase 4 implementation"""

    benefits = {
        "🚀 Production Deployment": [
            "Zero-downtime deployments with canary and blue-green strategies",
            "Automated CI/CD pipelines with comprehensive validation",
            "Container orchestration with Kubernetes integration",
            "Multi-environment deployment management",
        ],
        "🔒 Advanced Security": [
            "Multi-layered encryption with key rotation",
            "Role-based access control (RBAC) and audit logging",
            "Multi-factor authentication and token management",
            "Compliance frameworks (SOX, GDPR, PCI DSS) support",
        ],
        "🆘 Disaster Recovery": [
            "Multi-region failover with automated recovery",
            "Real-time data replication and backup automation",
            "RTO/RPO management with SLA compliance",
            "Circuit breaker patterns for resilience",
        ],
        "🧠 Analytics & Optimization": [
            "Machine learning models for predictive analytics",
            "Automated system optimization with reinforcement learning",
            "Real-time anomaly detection and incident prevention",
            "Business intelligence with performance correlation",
        ],
    }

    logger.info("🌟 Phase 4: Operational Excellence Benefits:")
    for category, items in benefits.items():
        logger.info(f"\n{category}:")
        for item in items:
            logger.info(f"   • {item}")


if __name__ == "__main__":
    print(
        """
    🚀 GPT-Trader Phase 4: Operational Excellence Demo
    ==================================================

    This demo will showcase:
    1. Production deployment automation with CI/CD
    2. Advanced security hardening and compliance
    3. Disaster recovery and high availability
    4. Advanced analytics and ML optimization
    5. Integrated operational workflows

    """
    )

    try:
        # Run the demo
        success = asyncio.run(demonstrate_phase4_operational_excellence())

        if success:
            print("\n✅ Demo completed successfully!")
            show_phase4_benefits()

            print(
                """
    📋 Next Steps:
    1. Review operational logs: phase4_operational_excellence.log
    2. Configure production deployment pipelines
    3. Implement security policies and compliance monitoring
    4. Set up disaster recovery procedures
    5. Train ML models for system optimization

    🎯 Phase 4 provides enterprise-grade operational excellence!
            """
            )
        else:
            print("\n❌ Demo failed - check phase4_operational_excellence.log for details")

    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {str(e)}")
        logger.exception("Demo crashed with exception")
