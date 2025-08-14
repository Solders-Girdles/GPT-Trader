"""
Phase 2: Component Integration Demo

Demonstrates the integration of all Phase 2 architecture components:
- Dependency injection framework
- Unified concurrency framework
- Advanced error handling and recovery
- Service container lifecycle management
- Message queues and inter-component communication

This example shows how to build a complete trading system component
using the new architecture patterns.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

# Import Phase 2 architecture components
from bot.core.base import BaseComponent, BaseMonitor, ComponentConfig, HealthStatus
from bot.core.concurrency import (
    IMessageHandler,
    TaskPriority,
    create_message_queue,
    get_concurrency_manager,
    initialize_concurrency,
    schedule_recurring_task,
    submit_io_task,
)
from bot.core.container import (
    ServiceContainer,
    ServiceLifetime,
    component,
    configure_services,
    get_container,
    injectable,
)
from bot.core.database import get_database
from bot.core.error_handling import (
    CircuitBreakerConfig,
    RetryConfig,
    RetryStrategy,
    error_handling_context,
    get_error_manager,
    handle_errors,
    report_error,
    with_circuit_breaker,
)
from bot.core.exceptions import (
    raise_data_error,
    raise_trading_error,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("phase2_integration.log")],
)

logger = logging.getLogger(__name__)


# Example 1: Service with Dependency Injection


@injectable
class MarketDataService(BaseComponent):
    """Market data service using dependency injection"""

    def __init__(self, config: ComponentConfig | None = None):
        if not config:
            config = ComponentConfig(
                component_id="market_data_service", component_type="data_service"
            )

        super().__init__(config)
        self.db_manager = get_database()
        self.concurrency_manager = get_concurrency_manager()
        self.error_manager = get_error_manager()

        # Market data state
        self.subscribed_symbols = set()
        self.current_prices: dict[str, Decimal] = {}

        # Message queue for price updates
        self.price_queue = create_message_queue("price_updates", maxsize=1000)

    def _initialize_component(self):
        """Initialize market data service"""
        self.logger.info("Initializing market data service...")

        # Set up price update handler
        self.price_queue.subscribe("market_data_service", self)

    def _start_component(self):
        """Start market data operations"""
        self.logger.info("Starting market data service...")

        # Schedule periodic price updates
        schedule_recurring_task(
            task_id="price_update_task",
            function=self._fetch_price_updates,
            interval=timedelta(seconds=5),
            priority=TaskPriority.HIGH,
            component_id=self.component_id,
        )

    def _stop_component(self):
        """Stop market data operations"""
        self.logger.info("Stopping market data service...")
        self.price_queue.unsubscribe("market_data_service")

    def _health_check(self) -> HealthStatus:
        """Check service health"""
        if len(self.subscribed_symbols) == 0:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    @with_circuit_breaker("market_data_fetch")
    @handle_errors(
        retry_config=RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF, max_attempts=3, base_delay=1.0
        )
    )
    def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to symbol price updates"""
        with error_handling_context(self.component_id, "subscribe_symbol"):
            # Simulate API call that might fail
            if symbol.startswith("INVALID"):
                raise_data_error(f"Invalid symbol: {symbol}", data_source="market_api")

            self.subscribed_symbols.add(symbol)
            self.current_prices[symbol] = Decimal("100.00")  # Default price

            self.logger.info(f"Subscribed to {symbol}")
            return True

    def _fetch_price_updates(self):
        """Fetch price updates for subscribed symbols"""
        try:
            for symbol in self.subscribed_symbols:
                # Simulate price movement
                current = self.current_prices.get(symbol, Decimal("100.00"))
                change = Decimal(
                    str((hash(symbol + str(time.time())) % 200 - 100) / 1000)
                )  # +/- 10 cents
                new_price = current + change

                self.current_prices[symbol] = new_price

                # Publish price update
                self.price_queue.publish(
                    {
                        "type": "price_update",
                        "symbol": symbol,
                        "price": str(new_price),
                        "timestamp": datetime.now().isoformat(),
                        "source": self.component_id,
                    }
                )

            self.record_operation(success=True)

        except Exception as e:
            self.logger.error(f"Error fetching price updates: {str(e)}")
            self.record_operation(success=False, error_message=str(e))

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle incoming messages"""
        # This service publishes but doesn't need to handle messages
        return None

    def get_current_price(self, symbol: str) -> Decimal | None:
        """Get current price for symbol"""
        return self.current_prices.get(symbol)


# Example 2: Trading Strategy with Integration


@component(lifetime=ServiceLifetime.SINGLETON)
class IntegratedTradingStrategy(BaseComponent):
    """Trading strategy demonstrating full integration"""

    def __init__(
        self, market_data_service: MarketDataService, config: ComponentConfig | None = None
    ):
        if not config:
            config = ComponentConfig(
                component_id="integrated_trading_strategy", component_type="trading_strategy"
            )

        super().__init__(config)

        # Injected dependencies
        self.market_data_service = market_data_service
        self.db_manager = get_database()
        self.concurrency_manager = get_concurrency_manager()

        # Strategy state
        self.positions: dict[str, dict[str, Any]] = {}
        self.active_orders: dict[str, dict[str, Any]] = {}

        # Message handling
        self.signal_queue = create_message_queue("trading_signals")
        self.signal_queue.subscribe("trading_strategy", self)

    def _initialize_component(self):
        """Initialize trading strategy"""
        self.logger.info("Initializing trading strategy...")

        # Subscribe to symbols we want to trade
        symbols = ["AAPL", "GOOGL", "MSFT"]
        for symbol in symbols:
            self.market_data_service.subscribe_symbol(symbol)

    def _start_component(self):
        """Start trading operations"""
        self.logger.info("Starting trading strategy...")

        # Schedule strategy execution
        schedule_recurring_task(
            task_id="strategy_execution",
            function=self._execute_strategy,
            interval=timedelta(seconds=10),
            priority=TaskPriority.HIGH,
            component_id=self.component_id,
        )

        # Schedule position monitoring
        schedule_recurring_task(
            task_id="position_monitoring",
            function=self._monitor_positions,
            interval=timedelta(seconds=30),
            priority=TaskPriority.NORMAL,
            component_id=self.component_id,
        )

    def _stop_component(self):
        """Stop trading operations"""
        self.logger.info("Stopping trading strategy...")
        self.signal_queue.unsubscribe("trading_strategy")

    def _health_check(self) -> HealthStatus:
        """Check strategy health"""
        if not self.market_data_service.subscribed_symbols:
            return HealthStatus.UNHEALTHY
        return HealthStatus.HEALTHY

    @handle_errors(retry_config=RetryConfig(strategy=RetryStrategy.NO_RETRY))
    def _execute_strategy(self):
        """Execute trading strategy logic"""
        try:
            for symbol in self.market_data_service.subscribed_symbols:
                current_price = self.market_data_service.get_current_price(symbol)

                if current_price:
                    # Simple momentum strategy
                    signal = self._generate_signal(symbol, current_price)

                    if signal and signal["action"] != "hold":
                        # Submit signal asynchronously
                        submit_io_task(
                            self._process_trading_signal,
                            signal,
                            task_id=f"process_signal_{symbol}",
                            component_id=self.component_id,
                        )

            self.record_operation(success=True)

        except Exception as e:
            self.logger.error(f"Strategy execution error: {str(e)}")
            self.record_operation(success=False, error_message=str(e))
            report_error(e, component=self.component_id)

    def _generate_signal(self, symbol: str, current_price: Decimal) -> dict[str, Any] | None:
        """Generate trading signal"""
        # Simple price-based signal (for demonstration)
        if current_price > Decimal("105.00"):
            return {
                "symbol": symbol,
                "action": "sell",
                "price": current_price,
                "quantity": 100,
                "timestamp": datetime.now(),
            }
        elif current_price < Decimal("95.00"):
            return {
                "symbol": symbol,
                "action": "buy",
                "price": current_price,
                "quantity": 100,
                "timestamp": datetime.now(),
            }
        else:
            return {
                "symbol": symbol,
                "action": "hold",
                "price": current_price,
                "timestamp": datetime.now(),
            }

    @with_circuit_breaker("order_submission", CircuitBreakerConfig(failure_threshold=3))
    def _process_trading_signal(self, signal: dict[str, Any]):
        """Process trading signal with error handling"""
        try:
            symbol = signal["symbol"]
            action = signal["action"]

            if action == "buy":
                self._submit_buy_order(symbol, signal["quantity"], signal["price"])
            elif action == "sell":
                self._submit_sell_order(symbol, signal["quantity"], signal["price"])

            self.logger.info(f"Processed {action} signal for {symbol}")

        except Exception as e:
            self.logger.error(f"Error processing signal: {str(e)}")
            raise

    def _submit_buy_order(self, symbol: str, quantity: int, price: Decimal):
        """Submit buy order"""
        # Simulate order submission that might fail
        if symbol == "INVALID":
            raise_trading_error(f"Cannot buy invalid symbol: {symbol}")

        order_id = f"order_{symbol}_{int(time.time())}"
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": "buy",
            "quantity": quantity,
            "price": price,
            "status": "submitted",
            "timestamp": datetime.now(),
        }

        self.active_orders[order_id] = order

        # Store in database
        self.db_manager.insert_record(
            "orders",
            {
                "order_id": order_id,
                "strategy_id": self.component_id,
                "component_id": self.component_id,
                "symbol": symbol,
                "side": "buy",
                "order_type": "limit",
                "quantity": str(quantity),
                "limit_price": str(price),
                "status": "submitted",
                "created_at": datetime.now().isoformat(),
            },
        )

    def _submit_sell_order(self, symbol: str, quantity: int, price: Decimal):
        """Submit sell order"""
        # Similar to buy order logic
        order_id = f"order_{symbol}_{int(time.time())}"
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "side": "sell",
            "quantity": quantity,
            "price": price,
            "status": "submitted",
            "timestamp": datetime.now(),
        }

        self.active_orders[order_id] = order

        # Store in database
        self.db_manager.insert_record(
            "orders",
            {
                "order_id": order_id,
                "strategy_id": self.component_id,
                "component_id": self.component_id,
                "symbol": symbol,
                "side": "sell",
                "order_type": "limit",
                "quantity": str(quantity),
                "limit_price": str(price),
                "status": "submitted",
                "created_at": datetime.now().isoformat(),
            },
        )

    def _monitor_positions(self):
        """Monitor existing positions"""
        try:
            # Get positions from database
            positions = self.db_manager.fetch_all(
                """
                SELECT * FROM positions
                WHERE closed_at IS NULL AND component_id = ?
            """,
                (self.component_id,),
            )

            for position in positions:
                # Monitor position P&L and risk
                self._check_position_risk(position)

            self.logger.debug(f"Monitored {len(positions)} positions")

        except Exception as e:
            self.logger.error(f"Position monitoring error: {str(e)}")
            report_error(e, component=self.component_id)

    def _check_position_risk(self, position: dict[str, Any]):
        """Check individual position risk"""
        # Risk checking logic would go here
        pass

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle incoming messages"""
        if message.get("type") == "price_update":
            # React to price updates
            symbol = message.get("symbol")
            price = Decimal(message.get("price", "0"))

            self.logger.debug(f"Received price update: {symbol} @ ${price}")

        return None


# Example 3: Performance Monitor with Full Integration


class IntegratedPerformanceMonitor(BaseMonitor, IMessageHandler):
    """Performance monitor demonstrating monitoring patterns"""

    def __init__(self, config: ComponentConfig | None = None):
        if not config:
            config = ComponentConfig(
                component_id="integrated_performance_monitor", component_type="performance_monitor"
            )

        super().__init__(config)

        # Dependencies
        self.concurrency_manager = get_concurrency_manager()
        self.error_manager = get_error_manager()

        # Performance tracking
        self.performance_metrics: dict[str, Any] = {}

        # Subscribe to all message queues for monitoring
        self.monitored_queues = []

    def _initialize_component(self):
        """Initialize performance monitor"""
        self.logger.info("Initializing performance monitor...")

        # Subscribe to system message queues
        price_queue = create_message_queue("price_updates")
        signal_queue = create_message_queue("trading_signals")

        price_queue.subscribe("performance_monitor_price", self)
        signal_queue.subscribe("performance_monitor_signals", self)

        self.monitored_queues = [price_queue, signal_queue]

    def _start_component(self):
        """Start performance monitoring"""
        self.logger.info("Starting performance monitoring...")

        # Schedule performance collection
        schedule_recurring_task(
            task_id="collect_performance_metrics",
            function=self._collect_performance_metrics,
            interval=timedelta(seconds=15),
            priority=TaskPriority.LOW,
            component_id=self.component_id,
        )

        # Schedule system health checks
        schedule_recurring_task(
            task_id="system_health_check",
            function=self._check_system_health,
            interval=timedelta(minutes=1),
            priority=TaskPriority.NORMAL,
            component_id=self.component_id,
        )

    def _stop_component(self):
        """Stop performance monitoring"""
        for queue in self.monitored_queues:
            queue.unsubscribe("performance_monitor_price")
            queue.unsubscribe("performance_monitor_signals")

    def _health_check(self) -> HealthStatus:
        """Check monitor health"""
        return HealthStatus.HEALTHY

    def _collect_performance_metrics(self):
        """Collect system performance metrics"""
        try:
            # Get concurrency statistics
            concurrency_stats = self.concurrency_manager.get_system_stats()

            # Get error statistics
            error_stats = self.error_manager.get_error_statistics()

            # Get database statistics
            db_stats = self.db_manager.get_database_stats()

            # Combine all metrics
            self.performance_metrics = {
                "timestamp": datetime.now().isoformat(),
                "concurrency": concurrency_stats,
                "errors": error_stats,
                "database": db_stats,
                "component_count": len(concurrency_stats.get("active_components", [])),
                "total_errors": error_stats.get("total_errors", 0),
                "error_rate": error_stats.get("error_rate_per_hour", 0.0),
            }

            # Store metrics in database
            submit_io_task(
                self._store_performance_metrics,
                self.performance_metrics.copy(),
                component_id=self.component_id,
            )

            self.logger.debug("Performance metrics collected")

        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {str(e)}")
            report_error(e, component=self.component_id)

    def _store_performance_metrics(self, metrics: dict[str, Any]):
        """Store performance metrics in database"""
        try:
            self.db_manager.insert_record(
                "performance_snapshots",
                {
                    "snapshot_id": f"perf_{int(time.time())}",
                    "component_id": self.component_id,
                    "snapshot_time": metrics["timestamp"],
                    "metrics_data": str(metrics),  # JSON string
                    "component_count": metrics["component_count"],
                    "error_count": metrics["total_errors"],
                    "error_rate": metrics["error_rate"],
                },
            )

        except Exception as e:
            self.logger.error(f"Error storing performance metrics: {str(e)}")

    def _check_system_health(self):
        """Check overall system health"""
        try:
            # Check error rates
            error_stats = self.error_manager.get_error_statistics()
            error_rate = error_stats.get("error_rate_per_hour", 0)

            if error_rate > 100:  # More than 100 errors per hour
                self.logger.warning(f"High error rate detected: {error_rate:.1f} errors/hour")

            # Check thread pool health
            concurrency_stats = self.concurrency_manager.get_system_stats()
            thread_pools = concurrency_stats.get("thread_pools", {})

            for pool_name, pool_stats in thread_pools.items():
                success_rate = pool_stats.get("success_rate", 100)
                if success_rate < 90:
                    self.logger.warning(
                        f"Thread pool {pool_name} has low success rate: {success_rate:.1f}%"
                    )

            self.logger.debug("System health check completed")

        except Exception as e:
            self.logger.error(f"System health check error: {str(e)}")

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle messages for monitoring"""
        message_type = message.get("type", "unknown")
        self.logger.debug(f"Monitoring message: {message_type}")

        # Update message statistics
        if not hasattr(self, "message_stats"):
            self.message_stats = {}

        self.message_stats[message_type] = self.message_stats.get(message_type, 0) + 1

        return None

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()


def configure_demo_services(container: ServiceContainer):
    """Configure services for the demo"""

    # Register market data service (already marked with @injectable)
    # This is done automatically

    # Register trading strategy with dependency injection
    container.register_singleton(
        IntegratedTradingStrategy,
        # Dependencies will be resolved automatically
    )

    # Register performance monitor
    container.register_singleton(IntegratedPerformanceMonitor)

    logger.info("Demo services configured")


async def demonstrate_phase2_integration():
    """Demonstrate Phase 2 integration features"""

    logger.info("🚀 Starting Phase 2 Integration Demo")

    try:
        # Step 1: Initialize core systems
        logger.info("📋 Step 1: Initializing core systems...")

        # Initialize concurrency system
        concurrency_manager = initialize_concurrency()

        # Get service container
        container = get_container()

        # Configure demo services
        configure_services(configure_demo_services)

        # Step 2: Start all services
        logger.info("🔧 Step 2: Starting services...")

        # Start all registered services
        container.start_all_services()

        # Get service instances
        market_data = container.resolve(MarketDataService)
        trading_strategy = container.resolve(IntegratedTradingStrategy)
        performance_monitor = container.resolve(IntegratedPerformanceMonitor)

        logger.info(f"   ✅ Started {len(container.services)} services")

        # Step 3: Demonstrate service operations
        logger.info("⚡ Step 3: Demonstrating service operations...")

        # Let services run for a while
        await asyncio.sleep(30)

        # Step 4: Show integration statistics
        logger.info("📊 Step 4: Integration statistics...")

        # Get service health
        service_health = container.get_service_health()
        logger.info(f"   📈 Service Health: {service_health['container_status']}")
        logger.info(f"   🏃 Running Services: {service_health['running_services']}")
        logger.info(f"   ❌ Failed Services: {service_health['failed_services']}")

        # Get concurrency statistics
        concurrency_stats = concurrency_manager.get_system_stats()
        logger.info(
            f"   🧵 Active Threads: {sum(pool['active_tasks'] for pool in concurrency_stats['thread_pools'].values())}"
        )
        logger.info(f"   📨 Message Queues: {len(concurrency_stats['message_queues'])}")
        logger.info(f"   📅 Scheduled Tasks: {concurrency_stats['total_scheduled_tasks']}")

        # Get error statistics
        error_stats = get_error_manager().get_error_statistics()
        logger.info(f"   🚨 Total Errors: {error_stats['total_errors']}")
        logger.info(f"   📈 Error Rate: {error_stats['error_rate_per_hour']:.1f}/hour")
        logger.info(f"   🔧 Recovery Rate: {error_stats['recovery_success_rate']:.1f}%")

        # Get current performance metrics
        current_metrics = performance_monitor.get_current_metrics()
        if current_metrics:
            logger.info("   💡 Performance Monitoring: Active")
            logger.info(f"   🎯 Components Monitored: {current_metrics.get('component_count', 0)}")

        # Step 5: Demonstrate error handling and recovery
        logger.info("🛡️ Step 5: Demonstrating error handling...")

        try:
            # Try to subscribe to an invalid symbol (will trigger error handling)
            market_data.subscribe_symbol("INVALID_SYMBOL")
        except Exception as e:
            logger.info(f"   ✅ Error handling demonstrated: {type(e).__name__}")

        # Step 6: Show circuit breaker status
        logger.info("🔌 Step 6: Circuit breaker status...")

        error_stats = get_error_manager().get_error_statistics()
        circuit_breakers = error_stats.get("circuit_breakers", {})

        for cb_name, cb_status in circuit_breakers.items():
            logger.info(
                f"   🔌 {cb_name}: {cb_status['state']} (failures: {cb_status['failure_count']})"
            )

        logger.info("✅ Phase 2 Integration Demo completed successfully!")

        return True

    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        return False

    finally:
        # Step 7: Cleanup
        logger.info("🧹 Step 7: Cleaning up...")

        try:
            # Stop all services
            container.stop_all_services()

            # Shutdown concurrency manager
            concurrency_manager.shutdown()

            logger.info("   ✅ Cleanup completed")

        except Exception as e:
            logger.error(f"   ❌ Cleanup error: {str(e)}")


def show_integration_benefits():
    """Show the benefits of Phase 2 integration"""

    benefits = {
        "🏗️ Dependency Injection": [
            "Automatic component wiring and lifecycle management",
            "Loose coupling through interface-based dependencies",
            "Service lifetime management (singleton, transient, scoped)",
            "Circular dependency detection and prevention",
        ],
        "🧵 Unified Concurrency": [
            "Centralized thread pool management for all operations",
            "Automatic task scheduling and background processing",
            "Inter-component message queues and communication",
            "Graceful shutdown coordination across all threads",
        ],
        "🛡️ Advanced Error Handling": [
            "Circuit breaker protection for failing services",
            "Intelligent retry mechanisms with backoff strategies",
            "Automated error recovery and trend analysis",
            "Comprehensive error logging and alerting",
        ],
        "📊 System Integration": [
            "Unified health monitoring across all components",
            "Performance metrics collection and analysis",
            "Service discovery and registration automation",
            "Message-based component communication",
        ],
    }

    logger.info("🌟 Phase 2 Integration Benefits:")
    for category, items in benefits.items():
        logger.info(f"\n{category}:")
        for item in items:
            logger.info(f"   • {item}")


if __name__ == "__main__":
    print(
        """
    🚀 GPT-Trader Phase 2: Component Integration Demo
    =================================================

    This demo will showcase:
    1. Dependency injection and service container
    2. Unified concurrency and thread management
    3. Advanced error handling and recovery
    4. Service integration and communication
    5. Performance monitoring and health checks

    """
    )

    try:
        # Run the demo
        success = asyncio.run(demonstrate_phase2_integration())

        if success:
            print("\n✅ Demo completed successfully!")
            show_integration_benefits()

            print(
                """
    📋 Next Steps:
    1. Review integration logs: phase2_integration.log
    2. Examine service container health and statistics
    3. Test error scenarios and recovery mechanisms
    4. Monitor thread pool utilization and performance
    5. Begin migrating existing components to new patterns

    🎯 Phase 2 integration is ready for production deployment!
            """
            )
        else:
            print("\n❌ Demo failed - check phase2_integration.log for details")

    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {str(e)}")
        logger.exception("Demo crashed with exception")
