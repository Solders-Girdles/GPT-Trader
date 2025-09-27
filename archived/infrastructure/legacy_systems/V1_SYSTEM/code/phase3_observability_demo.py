"""
Phase 3: Performance & Observability Integration Demo

Demonstrates the complete Phase 3 architecture including:
- Advanced caching layer with intelligent cache management
- Comprehensive metrics collection and analysis
- Performance optimization with automated bottleneck detection
- Advanced observability with distributed tracing and alerting

This example shows how all Phase 3 components work together to provide
enterprise-grade performance monitoring and optimization.
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

# Import Phase 3 architecture components
from bot.core.base import BaseComponent, ComponentConfig, HealthStatus

# Phase 3: Performance & Observability
from bot.core.caching import CacheConfig, cached, get_cache_manager
from bot.core.metrics import (
    MetricLabels,
    count_calls,
    get_metrics_collector,
    get_metrics_registry,
    metrics_context,
    track_execution_time,
)
from bot.core.observability import (
    AlertRule,
    AlertSeverity,
    create_alert,
    get_observability_engine,
    observability_context,
    start_trace,
    trace_operation,
)
from bot.core.performance import (
    create_profiler,
    get_performance_optimizer,
    performance_context,
    profile_performance,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("phase3_observability.log")],
)

logger = logging.getLogger(__name__)


# Example 1: Market Data Service with Full Observability


class ObservableMarketDataService(BaseComponent):
    """Market data service with comprehensive observability"""

    def __init__(self, config: ComponentConfig | None = None):
        if not config:
            config = ComponentConfig(
                component_id="observable_market_data_service", component_type="market_data_service"
            )

        super().__init__(config)

        # Get observability components
        self.cache_manager = get_cache_manager()
        self.metrics_registry = get_metrics_registry()
        self.performance_optimizer = get_performance_optimizer()
        self.observability_engine = get_observability_engine()

        # Create component-specific profiler
        self.profiler = create_profiler(f"profiler_{self.component_id}")

        # Set up caching
        cache_config = CacheConfig(
            cache_name=f"{self.component_id}_cache",
            max_size=1000,
            ttl_seconds=300,
            enable_compression=True,
        )
        self.price_cache = self.cache_manager.create_cache(cache_config)

        # Set up metrics
        self._setup_metrics()

        # Market data state
        self.subscribed_symbols = set()
        self.price_data: dict[str, dict] = {}

        logger.info(f"Observable market data service initialized: {self.component_id}")

    def _setup_metrics(self):
        """Set up component-specific metrics"""
        labels = MetricLabels().add("component", self.component_id)

        self.metrics = {
            "api_calls": self.metrics_registry.register_counter(
                "market_data_api_calls_total",
                "Total market data API calls",
                component_id=self.component_id,
                labels=labels,
            ),
            "cache_hits": self.metrics_registry.register_counter(
                "market_data_cache_hits_total",
                "Cache hits for market data",
                component_id=self.component_id,
                labels=labels,
            ),
            "price_updates": self.metrics_registry.register_counter(
                "market_data_price_updates_total",
                "Total price updates processed",
                component_id=self.component_id,
                labels=labels,
            ),
            "fetch_latency": self.metrics_registry.register_histogram(
                "market_data_fetch_latency_ms",
                "Market data fetch latency",
                component_id=self.component_id,
                labels=labels,
            ),
            "subscribed_symbols": self.metrics_registry.register_gauge(
                "market_data_subscribed_symbols",
                "Number of subscribed symbols",
                component_id=self.component_id,
                labels=labels,
            ),
        }

    def _initialize_component(self):
        """Initialize component with observability"""
        with observability_context("market_data_init", self.component_id):
            logger.info("Initializing observable market data service...")

            # Set up alert rules specific to this component
            self._setup_alert_rules()

    def _start_component(self):
        """Start component operations"""
        with observability_context("market_data_start", self.component_id):
            logger.info("Starting observable market data service...")

    def _stop_component(self):
        """Stop component operations"""
        with observability_context("market_data_stop", self.component_id):
            logger.info("Stopping observable market data service...")

    def _health_check(self) -> HealthStatus:
        """Health check with observability"""
        if len(self.subscribed_symbols) == 0:
            return HealthStatus.DEGRADED

        # Check cache performance
        cache_stats = self.price_cache.get_statistics()
        if cache_stats["hit_rate"] < 50:  # Less than 50% hit rate
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _setup_alert_rules(self):
        """Set up component-specific alert rules"""
        rules = [
            AlertRule(
                rule_id=f"{self.component_id}_high_latency",
                name="Market Data High Latency",
                description="Market data fetch latency is too high",
                metric_name="market_data_fetch_latency_ms",
                operator=">",
                threshold_value=1000.0,  # 1 second
                severity=AlertSeverity.WARNING,
                component_filter=self.component_id,
                notification_channels=["log"],
            ),
            AlertRule(
                rule_id=f"{self.component_id}_low_cache_hit",
                name="Market Data Low Cache Hit Rate",
                description="Cache hit rate is below acceptable threshold",
                metric_name="cache_hit_rate_percent",
                operator="<",
                threshold_value=70.0,
                severity=AlertSeverity.WARNING,
                component_filter=self.component_id,
                notification_channels=["log"],
            ),
        ]

        for rule in rules:
            self.observability_engine.add_alert_rule(rule)

    @trace_operation("subscribe_symbol")
    @profile_performance(sample_rate=0.5)
    @track_execution_time()
    @count_calls()
    def subscribe_symbol(self, symbol: str) -> bool:
        """Subscribe to symbol with full observability"""

        # Start distributed trace
        trace = start_trace(f"subscribe_symbol_{symbol}")
        trace.add_tag("symbol", symbol)
        trace.add_tag("component_id", self.component_id)

        try:
            with performance_context(f"subscribe_{symbol}", self.component_id):
                # Simulate API call with variable latency
                api_latency = random.uniform(50, 500)  # 50-500ms
                time.sleep(api_latency / 1000)

                self.metrics["api_calls"].increment()
                self.metrics["fetch_latency"].observe(api_latency)

                # Add to subscribed symbols
                self.subscribed_symbols.add(symbol)
                self.metrics["subscribed_symbols"].set(len(self.subscribed_symbols))

                trace.add_tag("success", True)
                trace.log(f"Successfully subscribed to {symbol}", level="info")

                logger.info(f"Subscribed to symbol: {symbol}")
                return True

        except Exception as e:
            trace.add_tag("success", False)
            trace.add_tag("error", str(e))
            trace.log(f"Subscription failed: {str(e)}", level="error")

            # Create alert for subscription failure
            create_alert(
                name="Symbol Subscription Failed",
                severity=AlertSeverity.ERROR,
                description=f"Failed to subscribe to symbol {symbol}: {str(e)}",
                component_id=self.component_id,
                trace_id=trace.trace_id,
                symbol=symbol,
                error=str(e),
            )

            raise

        finally:
            self.observability_engine.finish_trace(trace)

    @cached(cache_name="market_data_cache", ttl_seconds=60)
    @trace_operation("get_price")
    @track_execution_time()
    def get_price(self, symbol: str) -> Decimal | None:
        """Get price with caching and observability"""

        # Check if already cached (cache decorator handles this, but we want to track metrics)
        cached_price = self.price_cache.get(f"price_{symbol}")
        if cached_price:
            self.metrics["cache_hits"].increment()
            return cached_price

        # Simulate expensive price fetch
        with metrics_context("price_fetch", self.component_id):
            # Simulate variable fetch time
            fetch_time = random.uniform(100, 300)
            time.sleep(fetch_time / 1000)

            self.metrics["api_calls"].increment()
            self.metrics["fetch_latency"].observe(fetch_time)

            # Generate fake price data
            base_price = 100.0
            price_change = random.uniform(-5, 5)
            current_price = Decimal(str(base_price + price_change))

            # Store in cache
            self.price_cache.put(f"price_{symbol}", current_price, ttl_seconds=60)

            self.metrics["price_updates"].increment()

            return current_price

    def simulate_price_updates(self, duration_seconds: int = 30):
        """Simulate price updates for testing"""
        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            if self.subscribed_symbols:
                # Update random symbols
                symbols_to_update = random.sample(
                    list(self.subscribed_symbols), min(3, len(self.subscribed_symbols))
                )

                for symbol in symbols_to_update:
                    try:
                        price = self.get_price(symbol)
                        logger.debug(f"Updated price for {symbol}: {price}")

                        # Occasionally simulate errors
                        if random.random() < 0.05:  # 5% error rate
                            raise Exception(f"Simulated API error for {symbol}")

                    except Exception as e:
                        logger.error(f"Price update error for {symbol}: {str(e)}")

                        # This will be captured by observability system
                        create_alert(
                            name="Price Update Failed",
                            severity=AlertSeverity.WARNING,
                            description=f"Failed to update price for {symbol}",
                            component_id=self.component_id,
                            symbol=symbol,
                            error=str(e),
                        )

            time.sleep(1)  # Update every second

    def get_performance_summary(self) -> dict[str, Any]:
        """Get component performance summary"""
        cache_stats = self.price_cache.get_statistics()
        profiler_stats = self.profiler.get_performance_summary()

        return {
            "component_id": self.component_id,
            "subscribed_symbols": len(self.subscribed_symbols),
            "cache_performance": cache_stats,
            "profiler_performance": profiler_stats,
            "health_status": self.get_health_status().value,
        }


# Example 2: Trading Engine with Performance Optimization


class OptimizedTradingEngine(BaseComponent):
    """Trading engine with automated performance optimization"""

    def __init__(self, config: ComponentConfig | None = None):
        if not config:
            config = ComponentConfig(
                component_id="optimized_trading_engine", component_type="trading_engine"
            )

        super().__init__(config)

        # Performance optimization
        self.performance_optimizer = get_performance_optimizer()
        self.profiler = create_profiler(f"profiler_{self.component_id}", sample_rate=0.2)

        # Observability
        self.observability_engine = get_observability_engine()

        # Trading state
        self.orders: dict[str, dict] = {}
        self.positions: dict[str, dict] = {}

        logger.info(f"Optimized trading engine initialized: {self.component_id}")

    def _initialize_component(self):
        """Initialize with performance monitoring"""
        with observability_context("trading_engine_init", self.component_id):
            logger.info("Initializing optimized trading engine...")

    def _start_component(self):
        """Start trading operations"""
        logger.info("Starting optimized trading engine...")

    def _stop_component(self):
        """Stop trading operations"""
        logger.info("Stopping optimized trading engine...")

    def _health_check(self) -> HealthStatus:
        """Health check with performance considerations"""
        # Check recent performance issues
        current_issues = self.performance_optimizer.get_current_issues()
        critical_issues = [i for i in current_issues if i.severity.value == "critical"]

        if critical_issues:
            return HealthStatus.CRITICAL

        return HealthStatus.HEALTHY

    @profile_performance("order_submission", sample_rate=1.0)
    @trace_operation("submit_order")
    def submit_order(self, symbol: str, quantity: int, order_type: str = "market") -> str:
        """Submit trading order with performance profiling"""

        with performance_context("submit_order", self.component_id):
            order_id = f"order_{symbol}_{int(time.time())}"

            # Simulate order processing with variable complexity
            processing_time = random.uniform(10, 200)  # 10-200ms

            # Occasionally simulate slow orders (performance issue)
            if random.random() < 0.1:  # 10% of orders are slow
                processing_time = random.uniform(1000, 2000)  # 1-2 seconds

            time.sleep(processing_time / 1000)

            # Store order
            self.orders[order_id] = {
                "symbol": symbol,
                "quantity": quantity,
                "order_type": order_type,
                "submitted_at": datetime.now(),
                "processing_time_ms": processing_time,
            }

            logger.info(f"Order submitted: {order_id} ({processing_time:.1f}ms)")

            return order_id

    def simulate_trading_activity(self, duration_seconds: int = 60):
        """Simulate trading activity for performance testing"""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        start_time = time.time()

        order_count = 0

        while time.time() - start_time < duration_seconds:
            try:
                # Submit random orders
                symbol = random.choice(symbols)
                quantity = random.randint(1, 100)
                order_type = random.choice(["market", "limit"])

                order_id = self.submit_order(symbol, quantity, order_type)
                order_count += 1

                # Vary order submission rate
                delay = random.uniform(0.5, 2.0)
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Order submission error: {str(e)}")

        logger.info(f"Submitted {order_count} orders in {duration_seconds} seconds")


async def demonstrate_phase3_observability():
    """Demonstrate Phase 3 Performance & Observability features"""

    logger.info("🚀 Starting Phase 3: Performance & Observability Demo")

    try:
        # Step 1: Initialize all Phase 3 systems
        logger.info("📋 Step 1: Initializing Phase 3 systems...")

        cache_manager = get_cache_manager()
        metrics_collector = get_metrics_collector()
        performance_optimizer = get_performance_optimizer()
        observability_engine = get_observability_engine()

        logger.info("   ✅ All Phase 3 systems initialized")

        # Step 2: Create and start observable components
        logger.info("🔧 Step 2: Creating observable components...")

        market_data_service = ObservableMarketDataService()
        trading_engine = OptimizedTradingEngine()

        # Start components
        market_data_service.start()
        trading_engine.start()

        logger.info("   ✅ Components started with full observability")

        # Step 3: Demonstrate caching capabilities
        logger.info("💾 Step 3: Demonstrating intelligent caching...")

        # Subscribe to symbols (will populate cache)
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        for symbol in symbols:
            market_data_service.subscribe_symbol(symbol)

        # Get prices multiple times (demonstrate caching)
        for _ in range(5):
            for symbol in symbols[:3]:
                price = market_data_service.get_price(symbol)
                logger.debug(f"Price for {symbol}: {price}")

        # Show cache performance
        cache_stats = cache_manager.get_system_statistics()
        logger.info(
            f"   📊 Cache Performance: {cache_stats['global_statistics']['global_hit_rate']:.1f}% hit rate"
        )

        # Step 4: Demonstrate metrics collection
        logger.info("📈 Step 4: Demonstrating comprehensive metrics...")

        # Let components run to generate metrics
        await asyncio.sleep(5)

        # Get metrics summary
        metrics_summary = metrics_collector.get_metric_summary()
        logger.info(
            f"   📊 Metrics Summary: {metrics_summary['total_metrics']} metrics across {len(metrics_summary['components'])} components"
        )

        # Step 5: Demonstrate performance optimization
        logger.info("⚡ Step 5: Demonstrating performance optimization...")

        # Run trading activity to generate performance data
        logger.info("   🔄 Generating trading activity for performance analysis...")
        trading_task = asyncio.create_task(
            asyncio.to_thread(trading_engine.simulate_trading_activity, 30)
        )

        # Run price updates concurrently
        price_task = asyncio.create_task(
            asyncio.to_thread(market_data_service.simulate_price_updates, 30)
        )

        # Wait for activities to complete
        await asyncio.gather(trading_task, price_task)

        # Check for performance issues
        current_issues = performance_optimizer.get_current_issues()
        logger.info(f"   🔍 Performance Issues Detected: {len(current_issues)}")

        for issue in current_issues[:3]:  # Show first 3 issues
            logger.info(f"      • {issue.description} (Impact: {issue.impact_score:.1f}%)")

        # Step 6: Demonstrate observability and alerting
        logger.info("👁️ Step 6: Demonstrating advanced observability...")

        # Get system health summary
        health_summary = observability_engine.get_system_health_summary()
        logger.info(f"   💚 System Health: {health_summary['overall_health']}")
        logger.info(f"   🚨 Active Alerts: {health_summary['active_alerts']}")

        # Show trace statistics
        trace_stats = health_summary.get("trace_statistics", {})
        if trace_stats:
            logger.info(f"   🔍 Traces Processed: {trace_stats.get('total_traces_processed', 0)}")
            logger.info(f"   ⏱️ Avg Trace Duration: {trace_stats.get('avg_duration_ms', 0):.1f}ms")

        # Create a test alert
        test_alert = create_alert(
            name="Demo Alert",
            severity=AlertSeverity.INFO,
            description="This is a demonstration alert showing the alerting system",
            component_id="demo_component",
        )
        logger.info(f"   📢 Demo Alert Created: {test_alert.alert_id}")

        # Step 7: Show comprehensive statistics
        logger.info("📊 Step 7: System performance statistics...")

        # Cache statistics
        cache_stats = cache_manager.get_system_statistics()
        logger.info("   💾 Cache System:")
        logger.info(f"      • Total Caches: {cache_stats['global_statistics']['total_caches']}")
        logger.info(
            f"      • Global Hit Rate: {cache_stats['global_statistics']['global_hit_rate']:.1f}%"
        )
        logger.info(f"      • Total Requests: {cache_stats['global_statistics']['total_requests']}")

        # Performance statistics
        perf_issues = performance_optimizer.get_current_issues()
        perf_results = performance_optimizer.get_optimization_results()
        logger.info("   ⚡ Performance System:")
        logger.info(f"      • Issues Detected: {len(perf_issues)}")
        logger.info(f"      • Optimizations Applied: {len(perf_results)}")

        # Component performance
        market_data_perf = market_data_service.get_performance_summary()
        logger.info("   📡 Market Data Service:")
        logger.info(f"      • Subscribed Symbols: {market_data_perf['subscribed_symbols']}")
        logger.info(
            f"      • Cache Hit Rate: {market_data_perf['cache_performance']['hit_rate']:.1f}%"
        )

        # Step 8: Cleanup and final report
        logger.info("🧹 Step 8: Generating final observability report...")

        # Stop components
        market_data_service.stop()
        trading_engine.stop()

        # Resolve demo alert
        observability_engine.resolve_alert(test_alert.alert_id, "demo_system", "Demo completed")

        logger.info("✅ Phase 3: Performance & Observability Demo completed successfully!")

        return True

    except Exception as e:
        logger.error(f"❌ Demo failed: {str(e)}")
        return False


def show_phase3_benefits():
    """Show the benefits of Phase 3 implementation"""

    benefits = {
        "💾 Intelligent Caching": [
            "Multi-tier cache architecture (L1: memory, L2: distributed)",
            "Automatic cache warming and refresh-ahead strategies",
            "Compression and optimization for large objects",
            "Real-time hit rate monitoring and optimization",
        ],
        "📊 Comprehensive Metrics": [
            "Multi-dimensional metrics with tags and labels",
            "Custom metric types (counters, gauges, histograms, summaries)",
            "Real-time metrics collection with minimal overhead",
            "Business metrics tracking and SLA monitoring",
        ],
        "⚡ Performance Optimization": [
            "Real-time performance profiling and bottleneck detection",
            "Automated optimization recommendations and implementations",
            "Algorithm complexity analysis and memory leak detection",
            "Thread pool sizing and concurrency optimization",
        ],
        "👁️ Advanced Observability": [
            "Distributed tracing across all system components",
            "Intelligent alerting with machine learning-based anomaly detection",
            "Service mesh observability and health monitoring",
            "Incident management and escalation workflows",
        ],
    }

    logger.info("🌟 Phase 3: Performance & Observability Benefits:")
    for category, items in benefits.items():
        logger.info(f"\n{category}:")
        for item in items:
            logger.info(f"   • {item}")


if __name__ == "__main__":
    print(
        """
    🚀 GPT-Trader Phase 3: Performance & Observability Demo
    =======================================================

    This demo will showcase:
    1. Intelligent caching with multi-tier architecture
    2. Comprehensive metrics collection and analysis
    3. Automated performance optimization
    4. Advanced observability with distributed tracing
    5. Intelligent alerting and health monitoring

    """
    )

    try:
        # Run the demo
        success = asyncio.run(demonstrate_phase3_observability())

        if success:
            print("\n✅ Demo completed successfully!")
            show_phase3_benefits()

            print(
                """
    📋 Next Steps:
    1. Review observability logs: phase3_observability.log
    2. Examine cache performance and hit rates
    3. Analyze performance optimization results
    4. Review distributed traces and alert correlations
    5. Begin implementing Phase 3 patterns in production components

    🎯 Phase 3 implementation provides enterprise-grade observability!
            """
            )
        else:
            print("\n❌ Demo failed - check phase3_observability.log for details")

    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {str(e)}")
        logger.exception("Demo crashed with exception")
