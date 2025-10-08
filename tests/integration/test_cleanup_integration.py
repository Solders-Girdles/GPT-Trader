"""Integration tests demonstrating the cleaned-up codebase working together."""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.config.config_utilities import parse_mapping_env, parse_list_env, validate_required_env
from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder
from bot_v2.orchestration.deterministic_broker import DeterministicBroker
from bot_v2.errors.error_patterns import handle_brokerage_errors, ErrorContext
from bot_v2.utilities.logging_patterns import get_logger, log_operation, log_trade_event
from bot_v2.utilities.performance_monitoring import measure_performance_decorator, monitor_trading_operation
from bot_v2.monitoring.health_checks import setup_basic_health_checks, get_health_summary
from bot_v2.utilities.import_utils import optional_import, lazy_import
from bot_v2.utilities.async_utils import async_rate_limit, async_retry
from tests.fixtures.product_factory import ProductFactory


class TestConfigurationIntegration:
    """Test configuration management integration."""
    
    def test_end_to_end_configuration_loading(self) -> None:
        """Test loading and using configuration with all utilities."""
        # Load configuration using new utilities
        config = RiskConfig.from_env()
        
        # Verify configuration structure
        assert config is not None
        assert isinstance(config, RiskConfig)
        
        # Test configuration validation
        assert config.max_leverage > 0
        assert config.min_liquidation_buffer_pct >= 0


class TestBuilderPatternIntegration:
    """Test builder pattern integration with other components."""
    
    def test_perps_bot_builder_with_all_components(self) -> None:
        """Test building PerpsBot with all cleaned-up components."""
        # Create test broker using factory
        factory = ProductFactory()
        broker = DeterministicBroker(
            products=factory.create_test_products(["BTC-PERP", "ETH-PERP"]),
            equity=Decimal("100000")
        )
        
        # Create mock stores
        event_store = Mock()
        account_store = Mock()
        market_data = Mock()
        
        # Use builder pattern
        bot = (
            PerpsBotBuilder()
            .with_config(Mock())
            .with_event_store(event_store)
            .with_account_store(account_store)
            .with_brokerage(broker)
            .with_market_data(market_data)
            .build()
        )
        
        assert bot is not None
        assert bot.brokerage is broker
        assert bot.event_store is event_store


class TestErrorHandlingIntegration:
    """Test error handling integration across components."""
    
    def test_error_handling_with_logging(self) -> None:
        """Test error handling with structured logging."""
        logger = get_logger("test_integration", component="error_handling")
        
        @handle_brokerage_errors("test_operation")
        def risky_operation(should_fail: bool = False):
            if should_fail:
                raise ValueError("Test error")
            return "success"
            
        # Test successful operation
        result = risky_operation(False)
        assert result == "success"
        
        # Test failed operation with proper error handling
        with pytest.raises(ValueError):
            risky_operation(True)
            
    def test_error_context_with_performance_tracking(self) -> None:
        """Test error context with performance monitoring."""
        logger = get_logger("test_integration", component="performance")
        
        with ErrorContext("test_context", logger=logger):
            with log_operation("test_operation", logger):
                time.sleep(0.01)  # Simulate work
                # Operation would normally do work here


class TestLoggingIntegration:
    """Test logging integration across all components."""
    
    def test_structured_logging_with_performance(self) -> None:
        """Test structured logging with performance monitoring."""
        logger = get_logger("trading", component="integration_test")
        
        @measure_performance_decorator("test_trading_operation", tags={"component": "test"})
        def trading_operation():
            log_trade_event(
                "order_filled",
                "BTC-PERP",
                side="buy",
                quantity="0.1",
                price="50000"
            )
            return "order_filled"
            
        result = trading_operation()
        assert result == "order_filled"
        
    def test_logging_with_error_context(self) -> None:
        """Test logging within error context."""
        logger = get_logger("test_integration", component="logging")
        
        try:
            with ErrorContext("risky_operation", logger=logger):
                raise ValueError("Test error in context")
        except ValueError:
            pass  # Expected


class TestPerformanceMonitoringIntegration:
    """Test performance monitoring integration."""
    
    def test_performance_monitoring_with_trading(self) -> None:
        """Test performance monitoring with trading operations."""
        
        @monitor_trading_operation("place_order")
        def place_order(symbol: str, quantity: Decimal) -> str:
            # Simulate order placement
            time.sleep(0.01)
            return f"Order placed for {quantity} {symbol}"
            
        result = place_order("BTC-PERP", Decimal("0.1"))
        assert "Order placed" in result
        
    def test_performance_monitoring_with_error_handling(self) -> None:
        """Test performance monitoring with error handling."""
        
        @monitor_trading_operation("failing_operation")
        @handle_brokerage_errors("failing_operation")
        def failing_operation():
            raise ConnectionError("Network error")
            
        with pytest.raises(ConnectionError):
            failing_operation()


class TestHealthCheckIntegration:
    """Test health check integration."""
    
    def test_health_checks_with_all_components(self) -> None:
        """Test health checks with all system components."""
        # Create mock components
        mock_db = Mock()
        mock_brokerage = Mock()
        mock_api = Mock()
        
        # Set up health checks
        setup_basic_health_checks(
            database_connection=mock_db,
            brokerage=mock_brokerage,
            api_client=mock_api
        )
        
        # Get health summary
        health = get_health_summary()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "timestamp" in health


class TestImportUtilitiesIntegration:
    """Test import utilities integration."""
    
    def test_optional_imports_in_real_code(self) -> None:
        """Test optional imports in actual usage."""
        # Test with pandas (may or may not be available)
        pandas = optional_import("pandas")
        
        if pandas.is_available():
            # Use pandas if available
            assert pandas.get() is not None
        else:
            # Graceful fallback
            assert pandas.get() is None
            assert pandas.get("fallback") == "fallback"
            
    def test_lazy_imports_for_heavy_dependencies(self) -> None:
        """Test lazy imports for heavy dependencies."""
        # Lazy import a heavy module
        tensorflow = lazy_import("tensorflow")
        
        # Should not be loaded yet
        assert not tensorflow._loaded
        
        # Access should trigger import (or fail gracefully)
        try:
            _ = tensorflow.__version__
        except (ImportError, AttributeError):
            # Expected if tensorflow is not installed
            pass


class TestAsyncUtilitiesIntegration:
    """Test async utilities integration."""
    
    @pytest.mark.asyncio
    async def test_async_rate_limiting_with_monitoring(self) -> None:
        """Test async rate limiting with performance monitoring."""
        
        @async_rate_limit(rate_limit=10.0, burst_limit=2)
        async def rate_limited_api_call():
            await asyncio.sleep(0.01)
            return "api_response"
            
        # Multiple calls should be rate limited
        results = []
        for _ in range(3):
            result = await rate_limited_api_call()
            results.append(result)
            
        assert all(r == "api_response" for r in results)
        
    @pytest.mark.asyncio
    async def test_async_retry_with_error_handling(self) -> None:
        """Test async retry with error handling."""
        call_count = 0
        
        @async_retry(max_attempts=3, base_delay=0.01)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
            
        result = await flaky_operation()
        assert result == "success"
        assert call_count == 3


class TestEndToEndIntegration:
    """Test complete end-to-end integration."""
    
    def test_complete_trading_workflow(self) -> None:
        """Test complete trading workflow with all improvements."""
        # Set up logging
        logger = get_logger("integration_test", component="end_to_end")
        
        # Load configuration
        config = Mock()
        config.system = Mock()
        config.system.max_position_size = Decimal("10000")
        config.trading = Mock()
        config.trading.risk_limits = {}
        
        # Create broker using factory
        factory = ProductFactory()
        broker = DeterministicBroker(
            products=factory.create_test_products(["BTC-PERP"]),
            equity=Decimal("100000")
        )
        
        # Create stores
        event_store = Mock()
        account_store = Mock()
        market_data = Mock()
        
        # Build bot using builder pattern
        bot = (
            PerpsBotBuilder()
            .with_config(config)
            .with_event_store(event_store)
            .with_account_store(account_store)
            .with_brokerage(broker)
            .with_market_data(market_data)
            .build()
        )
        
        # Set up health checks
        setup_basic_health_checks(brokerage=broker)
        
        # Perform monitored operation
        @monitor_trading_operation("integration_test_trade")
        @handle_brokerage_errors("integration_test_trade")
        def test_trade():
            with log_operation("place_test_order", logger):
                log_trade_event(
                    "order_placed",
                    "BTC-PERP",
                    side="buy",
                    quantity="0.1",
                    price="50000"
                )
                return "order_placed"
                
        result = test_trade()
        assert result == "order_placed"
        
        # Check system health
        health = get_health_summary()
        assert health["status"] in ["healthy", "degraded", "unhealthy", "unknown"]
        
    @pytest.mark.asyncio
    async def test_async_integration_workflow(self) -> None:
        """Test async integration workflow."""
        logger = get_logger("async_integration", component="end_to_end")
        
        @async_rate_limit(rate_limit=5.0)
        @async_retry(max_attempts=2, base_delay=0.01)
        async def async_trading_operation(symbol: str) -> str:
            await asyncio.sleep(0.01)
            log_trade_event("async_order_filled", symbol, side="buy", quantity="0.1")
            return f"Async order filled for {symbol}"
            
        result = await async_trading_operation("BTC-PERP")
        assert "Async order filled" in result


class TestPerformanceIntegration:
    """Test performance across all integrated components."""
    
    def test_performance_of_integrated_system(self) -> None:
        """Test performance of the integrated system."""
        start_time = time.time()
        
        # Create all components
        factory = ProductFactory()
        broker = DeterministicBroker(
            products=factory.create_test_products(["BTC-PERP", "ETH-PERP"]),
            equity=Decimal("100000")
        )
        
        # Set up monitoring
        setup_basic_health_checks(brokerage=broker)
        
        # Perform operations
        for i in range(5):
            @monitor_trading_operation(f"test_operation_{i}")
            def operation():
                time.sleep(0.001)  # Small work
                return f"result_{i}"
                
            operation()
            
        elapsed = time.time() - start_time
        
        # Should complete quickly
        assert elapsed < 1.0
        
        # Check health
        health = get_health_summary()
        assert isinstance(health, dict)


class TestErrorRecoveryIntegration:
    """Test error recovery across integrated components."""
    
    def test_error_recovery_with_monitoring(self) -> None:
        """Test error recovery with monitoring and logging."""
        logger = get_logger("error_recovery", component="integration")
        
        call_count = 0
        
        @monitor_trading_operation("error_recovery_test")
        @handle_brokerage_errors("error_recovery_test")
        def operation_with_recovery():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # First call fails
                raise ConnectionError("Network error")
            else:
                # Second call succeeds
                log_trade_event("order_filled", "BTC-PERP", side="buy", quantity="0.1")
                return "success"
                
        # First call should fail
        with pytest.raises(ConnectionError):
            operation_with_recovery()
            
        # Second call should succeed
        result = operation_with_recovery()
        assert result == "success"
        assert call_count == 2


class TestConfigurationValidationIntegration:
    """Test configuration validation integration."""
    
    def test_configuration_validation_with_error_handling(self) -> None:
        """Test configuration validation with proper error handling."""
        logger = get_logger("config_validation", component="integration")
        
        # Test loading invalid configuration
        try:
            with ErrorContext("config_validation", logger=logger):
                # This would normally load from file, but we'll test the validation
                config = RiskConfig(max_leverage=-1)  # Invalid leverage
        except Exception:
            # Expected to fail with invalid config
            pass
            
        # Test loading valid configuration
        config = RiskConfig(max_leverage=5, min_liquidation_buffer_pct=0.15)
        assert config.max_leverage == 5
        assert config.min_liquidation_buffer_pct == 0.15


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])
