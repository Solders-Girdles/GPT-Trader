"""Tests for health check system."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.monitoring.health_checks import (
    HealthStatus,
    HealthCheckResult,
    HealthChecker,
    DatabaseHealthCheck,
    APIHealthCheck,
    BrokerageHealthCheck,
    MemoryHealthCheck,
    PerformanceHealthCheck,
    HealthCheckRegistry,
    HealthCheckEndpoint,
    setup_basic_health_checks,
    get_health_registry,
    get_health_summary,
)


class TestHealthStatus:
    """Test HealthStatus enumeration."""
    
    def test_health_status_values(self) -> None:
        """Test health status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    """Test HealthCheckResult functionality."""
    
    def test_health_check_result_creation(self) -> None:
        """Test creating a health check result."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
            details={"key": "value"}
        )
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.details["key"] == "value"
        assert result.timestamp > 0
        assert result.duration_ms == 0.0
        
    def test_health_check_result_string_representation(self) -> None:
        """Test string representation of health check result."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good"
        )
        
        str_repr = str(result)
        assert "test_check" in str_repr
        assert "healthy" in str_repr
        assert "All good" in str_repr
        
    def test_health_check_result_to_dict(self) -> None:
        """Test converting health check result to dictionary."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
            details={"key": "value"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["name"] == "test_check"
        assert result_dict["status"] == "healthy"
        assert result_dict["message"] == "All good"
        assert result_dict["details"]["key"] == "value"
        assert "timestamp" in result_dict
        assert "duration_ms" in result_dict


class TestHealthChecker:
    """Test HealthChecker base class."""
    
    @pytest.mark.asyncio
    async def test_health_checker_timeout(self) -> None:
        """Test health checker timeout."""
        
        class SlowHealthCheck(HealthChecker):
            async def _do_check(self):
                await asyncio.sleep(0.1)  # Longer than timeout
                return HealthCheckResult("slow", HealthStatus.HEALTHY, "OK")
                
        checker = SlowHealthCheck("slow_check", timeout=0.01)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower()
        
    @pytest.mark.asyncio
    async def test_health_checker_exception(self) -> None:
        """Test health checker exception handling."""
        
        class FailingHealthCheck(HealthChecker):
            async def _do_check(self):
                raise ValueError("Test error")
                
        checker = FailingHealthCheck("failing_check")
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()
        assert result.details["error_type"] == "ValueError"


class TestDatabaseHealthCheck:
    """Test DatabaseHealthCheck functionality."""
    
    @pytest.mark.asyncio
    async def test_database_health_check_success(self) -> None:
        """Test successful database health check."""
        mock_connection = AsyncMock()
        mock_connection.execute.return_value = None
        
        def connection_factory():
            return mock_connection
            
        checker = DatabaseHealthCheck(connection_factory)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.HEALTHY
        assert "successful" in result.message.lower()
        assert "query_time_ms" in result.details
        
    @pytest.mark.asyncio
    async def test_database_health_check_failure(self) -> None:
        """Test failing database health check."""
        def connection_factory():
            raise ConnectionError("Database connection failed")
            
        checker = DatabaseHealthCheck(connection_factory)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()


class TestAPIHealthCheck:
    """Test APIHealthCheck functionality."""
    
    @pytest.mark.asyncio
    async def test_api_health_check_success(self) -> None:
        """Test successful API health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        checker = APIHealthCheck(mock_client, "/health")
        result = await checker.check_health()
        
        assert result.status == HealthStatus.HEALTHY
        assert "responsive" in result.message.lower()
        assert result.details["status_code"] == 200
        
    @pytest.mark.asyncio
    async def test_api_health_check_degraded(self) -> None:
        """Test degraded API health check."""
        mock_response = Mock()
        mock_response.status_code = 500
        
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        
        checker = APIHealthCheck(mock_client, "/health")
        result = await checker.check_health()
        
        assert result.status == HealthStatus.DEGRADED
        assert "500" in result.message
        
    @pytest.mark.asyncio
    async def test_api_health_check_failure(self) -> None:
        """Test failing API health check."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = ConnectionError("API unreachable")
        
        checker = APIHealthCheck(mock_client, "/health")
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()


class TestBrokerageHealthCheck:
    """Test BrokerageHealthCheck functionality."""
    
    @pytest.mark.asyncio
    async def test_brokerage_health_check_success(self) -> None:
        """Test successful brokerage health check."""
        mock_brokerage = Mock()
        mock_brokerage.validate_connection.return_value = True
        mock_brokerage.list_balances.return_value = [{"asset": "BTC", "balance": "1.0"}]
        mock_brokerage.list_products.return_value = [{"id": "BTC-PERP"}]
        mock_brokerage.get_account_id.return_value = "account123"
        
        checker = BrokerageHealthCheck(mock_brokerage)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.HEALTHY
        assert "healthy" in result.message.lower()
        assert result.details["balance_count"] == 1
        assert result.details["product_count"] == 1
        
    @pytest.mark.asyncio
    async def test_brokerage_health_check_not_connected(self) -> None:
        """Test brokerage health check when not connected."""
        mock_brokerage = Mock()
        mock_brokerage.validate_connection.return_value = False
        
        checker = BrokerageHealthCheck(mock_brokerage)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "not connected" in result.message.lower()
        
    @pytest.mark.asyncio
    async def test_brokerage_health_check_failure(self) -> None:
        """Test failing brokerage health check."""
        mock_brokerage = Mock()
        mock_brokerage.validate_connection.side_effect = Exception("Brokerage error")
        
        checker = BrokerageHealthCheck(mock_brokerage)
        result = await checker.check_health()
        
        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()


class TestMemoryHealthCheck:
    """Test MemoryHealthCheck functionality."""
    
    @pytest.mark.asyncio
    async def test_memory_health_check_healthy(self) -> None:
        """Test healthy memory usage."""
        with patch('bot_v2.monitoring.health_checks.psutil') as mock_psutil:
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(rss=500 * 1024 * 1024)  # 500MB
            mock_process.memory_percent.return_value = 25.0
            mock_psutil.Process.return_value = mock_process
            
            checker = MemoryHealthCheck(warning_threshold_mb=1000, critical_threshold_mb=2000)
            result = await checker.check_health()
            
            assert result.status == HealthStatus.HEALTHY
            assert "normal" in result.message.lower()
            assert result.details["memory_mb"] == 500.0
            
    @pytest.mark.asyncio
    async def test_memory_health_check_degraded(self) -> None:
        """Test degraded memory usage."""
        with patch('bot_v2.monitoring.health_checks.psutil') as mock_psutil:
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(rss=1500 * 1024 * 1024)  # 1500MB
            mock_process.memory_percent.return_value = 75.0
            mock_psutil.Process.return_value = mock_process
            
            checker = MemoryHealthCheck(warning_threshold_mb=1000, critical_threshold_mb=2000)
            result = await checker.check_health()
            
            assert result.status == HealthStatus.DEGRADED
            assert "high" in result.message.lower()
            
    @pytest.mark.asyncio
    async def test_memory_health_check_unhealthy(self) -> None:
        """Test unhealthy memory usage."""
        with patch('bot_v2.monitoring.health_checks.psutil') as mock_psutil:
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(rss=2500 * 1024 * 1024)  # 2500MB
            mock_process.memory_percent.return_value = 90.0
            mock_psutil.Process.return_value = mock_process
            
            checker = MemoryHealthCheck(warning_threshold_mb=1000, critical_threshold_mb=2000)
            result = await checker.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY
            assert "critical" in result.message.lower()
            
    @pytest.mark.asyncio
    async def test_memory_health_check_no_psutil(self) -> None:
        """Test memory health check without psutil."""
        with patch('bot_v2.monitoring.health_checks.psutil', side_effect=ImportError):
            checker = MemoryHealthCheck()
            result = await checker.check_health()
            
            assert result.status == HealthStatus.UNKNOWN
            assert "psutil not available" in result.message.lower()


class TestPerformanceHealthCheck:
    """Test PerformanceHealthCheck functionality."""
    
    @pytest.mark.asyncio
    async def test_performance_health_check_healthy(self) -> None:
        """Test healthy performance metrics."""
        with patch('bot_v2.monitoring.health_checks.get_performance_health_check') as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "issues": [],
                "metrics": {"total_metrics": 5}
            }
            
            checker = PerformanceHealthCheck()
            result = await checker.check_health()
            
            assert result.status == HealthStatus.HEALTHY
            assert result.details["issues"] == []
            
    @pytest.mark.asyncio
    async def test_performance_health_check_degraded(self) -> None:
        """Test degraded performance metrics."""
        with patch('bot_v2.monitoring.health_checks.get_performance_health_check') as mock_health:
            mock_health.return_value = {
                "status": "degraded",
                "issues": ["Slow operation detected"],
                "metrics": {"total_metrics": 5}
            }
            
            checker = PerformanceHealthCheck()
            result = await checker.check_health()
            
            assert result.status == HealthStatus.DEGRADED
            assert len(result.details["issues"]) == 1
            
    @pytest.mark.asyncio
    async def test_performance_health_check_failure(self) -> None:
        """Test failing performance health check."""
        with patch('bot_v2.monitoring.health_checks.get_performance_health_check') as mock_health:
            mock_health.side_effect = Exception("Performance check failed")
            
            checker = PerformanceHealthCheck()
            result = await checker.check_health()
            
            assert result.status == HealthStatus.UNHEALTHY
            assert "failed" in result.message.lower()


class TestHealthCheckRegistry:
    """Test HealthCheckRegistry functionality."""
    
    def test_health_check_registry_initialization(self) -> None:
        """Test registry initialization."""
        registry = HealthCheckRegistry()
        assert len(registry._checkers) == 0
        
    def test_health_check_registry_register(self) -> None:
        """Test registering health checkers."""
        registry = HealthCheckRegistry()
        checker = MemoryHealthCheck()
        
        registry.register(checker)
        
        assert "memory" in registry._checkers
        assert registry._checkers["memory"] is checker
        
    def test_health_check_registry_unregister(self) -> None:
        """Test unregistering health checkers."""
        registry = HealthCheckRegistry()
        checker = MemoryHealthCheck()
        registry.register(checker)
        
        registry.unregister("memory")
        
        assert "memory" not in registry._checkers
        
    def test_health_check_registry_get_checker(self) -> None:
        """Test getting health checkers."""
        registry = HealthCheckRegistry()
        checker = MemoryHealthCheck()
        registry.register(checker)
        
        retrieved = registry.get_checker("memory")
        assert retrieved is checker
        
        nonexistent = registry.get_checker("nonexistent")
        assert nonexistent is None
        
    def test_health_check_registry_list_checkers(self) -> None:
        """Test listing health checkers."""
        registry = HealthCheckRegistry()
        
        registry.register(MemoryHealthCheck())
        registry.register(PerformanceHealthCheck())
        
        checkers = registry.list_checkers()
        assert len(checkers) == 2
        assert "memory" in checkers
        assert "performance" in checkers
        
    @pytest.mark.asyncio
    async def test_health_check_registry_run_all_checks(self) -> None:
        """Test running all health checks."""
        registry = HealthCheckRegistry()
        
        # Add checkers that will succeed
        registry.register(MemoryHealthCheck())
        registry.register(PerformanceHealthCheck())
        
        results = await registry.run_all_checks()
        
        assert len(results) == 2
        assert all(isinstance(result, HealthCheckResult) for result in results)
        
    @pytest.mark.asyncio
    async def test_health_check_registry_run_all_checks_empty(self) -> None:
        """Test running all checks with no checkers registered."""
        registry = HealthCheckRegistry()
        
        results = await registry.run_all_checks()
        
        assert len(results) == 0
        
    @pytest.mark.asyncio
    async def test_health_check_registry_run_specific_check(self) -> None:
        """Test running specific health check."""
        registry = HealthCheckRegistry()
        checker = MemoryHealthCheck()
        registry.register(checker)
        
        result = await registry.run_check("memory")
        
        assert isinstance(result, HealthCheckResult)
        assert result.name == "memory"
        
    @pytest.mark.asyncio
    async def test_health_check_registry_run_nonexistent_check(self) -> None:
        """Test running nonexistent health check."""
        registry = HealthCheckRegistry()
        
        result = await registry.run_check("nonexistent")
        
        assert result is None


class TestHealthCheckEndpoint:
    """Test HealthCheckEndpoint functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint_get_health_status_all(self) -> None:
        """Test getting health status for all checks."""
        registry = HealthCheckRegistry()
        registry.register(MemoryHealthCheck())
        
        endpoint = HealthCheckEndpoint(registry)
        status = await endpoint.get_health_status()
        
        assert "status" in status
        assert "timestamp" in status
        assert "checks" in status
        assert len(status["checks"]) == 1
        
    @pytest.mark.asyncio
    async def test_health_check_endpoint_get_health_status_specific(self) -> None:
        """Test getting health status for specific check."""
        registry = HealthCheckRegistry()
        registry.register(MemoryHealthCheck())
        
        endpoint = HealthCheckEndpoint(registry)
        status = await endpoint.get_health_status("memory")
        
        assert "status" in status
        assert "timestamp" in status
        assert "checks" in status
        assert len(status["checks"]) == 1
        assert status["checks"][0]["name"] == "memory"
        
    @pytest.mark.asyncio
    async def test_health_check_endpoint_get_health_status_nonexistent(self) -> None:
        """Test getting health status for nonexistent check."""
        registry = HealthCheckRegistry()
        
        endpoint = HealthCheckEndpoint(registry)
        status = await endpoint.get_health_status("nonexistent")
        
        assert status["status"] == "unknown"
        assert "not found" in status["message"].lower()
        
    @pytest.mark.asyncio
    async def test_health_check_endpoint_get_health_status_empty(self) -> None:
        """Test getting health status with no checks."""
        registry = HealthCheckRegistry()
        
        endpoint = HealthCheckEndpoint(registry)
        status = await endpoint.get_health_status()
        
        assert status["status"] == "unknown"
        assert "No health checks configured" in status["message"]
        
    @pytest.mark.asyncio
    async def test_health_check_endpoint_get_liveness(self) -> None:
        """Test liveness endpoint."""
        registry = HealthCheckRegistry()
        endpoint = HealthCheckEndpoint(registry)
        
        liveness = await endpoint.get_liveness()
        
        assert liveness["status"] == "alive"
        assert "timestamp" in liveness
        
    @pytest.mark.asyncio
    async def test_health_check_endpoint_get_readiness(self) -> None:
        """Test readiness endpoint."""
        registry = HealthCheckRegistry()
        endpoint = HealthCheckEndpoint(registry)
        
        readiness = await endpoint.get_readiness()
        
        assert readiness["status"] in ["ready", "not_ready"]
        assert "timestamp" in readiness
        
    def test_health_check_endpoint_include_details(self) -> None:
        """Test endpoint details inclusion."""
        registry = HealthCheckRegistry()
        
        endpoint_with_details = HealthCheckEndpoint(registry, include_details=True)
        endpoint_without_details = HealthCheckEndpoint(registry, include_details=False)
        
        assert endpoint_with_details.include_details is True
        assert endpoint_without_details.include_details is False


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_setup_basic_health_checks(self) -> None:
        """Test setting up basic health checks."""
        registry = get_health_registry()
        
        # Clear any existing checks
        registry._checkers.clear()
        
        # Set up basic checks
        setup_basic_health_checks()
        
        # Should have memory and performance checks
        checkers = registry.list_checkers()
        assert "memory" in checkers
        assert "performance" in checkers
        
    def test_setup_basic_health_checks_with_components(self) -> None:
        """Test setting up basic health checks with components."""
        registry = get_health_registry()
        registry._checkers.clear()
        
        mock_db = Mock()
        mock_brokerage = Mock()
        mock_api = Mock()
        
        setup_basic_health_checks(
            database_connection=mock_db,
            brokerage=mock_brokerage,
            api_client=mock_api
        )
        
        checkers = registry.list_checkers()
        assert "memory" in checkers
        assert "performance" in checkers
        assert "database" in checkers
        assert "brokerage" in checkers
        assert "api" in checkers
        
    def test_get_health_registry(self) -> None:
        """Test getting health registry."""
        registry = get_health_registry()
        assert isinstance(registry, HealthCheckRegistry)
        
        # Should return same instance
        registry2 = get_health_registry()
        assert registry is registry2
        
    def test_get_health_summary(self) -> None:
        """Test getting health summary."""
        summary = get_health_summary()
        
        assert isinstance(summary, dict)
        assert "status" in summary
        assert "timestamp" in summary


class TestHealthCheckIntegration:
    """Test health check integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_health_check_flow(self) -> None:
        """Test end-to-end health check flow."""
        registry = HealthCheckRegistry()
        
        # Add various health checks
        registry.register(MemoryHealthCheck())
        registry.register(PerformanceHealthCheck())
        
        # Create endpoint
        endpoint = HealthCheckEndpoint(registry)
        
        # Get overall health status
        health_status = await endpoint.get_health_status()
        
        assert "status" in health_status
        assert "checks" in health_status
        assert len(health_status["checks"]) == 2
        
        # Get liveness
        liveness = await endpoint.get_liveness()
        assert liveness["status"] == "alive"
        
        # Get readiness
        readiness = await endpoint.get_readiness()
        assert readiness["status"] in ["ready", "not_ready"]
        
    @pytest.mark.asyncio
    async def test_health_check_with_mixed_results(self) -> None:
        """Test health checks with mixed success/failure results."""
        registry = HealthCheckRegistry()
        
        # Add a check that will succeed
        registry.register(MemoryHealthCheck())
        
        # Add a check that will fail
        class FailingCheck(HealthChecker):
            async def _do_check(self):
                return HealthCheckResult(
                    "failing",
                    HealthStatus.UNHEALTHY,
                    "Check failed"
                )
                
        registry.register(FailingCheck("failing"))
        
        endpoint = HealthCheckEndpoint(registry)
        status = await endpoint.get_health_status()
        
        # Overall status should be unhealthy due to one failing check
        assert status["status"] == "unhealthy"
        assert len(status["checks"]) == 2
        
        # Find the failing check
        failing_check = next(c for c in status["checks"] if c["name"] == "failing")
        assert failing_check["status"] == "unhealthy"


class TestHealthCheckEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_health_check_concurrent_execution(self) -> None:
        """Test that health checks run concurrently."""
        import time
        
        class SlowCheck(HealthChecker):
            async def _do_check(self):
                await asyncio.sleep(0.01)
                return HealthCheckResult(
                    self.name,
                    HealthStatus.HEALTHY,
                    "OK"
                )
                
        registry = HealthCheckRegistry()
        registry.register(SlowCheck("check1"))
        registry.register(SlowCheck("check2"))
        registry.register(SlowCheck("check3"))
        
        start_time = time.time()
        results = await registry.run_all_checks()
        elapsed = time.time() - start_time
        
        # Should run concurrently, so total time should be close to single check time
        assert elapsed < 0.02  # Much less than 0.03 (3 * 0.01)
        assert len(results) == 3
        
    @pytest.mark.asyncio
    async def test_health_check_exception_in_registry(self) -> None:
        """Test handling exceptions in registry operations."""
        registry = HealthCheckRegistry()
        
        class ExceptionCheck(HealthChecker):
            async def _do_check(self):
                raise ValueError("Unexpected error")
                
        registry.register(ExceptionCheck("exception"))
        
        results = await registry.run_all_checks()
        
        assert len(results) == 1
        assert results[0].status == HealthStatus.UNHEALTHY
        assert "exception" in results[0].details["error_type"].lower()
        
    def test_health_check_registry_duplicate_registration(self) -> None:
        """Test registering duplicate health check names."""
        registry = HealthCheckRegistry()
        
        checker1 = MemoryHealthCheck("memory")
        checker2 = MemoryHealthCheck("memory")
        
        registry.register(checker1)
        registry.register(checker2)  # Should overwrite
        
        assert len(registry._checkers) == 1
        assert registry._checkers["memory"] is checker2
