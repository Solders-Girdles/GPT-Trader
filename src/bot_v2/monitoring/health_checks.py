"""Health check endpoints and monitoring utilities."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.performance_monitoring import get_performance_health_check

logger = get_logger("health", component="monitoring")


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    
    def __str__(self) -> str:
        """String representation of health check result."""
        return f"{self.name}: {self.status.value} - {self.message}"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class HealthChecker:
    """Base class for health check implementations."""
    
    def __init__(self, name: str, timeout: float = 10.0) -> None:
        """Initialize health checker.
        
        Args:
            name: Name of the health check
            timeout: Timeout for health check in seconds
        """
        self.name = name
        self.timeout = timeout
        
    async def check_health(self) -> HealthCheckResult:
        """Perform health check.
        
        Returns:
            Health check result
        """
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(self._do_check(), timeout=self.timeout)
            result.duration_ms = (time.time() - start_time) * 1000
            return result
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Health check {self.name} failed: {e}")
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error_type": type(e).__name__},
                duration_ms=(time.time() - start_time) * 1000,
            )
            
    async def _do_check(self) -> HealthCheckResult:
        """Implement actual health check logic.
        
        Returns:
            Health check result
        """
        raise NotImplementedError("Subclasses must implement _do_check")


class DatabaseHealthCheck(HealthChecker):
    """Health check for database connectivity."""
    
    def __init__(self, connection_factory: Callable[[], Any], name: str = "database") -> None:
        """Initialize database health check.
        
        Args:
            connection_factory: Function that returns database connection
            name: Name of health check
        """
        super().__init__(name)
        self.connection_factory = connection_factory
        
    async def _do_check(self) -> HealthCheckResult:
        """Check database connectivity.
        
        Returns:
            Health check result
        """
        try:
            # Try to get connection and execute simple query
            connection = self.connection_factory()
            
            # Simple connectivity check
            start_time = time.time()
            await connection.execute("SELECT 1")
            query_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={
                    "query_time_ms": query_time,
                    "connection_type": type(connection).__name__,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                details={"error_type": type(e).__name__},
            )


class APIHealthCheck(HealthChecker):
    """Health check for external API connectivity."""
    
    def __init__(
        self,
        api_client: Any,
        endpoint: str = "/health",
        name: str = "api",
    ) -> None:
        """Initialize API health check.
        
        Args:
            api_client: API client instance
            endpoint: Health check endpoint
            name: Name of health check
        """
        super().__init__(name)
        self.api_client = api_client
        self.endpoint = endpoint
        
    async def _do_check(self) -> HealthCheckResult:
        """Check API connectivity.
        
        Returns:
            Health check result
        """
        try:
            start_time = time.time()
            response = await self.api_client.get(self.endpoint)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.HEALTHY,
                    message="API endpoint responsive",
                    details={
                        "response_time_ms": response_time,
                        "status_code": response.status_code,
                        "endpoint": self.endpoint,
                    },
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"API returned status {response.status_code}",
                    details={
                        "response_time_ms": response_time,
                        "status_code": response.status_code,
                        "endpoint": self.endpoint,
                    },
                )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"API health check failed: {str(e)}",
                details={"error_type": type(e).__name__},
            )


class BrokerageHealthCheck(HealthChecker):
    """Health check for brokerage connectivity."""
    
    def __init__(self, brokerage: Any, name: str = "brokerage") -> None:
        """Initialize brokerage health check.
        
        Args:
            brokerage: Brokerage instance
            name: Name of health check
        """
        super().__init__(name)
        self.brokerage = brokerage
        
    async def _do_check(self) -> HealthCheckResult:
        """Check brokerage connectivity.
        
        Returns:
            Health check result
        """
        try:
            # Check connection status
            is_connected = self.brokerage.validate_connection()
            
            if not is_connected:
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.UNHEALTHY,
                    message="Brokerage not connected",
                )
                
            # Try to get account info
            start_time = time.time()
            balances = self.brokerage.list_balances()
            response_time = (time.time() - start_time) * 1000
            
            # Try to get product list
            products = self.brokerage.list_products()
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY,
                message="Brokerage connection healthy",
                details={
                    "response_time_ms": response_time,
                    "balance_count": len(balances),
                    "product_count": len(products),
                    "account_id": self.brokerage.get_account_id(),
                },
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Brokerage health check failed: {str(e)}",
                details={"error_type": type(e).__name__},
            )


class MemoryHealthCheck(HealthChecker):
    """Health check for memory usage."""
    
    def __init__(
        self,
        warning_threshold_mb: float = 1000.0,
        critical_threshold_mb: float = 2000.0,
        name: str = "memory",
    ) -> None:
        """Initialize memory health check.
        
        Args:
            warning_threshold_mb: Memory usage warning threshold in MB
            critical_threshold_mb: Memory usage critical threshold in MB
            name: Name of health check
        """
        super().__init__(name)
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        
    async def _do_check(self) -> HealthCheckResult:
        """Check memory usage.
        
        Returns:
            Health check result
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > self.critical_threshold_mb:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory_mb:.1f}MB"
            elif memory_mb > self.warning_threshold_mb:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_mb:.1f}MB"
                
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details={
                    "memory_mb": memory_mb,
                    "warning_threshold_mb": self.warning_threshold_mb,
                    "critical_threshold_mb": self.critical_threshold_mb,
                    "memory_percent": process.memory_percent(),
                },
            )
        except ImportError:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message="psutil not available for memory monitoring",
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Memory health check failed: {str(e)}",
                details={"error_type": type(e).__name__},
            )


class PerformanceHealthCheck(HealthChecker):
    """Health check based on performance metrics."""
    
    def __init__(
        self,
        slow_operation_threshold_s: float = 1.0,
        very_slow_operation_threshold_s: float = 5.0,
        name: str = "performance",
    ) -> None:
        """Initialize performance health check.
        
        Args:
            slow_operation_threshold_s: Threshold for slow operations
            very_slow_operation_threshold_s: Threshold for very slow operations
            name: Name of health check
        """
        super().__init__(name)
        self.slow_operation_threshold_s = slow_operation_threshold_s
        self.very_slow_operation_threshold_s = very_slow_operation_threshold_s
        
    async def _do_check(self) -> HealthCheckResult:
        """Check performance metrics.
        
        Returns:
            Health check result
        """
        try:
            perf_health = get_performance_health_check()
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus(perf_health["status"]),
                message=f"Performance status: {perf_health['status']}",
                details={
                    "issues": perf_health["issues"],
                    "metrics": perf_health["metrics"],
                    "slow_threshold_s": self.slow_operation_threshold_s,
                    "very_slow_threshold_s": self.very_slow_operation_threshold_s,
                },
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Performance health check failed: {str(e)}",
                details={"error_type": type(e).__name__},
            )


class HealthCheckRegistry:
    """Registry for managing health checks."""
    
    def __init__(self) -> None:
        """Initialize health check registry."""
        self._checkers: Dict[str, HealthChecker] = {}
        
    def register(self, checker: HealthChecker) -> None:
        """Register a health checker.
        
        Args:
            checker: Health checker to register
        """
        self._checkers[checker.name] = checker
        logger.info(f"Registered health check: {checker.name}")
        
    def unregister(self, name: str) -> None:
        """Unregister a health checker.
        
        Args:
            name: Name of health checker to unregister
        """
        if name in self._checkers:
            del self._checkers[name]
            logger.info(f"Unregistered health check: {name}")
            
    def get_checker(self, name: str) -> Optional[HealthChecker]:
        """Get a health checker by name.
        
        Args:
            name: Name of health checker
            
        Returns:
            Health checker or None if not found
        """
        return self._checkers.get(name)
        
    def list_checkers(self) -> List[str]:
        """List all registered health checker names.
        
        Returns:
            List of health checker names
        """
        return list(self._checkers.keys())
        
    async def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks.
        
        Returns:
            List of health check results
        """
        if not self._checkers:
            logger.warning("No health checks registered")
            return []
            
        logger.info(f"Running {len(self._checkers)} health checks")
        
        # Run all checks concurrently
        tasks = [checker.check_health() for checker in self._checkers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to unhealthy results
        health_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                checker_name = list(self._checkers.keys())[i]
                health_results.append(HealthCheckResult(
                    name=checker_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed with exception: {str(result)}",
                    details={"error_type": type(result).__name__},
                ))
            else:
                health_results.append(result)
                
        # Log results
        healthy_count = sum(1 for r in health_results if r.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for r in health_results if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in health_results if r.status == HealthStatus.UNHEALTHY)
        
        logger.info(
            f"Health checks completed: {healthy_count} healthy, "
            f"{degraded_count} degraded, {unhealthy_count} unhealthy"
        )
        
        return health_results
        
    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check.
        
        Args:
            name: Name of health check to run
            
        Returns:
            Health check result or None if not found
        """
        checker = self.get_checker(name)
        if not checker:
            logger.warning(f"Health check not found: {name}")
            return None
            
        logger.info(f"Running health check: {name}")
        return await checker.check_health()


# Global health check registry
_global_registry = HealthCheckRegistry()


def get_health_registry() -> HealthCheckRegistry:
    """Get the global health check registry.
    
    Returns:
        Global health check registry
    """
    return _global_registry


class HealthCheckEndpoint:
    """HTTP endpoint for health checks."""
    
    def __init__(
        self,
        registry: HealthCheckRegistry | None = None,
        include_details: bool = True,
    ) -> None:
        """Initialize health check endpoint.
        
        Args:
            registry: Health check registry (uses global if None)
            include_details: Whether to include detailed information
        """
        self.registry = registry or _global_registry
        self.include_details = include_details
        
    async def get_health_status(self, check_name: str | None = None) -> Dict[str, Any]:
        """Get health status.
        
        Args:
            check_name: Specific check to run (None for all)
            
        Returns:
            Health status response
        """
        if check_name:
            result = await self.registry.run_check(check_name)
            if not result:
                return {
                    "status": "unknown",
                    "message": f"Health check '{check_name}' not found",
                    "timestamp": time.time(),
                }
                
            overall_status = result.status.value
            checks = [result.to_dict()]
        else:
            results = await self.registry.run_all_checks()
            
            if not results:
                return {
                    "status": "unknown",
                    "message": "No health checks configured",
                    "timestamp": time.time(),
                }
                
            # Determine overall status
            if any(r.status == HealthStatus.UNHEALTHY for r in results):
                overall_status = "unhealthy"
            elif any(r.status == HealthStatus.DEGRADED for r in results):
                overall_status = "degraded"
            else:
                overall_status = "healthy"
                
            checks = [r.to_dict() for r in results]
            
        response = {
            "status": overall_status,
            "timestamp": time.time(),
            "checks": checks,
        }
        
        if self.include_details:
            response["summary"] = {
                "total_checks": len(checks),
                "healthy": sum(1 for c in checks if c["status"] == "healthy"),
                "degraded": sum(1 for c in checks if c["status"] == "degraded"),
                "unhealthy": sum(1 for c in checks if c["status"] == "unhealthy"),
                "unknown": sum(1 for c in checks if c["status"] == "unknown"),
            }
            
        return response
        
    async def get_liveness(self) -> Dict[str, Any]:
        """Simple liveness check.
        
        Returns:
            Liveness status
        """
        return {
            "status": "alive",
            "timestamp": time.time(),
        }
        
    async def get_readiness(self) -> Dict[str, Any]:
        """Readiness check.
        
        Returns:
            Readiness status
        """
        # Run critical health checks
        critical_checks = ["database", "brokerage"]
        results = []
        
        for check_name in critical_checks:
            result = await self.registry.run_check(check_name)
            if result:
                results.append(result)
                
        # Determine readiness
        if all(r.status == HealthStatus.HEALTHY for r in results):
            status = "ready"
        else:
            status = "not_ready"
            
        return {
            "status": status,
            "timestamp": time.time(),
            "checks": [r.to_dict() for r in results],
        }


# Utility functions for common health check setups

def setup_basic_health_checks(
    database_connection: Any | None = None,
    brokerage: Any | None = None,
    api_client: Any | None = None,
) -> None:
    """Set up basic health checks.
    
    Args:
        database_connection: Database connection for health check
        brokerage: Brokerage instance for health check
        api_client: API client for health check
    """
    registry = get_health_registry()
    
    # Always add memory and performance checks
    registry.register(MemoryHealthCheck())
    registry.register(PerformanceHealthCheck())
    
    # Add optional checks if components are provided
    if database_connection:
        registry.register(DatabaseHealthCheck(
            connection_factory=lambda: database_connection
        ))
        
    if brokerage:
        registry.register(BrokerageHealthCheck(brokerage))
        
    if api_client:
        registry.register(APIHealthCheck(api_client))


def get_health_summary() -> Dict[str, Any]:
    """Get quick health summary.
    
    Returns:
        Health summary
    """
    registry = get_health_registry()
    endpoint = HealthCheckEndpoint(registry, include_details=False)
    
    # Run health checks synchronously for simple summary
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already in async context, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, endpoint.get_health_status())
                return future.result(timeout=10.0)
        else:
            # If not in async context, run directly
            return asyncio.run(endpoint.get_health_status())
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        return {
            "status": "unknown",
            "message": f"Health check failed: {str(e)}",
            "timestamp": time.time(),
        }
