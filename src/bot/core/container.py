"""
GPT-Trader Dependency Injection Container

Service container providing automated component lifecycle management:
- Interface-based dependency resolution
- Automatic component wiring and initialization
- Singleton and factory service patterns
- Circular dependency detection
- Service lifecycle coordination
- Component health monitoring integration

This eliminates manual component wiring throughout the codebase and enables
loose coupling between components through interface-based dependencies.
"""

import inspect
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .base import BaseComponent, ComponentStatus
from .exceptions import (
    ComponentException,
    raise_config_error,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceLifetime(Enum):
    """Service lifetime management"""

    SINGLETON = "singleton"  # Single instance for application lifetime
    SCOPED = "scoped"  # Single instance per scope/request
    TRANSIENT = "transient"  # New instance every time


class ServiceStatus(Enum):
    """Service registration status"""

    REGISTERED = "registered"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ServiceDescriptor:
    """Service registration descriptor"""

    service_type: type
    implementation_type: type | None = None
    factory: Callable | None = None
    instance: Any | None = None
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON

    # Dependencies
    dependencies: set[type] = field(default_factory=set)
    dependents: set[type] = field(default_factory=set)

    # Status tracking
    status: ServiceStatus = ServiceStatus.REGISTERED
    registration_time: datetime = field(default_factory=datetime.now)
    initialization_time: datetime | None = None

    # Configuration
    auto_start: bool = True
    required: bool = True

    def __post_init__(self):
        """Validate service descriptor"""
        if not self.implementation_type and not self.factory and not self.instance:
            raise_config_error("Service must have implementation_type, factory, or instance")

        if self.instance and self.lifetime != ServiceLifetime.SINGLETON:
            raise_config_error("Existing instances must use SINGLETON lifetime")


class ServiceScope:
    """Service scope for scoped service lifetime management"""

    def __init__(self, scope_id: str) -> None:
        self.scope_id = scope_id
        self.scoped_services: dict[type, Any] = {}
        self.created_at = datetime.now()

    def get_service(self, service_type: type) -> Any | None:
        """Get scoped service instance"""
        return self.scoped_services.get(service_type)

    def set_service(self, service_type: type, instance: Any) -> None:
        """Set scoped service instance"""
        self.scoped_services[service_type] = instance

    def dispose(self) -> None:
        """Dispose all scoped services"""
        for service in self.scoped_services.values():
            if hasattr(service, "stop") and callable(service.stop):
                try:
                    service.stop()
                except Exception as e:
                    logger.error(f"Error stopping scoped service: {str(e)}")

        self.scoped_services.clear()


class IDependencyResolver(ABC):
    """Interface for dependency resolution"""

    @abstractmethod
    def resolve(self, service_type: type[T]) -> T:
        """Resolve service instance"""
        pass

    @abstractmethod
    def resolve_all(self, service_type: type[T]) -> list[T]:
        """Resolve all instances of service type"""
        pass


class ServiceContainer(IDependencyResolver):
    """
    Dependency injection container for GPT-Trader components

    Provides automated service registration, dependency resolution, and
    lifecycle management for all system components.
    """

    def __init__(self) -> None:
        """Initialize service container"""
        self.services: dict[type, ServiceDescriptor] = {}
        self.service_instances: dict[type, Any] = {}
        self.service_factories: dict[type, Callable] = {}

        # Scoped service management
        self.current_scope: ServiceScope | None = None
        self.scoped_services: dict[str, ServiceScope] = {}

        # Lifecycle management
        self.container_lock = threading.RLock()
        self.is_building = False
        self.build_order: list[type] = []

        # Health monitoring
        self.health_checks: dict[type, Callable[[], bool]] = {}
        self.last_health_check: datetime | None = None

        logger.info("Service container initialized")

    def register_singleton(
        self,
        service_type: type[T],
        implementation_type: type[T] | None = None,
        factory: Callable[[], T] | None = None,
        instance: T | None = None,
    ) -> "ServiceContainer":
        """Register singleton service"""
        return self._register_service(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON,
        )

    def register_transient(
        self,
        service_type: type[T],
        implementation_type: type[T] | None = None,
        factory: Callable[[], T] | None = None,
    ) -> "ServiceContainer":
        """Register transient service"""
        return self._register_service(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=ServiceLifetime.TRANSIENT,
        )

    def register_scoped(
        self,
        service_type: type[T],
        implementation_type: type[T] | None = None,
        factory: Callable[[], T] | None = None,
    ) -> "ServiceContainer":
        """Register scoped service"""
        return self._register_service(
            service_type=service_type,
            implementation_type=implementation_type,
            factory=factory,
            lifetime=ServiceLifetime.SCOPED,
        )

    def _register_service(
        self,
        service_type: type,
        implementation_type: type | None = None,
        factory: Callable | None = None,
        instance: Any | None = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> "ServiceContainer":
        """Register service with container"""

        with self.container_lock:
            if service_type in self.services:
                logger.warning(f"Service {service_type.__name__} already registered, replacing")

            # Determine implementation type if not provided
            if not implementation_type and not factory and not instance:
                implementation_type = service_type

            # Analyze dependencies
            dependencies = self._analyze_dependencies(implementation_type or service_type)

            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                instance=instance,
                lifetime=lifetime,
                dependencies=dependencies,
            )

            self.services[service_type] = descriptor

            # Update dependent relationships
            for dep_type in dependencies:
                if dep_type in self.services:
                    self.services[dep_type].dependents.add(service_type)

            logger.info(f"Registered {lifetime.value} service: {service_type.__name__}")

            return self

    def _analyze_dependencies(self, service_type: type) -> set[type]:
        """Analyze service dependencies from constructor"""
        dependencies = set()

        try:
            # Get constructor signature
            if hasattr(service_type, "__init__"):
                sig = inspect.signature(service_type.__init__)
                type_hints = get_type_hints(service_type.__init__)

                for param_name, _param in sig.parameters.items():
                    if param_name == "self":
                        continue

                    # Check if parameter has type annotation
                    if param_name in type_hints:
                        param_type = type_hints[param_name]

                        # Handle Optional types
                        if get_origin(param_type) is Union:
                            args = get_args(param_type)
                            if len(args) == 2 and type(None) in args:
                                # This is Optional[T]
                                param_type = next(arg for arg in args if arg != type(None))

                        # Only add if it's a class type (not primitive)
                        if (
                            inspect.isclass(param_type)
                            and param_type != str
                            and param_type != int
                            and param_type != float
                            and param_type != bool
                        ):
                            dependencies.add(param_type)

        except Exception as e:
            logger.warning(f"Could not analyze dependencies for {service_type.__name__}: {str(e)}")

        return dependencies

    def register_component(self, component: BaseComponent) -> "ServiceContainer":
        """Register an existing component instance"""
        return self.register_singleton(service_type=type(component), instance=component)

    def resolve(self, service_type: type[T]) -> T:
        """Resolve service instance"""
        with self.container_lock:
            try:
                return self._resolve_service(service_type)
            except Exception as e:
                raise ComponentException(
                    f"Failed to resolve service {service_type.__name__}: {str(e)}",
                    component="service_container",
                    context={"service_type": service_type.__name__},
                )

    def _resolve_service(self, service_type: type[T]) -> T:
        """Internal service resolution logic"""

        # Check if service is registered
        if service_type not in self.services:
            raise ComponentException(
                f"Service {service_type.__name__} is not registered", component="service_container"
            )

        descriptor = self.services[service_type]

        # Handle different lifetimes
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            return self._resolve_singleton(service_type, descriptor)
        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            return self._resolve_scoped(service_type, descriptor)
        else:  # TRANSIENT
            return self._resolve_transient(service_type, descriptor)

    def _resolve_singleton(self, service_type: type[T], descriptor: ServiceDescriptor) -> T:
        """Resolve singleton service"""

        # Return existing instance if available
        if service_type in self.service_instances:
            return self.service_instances[service_type]

        # Create new instance
        instance = self._create_service_instance(descriptor)
        self.service_instances[service_type] = instance

        return instance

    def _resolve_scoped(self, service_type: type[T], descriptor: ServiceDescriptor) -> T:
        """Resolve scoped service"""

        if not self.current_scope:
            raise ComponentException(
                "No active scope for scoped service resolution", component="service_container"
            )

        # Check if instance exists in current scope
        instance = self.current_scope.get_service(service_type)
        if instance:
            return instance

        # Create new scoped instance
        instance = self._create_service_instance(descriptor)
        self.current_scope.set_service(service_type, instance)

        return instance

    def _resolve_transient(self, service_type: type[T], descriptor: ServiceDescriptor) -> T:
        """Resolve transient service (always creates new instance)"""
        return self._create_service_instance(descriptor)

    def _create_service_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create service instance using factory or constructor"""

        descriptor.status = ServiceStatus.INITIALIZING
        descriptor.initialization_time = datetime.now()

        try:
            # Use existing instance if provided
            if descriptor.instance is not None:
                return descriptor.instance

            # Use factory if provided
            if descriptor.factory:
                return descriptor.factory()

            # Use constructor with dependency injection
            if descriptor.implementation_type:
                return self._create_with_dependencies(
                    descriptor.implementation_type, descriptor.dependencies
                )

            raise ComponentException(
                f"No way to create instance for service {descriptor.service_type.__name__}",
                component="service_container",
            )

        except Exception as e:
            descriptor.status = ServiceStatus.FAILED
            logger.error(f"Failed to create service {descriptor.service_type.__name__}: {str(e)}")
            raise

        finally:
            if descriptor.status != ServiceStatus.FAILED:
                descriptor.status = ServiceStatus.INITIALIZED

    def _create_with_dependencies(self, implementation_type: type, dependencies: set[type]) -> Any:
        """Create instance with automatic dependency injection"""

        # Detect circular dependencies
        if self.is_building:
            if implementation_type in self.build_order:
                cycle = self.build_order[self.build_order.index(implementation_type) :]
                cycle.append(implementation_type)
                cycle_names = [t.__name__ for t in cycle]
                raise ComponentException(
                    f"Circular dependency detected: {' -> '.join(cycle_names)}",
                    component="service_container",
                )

        self.is_building = True
        self.build_order.append(implementation_type)

        try:
            # Resolve constructor parameters
            constructor_params = {}

            if hasattr(implementation_type, "__init__"):
                sig = inspect.signature(implementation_type.__init__)
                type_hints = get_type_hints(implementation_type.__init__)

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue

                    # Check if parameter has type annotation
                    if param_name in type_hints:
                        param_type = type_hints[param_name]

                        # Handle Optional types
                        is_optional = False
                        if get_origin(param_type) is Union:
                            args = get_args(param_type)
                            if len(args) == 2 and type(None) in args:
                                param_type = next(arg for arg in args if arg != type(None))
                                is_optional = True

                        # Try to resolve dependency
                        if param_type in dependencies:
                            try:
                                constructor_params[param_name] = self._resolve_service(param_type)
                            except Exception:
                                if not is_optional and param.default == inspect.Parameter.empty:
                                    raise ComponentException(
                                        f"Failed to resolve required dependency {param_type.__name__} for {implementation_type.__name__}",
                                        component="service_container",
                                        context={
                                            "dependency": param_type.__name__,
                                            "service": implementation_type.__name__,
                                        },
                                    )
                                logger.warning(
                                    f"Optional dependency {param_type.__name__} not available for {implementation_type.__name__}"
                                )

            # Create instance
            return implementation_type(**constructor_params)

        finally:
            self.build_order.pop()
            if not self.build_order:
                self.is_building = False

    def resolve_all(self, service_type: type[T]) -> list[T]:
        """Resolve all instances of service type (for multiple implementations)"""
        instances = []

        for registered_type, _descriptor in self.services.items():
            # Check if registered type implements or inherits from service_type
            if registered_type == service_type or (
                inspect.isclass(registered_type) and issubclass(registered_type, service_type)
            ):
                try:
                    instance = self._resolve_service(registered_type)
                    instances.append(instance)
                except Exception as e:
                    logger.error(f"Failed to resolve {registered_type.__name__}: {str(e)}")

        return instances

    def create_scope(self, scope_id: str = None) -> ServiceScope:
        """Create new service scope"""
        if scope_id is None:
            scope_id = f"scope_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        scope = ServiceScope(scope_id)
        self.scoped_services[scope_id] = scope

        return scope

    def set_scope(self, scope: ServiceScope) -> None:
        """Set current service scope"""
        self.current_scope = scope

    def clear_scope(self) -> None:
        """Clear current service scope"""
        if self.current_scope:
            self.current_scope.dispose()
            self.current_scope = None

    def start_all_services(self) -> None:
        """Start all registered services that support starting"""
        logger.info("Starting all registered services...")

        started_count = 0
        failed_count = 0

        # Start services in dependency order
        start_order = self._get_startup_order()

        for service_type in start_order:
            descriptor = self.services[service_type]

            if not descriptor.auto_start:
                continue

            try:
                # Resolve service (creates instance if needed)
                instance = self.resolve(service_type)

                # Start service if it supports starting
                if isinstance(instance, BaseComponent):
                    if instance.status not in [ComponentStatus.RUNNING]:
                        instance.start()
                        descriptor.status = ServiceStatus.RUNNING
                        started_count += 1
                        logger.info(f"Started service: {service_type.__name__}")
                elif hasattr(instance, "start") and callable(instance.start):
                    instance.start()
                    descriptor.status = ServiceStatus.RUNNING
                    started_count += 1
                    logger.info(f"Started service: {service_type.__name__}")

            except Exception as e:
                descriptor.status = ServiceStatus.FAILED
                failed_count += 1
                logger.error(f"Failed to start service {service_type.__name__}: {str(e)}")

        logger.info(f"Service startup complete: {started_count} started, {failed_count} failed")

    def stop_all_services(self) -> None:
        """Stop all running services"""
        logger.info("Stopping all services...")

        stopped_count = 0

        # Stop services in reverse dependency order
        stop_order = list(reversed(self._get_startup_order()))

        for service_type in stop_order:
            descriptor = self.services[service_type]

            if descriptor.status != ServiceStatus.RUNNING:
                continue

            try:
                if service_type in self.service_instances:
                    instance = self.service_instances[service_type]

                    # Stop service if it supports stopping
                    if isinstance(instance, BaseComponent):
                        instance.stop()
                        descriptor.status = ServiceStatus.STOPPED
                        stopped_count += 1
                        logger.info(f"Stopped service: {service_type.__name__}")
                    elif hasattr(instance, "stop") and callable(instance.stop):
                        instance.stop()
                        descriptor.status = ServiceStatus.STOPPED
                        stopped_count += 1
                        logger.info(f"Stopped service: {service_type.__name__}")

            except Exception as e:
                logger.error(f"Error stopping service {service_type.__name__}: {str(e)}")

        logger.info(f"Service shutdown complete: {stopped_count} services stopped")

    def _get_startup_order(self) -> list[type]:
        """Get service startup order based on dependencies"""
        ordered_services = []
        visited = set()
        visiting = set()

        def visit(service_type: type) -> None:
            if service_type in visiting:
                raise ComponentException(
                    f"Circular dependency detected during startup ordering involving {service_type.__name__}",
                    component="service_container",
                )

            if service_type in visited:
                return

            visiting.add(service_type)

            # Visit all dependencies first
            if service_type in self.services:
                for dep_type in self.services[service_type].dependencies:
                    if dep_type in self.services:
                        visit(dep_type)

            visiting.remove(service_type)
            visited.add(service_type)
            ordered_services.append(service_type)

        # Visit all services
        for service_type in self.services.keys():
            visit(service_type)

        return ordered_services

    def get_service_health(self) -> dict[str, Any]:
        """Get health status of all registered services"""
        health_status = {
            "container_status": "healthy",
            "total_services": len(self.services),
            "running_services": 0,
            "failed_services": 0,
            "service_details": {},
        }

        for service_type, descriptor in self.services.items():
            service_name = service_type.__name__

            service_health = {
                "status": descriptor.status.value,
                "lifetime": descriptor.lifetime.value,
                "dependencies": [dep.__name__ for dep in descriptor.dependencies],
                "dependents": [dep.__name__ for dep in descriptor.dependents],
            }

            # Get component health if available
            if service_type in self.service_instances:
                instance = self.service_instances[service_type]
                if isinstance(instance, BaseComponent):
                    service_health["component_status"] = instance.status.value
                    service_health["health_status"] = instance.get_health_status().value
                    service_health["uptime"] = instance.metrics.uptime.total_seconds()

            health_status["service_details"][service_name] = service_health

            if descriptor.status == ServiceStatus.RUNNING:
                health_status["running_services"] += 1
            elif descriptor.status == ServiceStatus.FAILED:
                health_status["failed_services"] += 1

        # Update container health based on service health
        if health_status["failed_services"] > 0:
            health_status["container_status"] = (
                "degraded" if health_status["running_services"] > 0 else "unhealthy"
            )

        self.last_health_check = datetime.now()
        return health_status

    def dispose(self) -> None:
        """Dispose container and cleanup all services"""
        logger.info("Disposing service container...")

        # Stop all services
        self.stop_all_services()

        # Clear all scopes
        for scope in self.scoped_services.values():
            scope.dispose()
        self.scoped_services.clear()

        # Clear service instances
        self.service_instances.clear()
        self.services.clear()

        logger.info("Service container disposed")


# Global service container instance
_service_container: ServiceContainer | None = None
_container_lock = threading.Lock()


def get_container() -> ServiceContainer:
    """Get global service container instance"""
    global _service_container

    with _container_lock:
        if _service_container is None:
            _service_container = ServiceContainer()
            logger.info("Global service container created")

        return _service_container


def configure_services(configurator: Callable[[ServiceContainer], None]) -> None:
    """Configure services using a configurator function"""
    container = get_container()
    configurator(container)
    logger.info("Service configuration completed")


# Decorators for simplified service registration


def injectable(cls):
    """Decorator to mark class as injectable service"""
    # Register the class automatically with the container
    container = get_container()
    container.register_singleton(cls)
    return cls


def component(lifetime: ServiceLifetime = ServiceLifetime.SINGLETON):
    """Decorator to register component with specific lifetime"""

    def decorator(cls):
        container = get_container()

        if lifetime == ServiceLifetime.SINGLETON:
            container.register_singleton(cls)
        elif lifetime == ServiceLifetime.TRANSIENT:
            container.register_transient(cls)
        elif lifetime == ServiceLifetime.SCOPED:
            container.register_scoped(cls)

        return cls

    return decorator
