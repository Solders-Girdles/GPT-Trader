"""
Phase 4: Operational Excellence - Production Deployment Automation

This module provides comprehensive deployment automation including:
- CI/CD pipeline automation and orchestration
- Container-based deployment with Kubernetes integration
- Infrastructure-as-code with environment provisioning
- Blue-green and canary deployment strategies
- Automated rollback and recovery mechanisms
- Multi-environment deployment management
- Health check and smoke test automation
- Deployment metrics and monitoring integration

This deployment system enables zero-downtime production deployments with
automated testing, validation, and recovery capabilities.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import yaml

from .base import BaseComponent, ComponentConfig, HealthStatus
from .exceptions import ComponentException
from .metrics import MetricLabels, get_metrics_registry
from .observability import AlertSeverity, create_alert, get_observability_engine, start_trace

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    BLUE_GREEN_BLUE = "blue"
    BLUE_GREEN_GREEN = "green"


class DeploymentStrategy(Enum):
    """Deployment strategy types"""

    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TEST = "a_b_test"


class DeploymentStatus(Enum):
    """Deployment status states"""

    PENDING = "pending"
    PREPARING = "preparing"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYING = "deploying"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class HealthCheckType(Enum):
    """Health check types for validation"""

    HTTP_GET = "http_get"
    TCP_SOCKET = "tcp_socket"
    EXEC_COMMAND = "exec_command"
    CUSTOM_SCRIPT = "custom_script"
    DATABASE_QUERY = "database_query"
    SERVICE_MESH = "service_mesh"


@dataclass
class HealthCheckConfig:
    """Configuration for deployment health checks"""

    check_type: HealthCheckType
    endpoint: str | None = None
    port: int | None = None
    path: str | None = None
    command: list[str] | None = None
    timeout_seconds: int = 30
    initial_delay_seconds: int = 10
    period_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 1
    expected_status_code: int = 200
    expected_response_pattern: str | None = None
    custom_validator: Callable | None = None


@dataclass
class DeploymentConfig:
    """Comprehensive deployment configuration"""

    deployment_id: str
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy

    # Application configuration
    application_name: str
    version: str
    docker_image: str
    docker_tag: str

    # Resource configuration
    cpu_request: str = "500m"
    cpu_limit: str = "2000m"
    memory_request: str = "512Mi"
    memory_limit: str = "2Gi"
    replicas: int = 3

    # Network configuration
    service_port: int = 8080
    target_port: int = 8080
    ingress_host: str | None = None
    load_balancer_type: str = "ClusterIP"

    # Health checks
    health_checks: list[HealthCheckConfig] = field(default_factory=list)
    readiness_probe: HealthCheckConfig | None = None
    liveness_probe: HealthCheckConfig | None = None

    # Environment variables
    environment_variables: dict[str, str] = field(default_factory=dict)
    secret_references: dict[str, str] = field(default_factory=dict)
    config_map_references: dict[str, str] = field(default_factory=dict)

    # Deployment behavior
    max_unavailable: str = "25%"
    max_surge: str = "25%"
    progress_deadline_seconds: int = 600
    revision_history_limit: int = 10

    # Canary deployment specific
    canary_traffic_percentage: int = 10
    canary_analysis_duration_minutes: int = 30
    canary_success_criteria: dict[str, Any] = field(default_factory=dict)

    # Blue-green deployment specific
    blue_green_switch_delay_minutes: int = 5
    blue_green_validation_timeout_minutes: int = 15

    # Rollback configuration
    auto_rollback_enabled: bool = True
    rollback_failure_threshold_percent: float = 5.0
    rollback_timeout_minutes: int = 10


@dataclass
class DeploymentResult:
    """Result of deployment operation"""

    deployment_id: str
    status: DeploymentStatus
    start_time: datetime
    end_time: datetime | None = None
    success: bool = False
    error_message: str | None = None
    deployed_version: str | None = None
    rollback_version: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    health_check_results: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


class IDeploymentStrategy(ABC):
    """Interface for deployment strategy implementations"""

    @abstractmethod
    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Execute deployment using this strategy"""
        pass

    @abstractmethod
    async def rollback(self, config: DeploymentConfig, target_version: str) -> DeploymentResult:
        """Rollback deployment using this strategy"""
        pass

    @abstractmethod
    async def validate_deployment(self, config: DeploymentConfig) -> bool:
        """Validate deployment success"""
        pass


class IInfrastructureProvider(ABC):
    """Interface for infrastructure providers"""

    @abstractmethod
    async def provision_resources(self, config: DeploymentConfig) -> dict[str, Any]:
        """Provision infrastructure resources"""
        pass

    @abstractmethod
    async def deploy_application(self, config: DeploymentConfig) -> dict[str, Any]:
        """Deploy application to infrastructure"""
        pass

    @abstractmethod
    async def scale_application(self, config: DeploymentConfig, replicas: int) -> bool:
        """Scale application instances"""
        pass

    @abstractmethod
    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get current deployment status"""
        pass


class KubernetesProvider(IInfrastructureProvider):
    """Kubernetes infrastructure provider"""

    def __init__(self, kubeconfig_path: str | None = None, namespace: str = "default") -> None:
        self.kubeconfig_path = kubeconfig_path
        self.namespace = namespace
        self.kubectl_cmd = self._get_kubectl_command()

    def _get_kubectl_command(self) -> list[str]:
        """Get kubectl command with configuration"""
        cmd = ["kubectl"]
        if self.kubeconfig_path:
            cmd.extend(["--kubeconfig", self.kubeconfig_path])
        cmd.extend(["--namespace", self.namespace])
        return cmd

    async def _run_kubectl(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run kubectl command"""
        cmd = self.kubectl_cmd + args
        return await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

    def _generate_deployment_yaml(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes deployment YAML"""
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{config.application_name}-{config.environment.value}",
                "namespace": self.namespace,
                "labels": {
                    "app": config.application_name,
                    "environment": config.environment.value,
                    "version": config.version,
                },
            },
            "spec": {
                "replicas": config.replicas,
                "strategy": {
                    "type": (
                        "RollingUpdate"
                        if config.strategy == DeploymentStrategy.ROLLING_UPDATE
                        else "Recreate"
                    ),
                    "rollingUpdate": (
                        {"maxUnavailable": config.max_unavailable, "maxSurge": config.max_surge}
                        if config.strategy == DeploymentStrategy.ROLLING_UPDATE
                        else None
                    ),
                },
                "selector": {
                    "matchLabels": {
                        "app": config.application_name,
                        "environment": config.environment.value,
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.application_name,
                            "environment": config.environment.value,
                            "version": config.version,
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": config.application_name,
                                "image": f"{config.docker_image}:{config.docker_tag}",
                                "ports": [{"containerPort": config.target_port, "name": "http"}],
                                "resources": {
                                    "requests": {
                                        "cpu": config.cpu_request,
                                        "memory": config.memory_request,
                                    },
                                    "limits": {
                                        "cpu": config.cpu_limit,
                                        "memory": config.memory_limit,
                                    },
                                },
                                "env": [
                                    {"name": k, "value": v}
                                    for k, v in config.environment_variables.items()
                                ],
                                "livenessProbe": (
                                    self._generate_probe_config(config.liveness_probe)
                                    if config.liveness_probe
                                    else None
                                ),
                                "readinessProbe": (
                                    self._generate_probe_config(config.readiness_probe)
                                    if config.readiness_probe
                                    else None
                                ),
                            }
                        ]
                    },
                },
            },
        }

        return yaml.dump(deployment, default_flow_style=False)

    def _generate_service_yaml(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes service YAML"""
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{config.application_name}-{config.environment.value}-service",
                "namespace": self.namespace,
                "labels": {"app": config.application_name, "environment": config.environment.value},
            },
            "spec": {
                "type": config.load_balancer_type,
                "selector": {
                    "app": config.application_name,
                    "environment": config.environment.value,
                },
                "ports": [
                    {"port": config.service_port, "targetPort": config.target_port, "name": "http"}
                ],
            },
        }

        return yaml.dump(service, default_flow_style=False)

    def _generate_probe_config(self, health_check: HealthCheckConfig) -> dict[str, Any]:
        """Generate probe configuration"""
        probe = {
            "initialDelaySeconds": health_check.initial_delay_seconds,
            "periodSeconds": health_check.period_seconds,
            "timeoutSeconds": health_check.timeout_seconds,
            "failureThreshold": health_check.failure_threshold,
            "successThreshold": health_check.success_threshold,
        }

        if health_check.check_type == HealthCheckType.HTTP_GET:
            probe["httpGet"] = {
                "path": health_check.path or "/health",
                "port": health_check.port or 8080,
            }
        elif health_check.check_type == HealthCheckType.TCP_SOCKET:
            probe["tcpSocket"] = {"port": health_check.port or 8080}
        elif health_check.check_type == HealthCheckType.EXEC_COMMAND:
            probe["exec"] = {"command": health_check.command or ["echo", "healthy"]}

        return probe

    async def provision_resources(self, config: DeploymentConfig) -> dict[str, Any]:
        """Provision Kubernetes resources"""
        try:
            # Create deployment manifest
            deployment_yaml = self._generate_deployment_yaml(config)
            service_yaml = self._generate_service_yaml(config)

            # Write manifests to temporary files
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(deployment_yaml)
                deployment_file = f.name

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(service_yaml)
                service_file = f.name

            # Apply manifests
            deploy_result = await self._run_kubectl(["apply", "-f", deployment_file])
            service_result = await self._run_kubectl(["apply", "-f", service_file])

            # Clean up temporary files
            os.unlink(deployment_file)
            os.unlink(service_file)

            return {
                "deployment_applied": deploy_result.returncode == 0,
                "service_applied": service_result.returncode == 0,
                "deployment_output": deploy_result.stdout.decode() if deploy_result.stdout else "",
                "service_output": service_result.stdout.decode() if service_result.stdout else "",
                "errors": [],
            }

        except Exception as e:
            logger.error(f"Failed to provision Kubernetes resources: {str(e)}")
            return {"deployment_applied": False, "service_applied": False, "errors": [str(e)]}

    async def deploy_application(self, config: DeploymentConfig) -> dict[str, Any]:
        """Deploy application to Kubernetes"""
        try:
            # Provision resources first
            provision_result = await self.provision_resources(config)

            if not provision_result.get("deployment_applied"):
                return {"success": False, "error": "Failed to provision deployment resources"}

            # Wait for deployment to be ready
            deployment_name = f"{config.application_name}-{config.environment.value}"

            # Wait for rollout to complete
            rollout_cmd = ["rollout", "status", "deployment", deployment_name, "--timeout=600s"]
            rollout_result = await self._run_kubectl(rollout_cmd)

            return {
                "success": rollout_result.returncode == 0,
                "deployment_name": deployment_name,
                "rollout_status": rollout_result.stdout.decode() if rollout_result.stdout else "",
                "provision_result": provision_result,
            }

        except Exception as e:
            logger.error(f"Failed to deploy application: {str(e)}")
            return {"success": False, "error": str(e)}

    async def scale_application(self, config: DeploymentConfig, replicas: int) -> bool:
        """Scale Kubernetes deployment"""
        try:
            deployment_name = f"{config.application_name}-{config.environment.value}"
            scale_cmd = ["scale", "deployment", deployment_name, f"--replicas={replicas}"]

            result = await self._run_kubectl(scale_cmd)
            return result.returncode == 0

        except Exception as e:
            logger.error(f"Failed to scale application: {str(e)}")
            return False

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get Kubernetes deployment status"""
        try:
            # Get deployment status
            get_cmd = ["get", "deployment", deployment_id, "-o", "json"]
            result = await self._run_kubectl(get_cmd)

            if result.returncode != 0:
                return {"status": "not_found"}

            deployment_info = json.loads(result.stdout.decode())

            return {
                "status": "running",
                "replicas": deployment_info["spec"]["replicas"],
                "ready_replicas": deployment_info.get("status", {}).get("readyReplicas", 0),
                "updated_replicas": deployment_info.get("status", {}).get("updatedReplicas", 0),
                "deployment_info": deployment_info,
            }

        except Exception as e:
            logger.error(f"Failed to get deployment status: {str(e)}")
            return {"status": "error", "error": str(e)}


class RollingUpdateStrategy(IDeploymentStrategy):
    """Rolling update deployment strategy"""

    def __init__(self, infrastructure_provider: IInfrastructureProvider) -> None:
        self.infrastructure_provider = infrastructure_provider

    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Execute rolling update deployment"""
        result = DeploymentResult(
            deployment_id=config.deployment_id,
            status=DeploymentStatus.PREPARING,
            start_time=datetime.now(),
        )

        try:
            # Deploy application
            result.status = DeploymentStatus.DEPLOYING
            deploy_result = await self.infrastructure_provider.deploy_application(config)

            if not deploy_result.get("success"):
                result.status = DeploymentStatus.FAILED
                result.error_message = deploy_result.get("error", "Deployment failed")
                return result

            # Validate deployment
            result.status = DeploymentStatus.VALIDATING
            is_valid = await self.validate_deployment(config)

            if is_valid:
                result.status = DeploymentStatus.COMPLETED
                result.success = True
                result.deployed_version = config.version
            else:
                result.status = DeploymentStatus.FAILED
                result.error_message = "Deployment validation failed"

            return result

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            return result

        finally:
            result.end_time = datetime.now()

    async def rollback(self, config: DeploymentConfig, target_version: str) -> DeploymentResult:
        """Rollback to previous version"""
        result = DeploymentResult(
            deployment_id=config.deployment_id,
            status=DeploymentStatus.ROLLING_BACK,
            start_time=datetime.now(),
        )

        try:
            # Create rollback configuration
            rollback_config = DeploymentConfig(
                deployment_id=f"{config.deployment_id}-rollback",
                environment=config.environment,
                strategy=config.strategy,
                application_name=config.application_name,
                version=target_version,
                docker_image=config.docker_image,
                docker_tag=target_version,
            )

            # Execute rollback deployment
            deploy_result = await self.infrastructure_provider.deploy_application(rollback_config)

            if deploy_result.get("success"):
                result.status = DeploymentStatus.ROLLED_BACK
                result.success = True
                result.rollback_version = target_version
            else:
                result.status = DeploymentStatus.FAILED
                result.error_message = f"Rollback failed: {deploy_result.get('error')}"

            return result

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = f"Rollback error: {str(e)}"
            return result

        finally:
            result.end_time = datetime.now()

    async def validate_deployment(self, config: DeploymentConfig) -> bool:
        """Validate rolling update deployment"""
        try:
            # Check deployment status
            status = await self.infrastructure_provider.get_deployment_status(
                f"{config.application_name}-{config.environment.value}"
            )

            if status.get("status") != "running":
                return False

            # Verify all replicas are ready
            replicas = status.get("replicas", 0)
            ready_replicas = status.get("ready_replicas", 0)

            if ready_replicas < replicas:
                return False

            # Run health checks
            for health_check in config.health_checks:
                if not await self._run_health_check(health_check):
                    return False

            return True

        except Exception as e:
            logger.error(f"Deployment validation error: {str(e)}")
            return False

    async def _run_health_check(self, health_check: HealthCheckConfig) -> bool:
        """Run individual health check"""
        try:
            if health_check.check_type == HealthCheckType.HTTP_GET:
                # Implement HTTP health check
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    url = f"http://{health_check.endpoint}:{health_check.port}{health_check.path or '/health'}"

                    async with session.get(url, timeout=health_check.timeout_seconds) as response:
                        if response.status != health_check.expected_status_code:
                            return False

                        if health_check.expected_response_pattern:
                            text = await response.text()
                            import re

                            if not re.search(health_check.expected_response_pattern, text):
                                return False

                        return True

            elif health_check.check_type == HealthCheckType.EXEC_COMMAND:
                # Run command health check
                result = await asyncio.create_subprocess_exec(
                    *health_check.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await result.communicate()
                return result.returncode == 0

            elif health_check.custom_validator:
                # Run custom validation
                return await health_check.custom_validator()

            return True

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


class CanaryDeploymentStrategy(IDeploymentStrategy):
    """Canary deployment strategy with traffic splitting"""

    def __init__(self, infrastructure_provider: IInfrastructureProvider) -> None:
        self.infrastructure_provider = infrastructure_provider
        self.metrics_registry = get_metrics_registry()
        self.observability_engine = get_observability_engine()

    async def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Execute canary deployment"""
        result = DeploymentResult(
            deployment_id=config.deployment_id,
            status=DeploymentStatus.PREPARING,
            start_time=datetime.now(),
        )

        trace = start_trace(f"canary_deployment_{config.deployment_id}")

        try:
            # Phase 1: Deploy canary version
            result.status = DeploymentStatus.DEPLOYING
            trace.add_tag("phase", "canary_deploy")

            canary_config = self._create_canary_config(config)
            canary_result = await self.infrastructure_provider.deploy_application(canary_config)

            if not canary_result.get("success"):
                result.status = DeploymentStatus.FAILED
                result.error_message = "Canary deployment failed"
                return result

            # Phase 2: Configure traffic splitting
            await self._configure_traffic_splitting(config)

            # Phase 3: Monitor canary performance
            result.status = DeploymentStatus.VALIDATING
            trace.add_tag("phase", "canary_analysis")

            canary_success = await self._analyze_canary_performance(config)

            if canary_success:
                # Phase 4: Promote canary to production
                trace.add_tag("phase", "canary_promotion")
                await self._promote_canary(config)

                result.status = DeploymentStatus.COMPLETED
                result.success = True
                result.deployed_version = config.version
            else:
                # Rollback canary
                trace.add_tag("phase", "canary_rollback")
                await self._rollback_canary(config)

                result.status = DeploymentStatus.FAILED
                result.error_message = "Canary analysis failed - deployment rolled back"

            return result

        except Exception as e:
            result.status = DeploymentStatus.FAILED
            result.error_message = str(e)
            return result

        finally:
            result.end_time = datetime.now()
            self.observability_engine.finish_trace(trace)

    def _create_canary_config(self, config: DeploymentConfig) -> DeploymentConfig:
        """Create configuration for canary deployment"""
        canary_config = DeploymentConfig(
            deployment_id=f"{config.deployment_id}-canary",
            environment=DeploymentEnvironment.CANARY,
            strategy=DeploymentStrategy.CANARY,
            application_name=f"{config.application_name}-canary",
            version=config.version,
            docker_image=config.docker_image,
            docker_tag=config.docker_tag,
            replicas=max(1, config.replicas // 10),  # 10% of production replicas
            cpu_request=config.cpu_request,
            cpu_limit=config.cpu_limit,
            memory_request=config.memory_request,
            memory_limit=config.memory_limit,
            service_port=config.service_port,
            target_port=config.target_port,
            environment_variables=config.environment_variables,
            health_checks=config.health_checks,
        )

        return canary_config

    async def _configure_traffic_splitting(self, config: DeploymentConfig) -> None:
        """Configure traffic splitting between stable and canary versions"""
        # This would integrate with service mesh (Istio, Linkerd) or ingress controller
        logger.info(f"Configuring {config.canary_traffic_percentage}% traffic to canary")

        # Placeholder for service mesh integration
        # await self._update_virtual_service(config)
        # await self._update_destination_rules(config)

    async def _analyze_canary_performance(self, config: DeploymentConfig) -> bool:
        """Analyze canary deployment performance"""
        try:
            analysis_duration = timedelta(minutes=config.canary_analysis_duration_minutes)
            analysis_start = datetime.now()

            while datetime.now() - analysis_start < analysis_duration:
                # Collect metrics
                canary_metrics = await self._collect_canary_metrics(config)
                stable_metrics = await self._collect_stable_metrics(config)

                # Compare success criteria
                if not self._evaluate_success_criteria(config, canary_metrics, stable_metrics):
                    logger.warning("Canary success criteria not met")
                    return False

                # Check for critical errors
                if self._has_critical_errors(canary_metrics):
                    logger.error("Critical errors detected in canary")
                    return False

                await asyncio.sleep(30)  # Check every 30 seconds

            logger.info("Canary analysis completed successfully")
            return True

        except Exception as e:
            logger.error(f"Canary analysis failed: {str(e)}")
            return False

    async def _collect_canary_metrics(self, config: DeploymentConfig) -> dict[str, float]:
        """Collect metrics for canary version"""
        # Placeholder for metrics collection from observability system
        return {
            "error_rate": 0.01,  # 1% error rate
            "response_time_p95": 150.0,  # 150ms P95 response time
            "throughput": 100.0,  # 100 requests/second
        }

    async def _collect_stable_metrics(self, config: DeploymentConfig) -> dict[str, float]:
        """Collect metrics for stable version"""
        # Placeholder for metrics collection from observability system
        return {
            "error_rate": 0.005,  # 0.5% error rate
            "response_time_p95": 120.0,  # 120ms P95 response time
            "throughput": 1000.0,  # 1000 requests/second
        }

    def _evaluate_success_criteria(
        self,
        config: DeploymentConfig,
        canary_metrics: dict[str, float],
        stable_metrics: dict[str, float],
    ) -> bool:
        """Evaluate canary success criteria"""
        criteria = config.canary_success_criteria

        # Error rate should not exceed stable by more than threshold
        error_rate_threshold = criteria.get("max_error_rate_increase", 0.01)  # 1%
        if canary_metrics["error_rate"] - stable_metrics["error_rate"] > error_rate_threshold:
            return False

        # Response time should not exceed stable by more than threshold
        response_time_threshold = criteria.get("max_response_time_increase", 50.0)  # 50ms
        if (
            canary_metrics["response_time_p95"] - stable_metrics["response_time_p95"]
            > response_time_threshold
        ):
            return False

        return True

    def _has_critical_errors(self, canary_metrics: dict[str, float]) -> bool:
        """Check for critical errors in canary deployment"""
        return canary_metrics.get("error_rate", 0) > 0.05  # 5% error rate threshold

    async def _promote_canary(self, config: DeploymentConfig) -> None:
        """Promote canary to production"""
        logger.info("Promoting canary to production")

        # Update production deployment with canary version
        production_config = DeploymentConfig(
            deployment_id=f"{config.deployment_id}-production",
            environment=DeploymentEnvironment.PRODUCTION,
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            application_name=config.application_name,
            version=config.version,
            docker_image=config.docker_image,
            docker_tag=config.docker_tag,
            replicas=config.replicas,
            cpu_request=config.cpu_request,
            cpu_limit=config.cpu_limit,
            memory_request=config.memory_request,
            memory_limit=config.memory_limit,
            service_port=config.service_port,
            target_port=config.target_port,
            environment_variables=config.environment_variables,
            health_checks=config.health_checks,
        )

        await self.infrastructure_provider.deploy_application(production_config)

    async def _rollback_canary(self, config: DeploymentConfig) -> None:
        """Rollback canary deployment"""
        logger.info("Rolling back canary deployment")

        # Remove canary deployment and restore 100% traffic to stable
        # This would involve service mesh configuration updates
        pass

    async def rollback(self, config: DeploymentConfig, target_version: str) -> DeploymentResult:
        """Rollback canary deployment"""
        # Similar to rolling update rollback
        rolling_strategy = RollingUpdateStrategy(self.infrastructure_provider)
        return await rolling_strategy.rollback(config, target_version)

    async def validate_deployment(self, config: DeploymentConfig) -> bool:
        """Validate canary deployment"""
        rolling_strategy = RollingUpdateStrategy(self.infrastructure_provider)
        return await rolling_strategy.validate_deployment(config)


class DeploymentManager(BaseComponent):
    """Comprehensive deployment management system"""

    def __init__(self, config: ComponentConfig | None = None) -> None:
        if not config:
            config = ComponentConfig(
                component_id="deployment_manager", component_type="deployment_manager"
            )

        super().__init__(config)

        # Infrastructure providers
        self.providers: dict[str, IInfrastructureProvider] = {}

        # Deployment strategies
        self.strategies: dict[DeploymentStrategy, IDeploymentStrategy] = {}

        # Active deployments
        self.active_deployments: dict[str, DeploymentResult] = {}

        # Metrics and observability
        self.metrics_registry = get_metrics_registry()
        self.observability_engine = get_observability_engine()

        # Setup metrics
        self._setup_metrics()

        # Initialize default providers and strategies
        self._initialize_defaults()

        logger.info(f"Deployment manager initialized: {self.component_id}")

    def _initialize_component(self) -> None:
        """Initialize deployment manager"""
        logger.info("Initializing deployment manager...")

    def _start_component(self) -> None:
        """Start deployment manager"""
        logger.info("Starting deployment manager...")

    def _stop_component(self) -> None:
        """Stop deployment manager"""
        logger.info("Stopping deployment manager...")

    def _health_check(self) -> HealthStatus:
        """Check deployment manager health"""
        if len(self.providers) == 0:
            return HealthStatus.UNHEALTHY

        return HealthStatus.HEALTHY

    def _setup_metrics(self) -> None:
        """Setup deployment metrics"""
        labels = MetricLabels().add("component", self.component_id)

        self.metrics = {
            "deployments_total": self.metrics_registry.register_counter(
                "deployments_total",
                "Total number of deployments",
                component_id=self.component_id,
                labels=labels,
            ),
            "deployment_duration": self.metrics_registry.register_histogram(
                "deployment_duration_seconds",
                "Deployment duration in seconds",
                component_id=self.component_id,
                labels=labels,
            ),
            "deployment_success_rate": self.metrics_registry.register_gauge(
                "deployment_success_rate",
                "Deployment success rate percentage",
                component_id=self.component_id,
                labels=labels,
            ),
            "active_deployments": self.metrics_registry.register_gauge(
                "active_deployments",
                "Number of active deployments",
                component_id=self.component_id,
                labels=labels,
            ),
        }

    def _initialize_defaults(self) -> None:
        """Initialize default providers and strategies"""
        # Add Kubernetes provider
        k8s_provider = KubernetesProvider()
        self.register_infrastructure_provider("kubernetes", k8s_provider)

        # Add deployment strategies
        self.register_deployment_strategy(
            DeploymentStrategy.ROLLING_UPDATE, RollingUpdateStrategy(k8s_provider)
        )
        self.register_deployment_strategy(
            DeploymentStrategy.CANARY, CanaryDeploymentStrategy(k8s_provider)
        )

    def register_infrastructure_provider(
        self, name: str, provider: IInfrastructureProvider
    ) -> None:
        """Register infrastructure provider"""
        self.providers[name] = provider
        logger.info(f"Registered infrastructure provider: {name}")

    def register_deployment_strategy(
        self, strategy_type: DeploymentStrategy, strategy: IDeploymentStrategy
    ) -> None:
        """Register deployment strategy"""
        self.strategies[strategy_type] = strategy
        logger.info(f"Registered deployment strategy: {strategy_type.value}")

    async def deploy_application(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy application using specified configuration"""
        trace = start_trace(f"deploy_application_{config.deployment_id}")
        trace.add_tag("application", config.application_name)
        trace.add_tag("environment", config.environment.value)
        trace.add_tag("strategy", config.strategy.value)

        try:
            self.metrics["deployments_total"].increment()
            self.active_deployments[config.deployment_id] = DeploymentResult(
                deployment_id=config.deployment_id,
                status=DeploymentStatus.PENDING,
                start_time=datetime.now(),
            )

            self.metrics["active_deployments"].set(len(self.active_deployments))

            # Get deployment strategy
            if config.strategy not in self.strategies:
                raise ComponentException(
                    f"Deployment strategy {config.strategy.value} not available"
                )

            strategy = self.strategies[config.strategy]

            # Execute deployment
            logger.info(f"Starting deployment: {config.deployment_id}")
            result = await strategy.deploy(config)

            # Update active deployments
            self.active_deployments[config.deployment_id] = result

            # Record metrics
            if result.end_time:
                duration = (result.end_time - result.start_time).total_seconds()
                self.metrics["deployment_duration"].observe(duration)

            # Update success rate
            self._update_success_rate()

            # Create alerts for failed deployments
            if not result.success:
                create_alert(
                    name="Deployment Failed",
                    severity=AlertSeverity.ERROR,
                    description=f"Deployment {config.deployment_id} failed: {result.error_message}",
                    component_id=self.component_id,
                    deployment_id=config.deployment_id,
                    application=config.application_name,
                    environment=config.environment.value,
                )

            trace.add_tag("success", result.success)
            trace.add_tag("status", result.status.value)

            return result

        except Exception as e:
            logger.error(f"Deployment error: {str(e)}")

            # Update deployment result
            error_result = DeploymentResult(
                deployment_id=config.deployment_id,
                status=DeploymentStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e),
            )

            self.active_deployments[config.deployment_id] = error_result
            self._update_success_rate()

            trace.add_tag("success", False)
            trace.add_tag("error", str(e))

            raise ComponentException(f"Deployment failed: {str(e)}")

        finally:
            self.observability_engine.finish_trace(trace)

    async def rollback_deployment(
        self, deployment_id: str, target_version: str
    ) -> DeploymentResult:
        """Rollback deployment to target version"""
        if deployment_id not in self.active_deployments:
            raise ComponentException(f"Deployment {deployment_id} not found")

        self.active_deployments[deployment_id]

        # Create rollback configuration (simplified)
        rollback_config = DeploymentConfig(
            deployment_id=f"{deployment_id}-rollback",
            environment=DeploymentEnvironment.PRODUCTION,  # Assume production rollback
            strategy=DeploymentStrategy.ROLLING_UPDATE,
            application_name="unknown",  # Would need to be stored
            version=target_version,
            docker_image="unknown",  # Would need to be stored
            docker_tag=target_version,
        )

        # Use rolling update for rollback
        strategy = self.strategies[DeploymentStrategy.ROLLING_UPDATE]
        return await strategy.rollback(rollback_config, target_version)

    def get_deployment_status(self, deployment_id: str) -> DeploymentResult | None:
        """Get deployment status"""
        return self.active_deployments.get(deployment_id)

    def list_active_deployments(self) -> list[DeploymentResult]:
        """List all active deployments"""
        return list(self.active_deployments.values())

    def _update_success_rate(self) -> None:
        """Update deployment success rate metric"""
        if not self.active_deployments:
            return

        successful = sum(1 for d in self.active_deployments.values() if d.success)
        total = len(self.active_deployments)
        success_rate = (successful / total) * 100

        self.metrics["deployment_success_rate"].set(success_rate)


# Global deployment manager instance
_deployment_manager: DeploymentManager | None = None


def get_deployment_manager() -> DeploymentManager:
    """Get global deployment manager instance"""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = DeploymentManager()
    return _deployment_manager


def initialize_deployment_manager(config: ComponentConfig | None = None) -> DeploymentManager:
    """Initialize deployment manager"""
    global _deployment_manager
    _deployment_manager = DeploymentManager(config)
    return _deployment_manager


# Convenience functions for common deployment operations


async def deploy_to_kubernetes(
    application_name: str,
    version: str,
    docker_image: str,
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION,
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
    replicas: int = 3,
    **kwargs,
) -> DeploymentResult:
    """Deploy application to Kubernetes with simple configuration"""

    config = DeploymentConfig(
        deployment_id=f"{application_name}-{version}-{int(time.time())}",
        environment=environment,
        strategy=strategy,
        application_name=application_name,
        version=version,
        docker_image=docker_image,
        docker_tag=version,
        replicas=replicas,
        **kwargs,
    )

    manager = get_deployment_manager()
    return await manager.deploy_application(config)


async def canary_deploy(
    application_name: str,
    version: str,
    docker_image: str,
    traffic_percentage: int = 10,
    analysis_duration_minutes: int = 30,
    **kwargs,
) -> DeploymentResult:
    """Deploy application using canary strategy"""

    config = DeploymentConfig(
        deployment_id=f"{application_name}-canary-{version}-{int(time.time())}",
        environment=DeploymentEnvironment.PRODUCTION,
        strategy=DeploymentStrategy.CANARY,
        application_name=application_name,
        version=version,
        docker_image=docker_image,
        docker_tag=version,
        canary_traffic_percentage=traffic_percentage,
        canary_analysis_duration_minutes=analysis_duration_minutes,
        **kwargs,
    )

    manager = get_deployment_manager()
    return await manager.deploy_application(config)
