"""
Health check dependency planning.

This module keeps deterministic health-check ordering separate from concrete
health probes so planner changes can be reviewed and tested independently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from gpt_trader.monitoring.interfaces import HealthCheckDependency, HealthCheckResult

HealthCheckOutcome = HealthCheckResult
HealthCheckMode = Literal["blocking", "fast"]


class HealthCheckPlanError(RuntimeError):
    """Base error for health check planning failures."""


class HealthCheckDependencyError(HealthCheckPlanError):
    """Raised when required dependencies are missing."""

    def __init__(self, missing_dependencies: dict[str, tuple[str, ...]]) -> None:
        self.missing_dependencies = missing_dependencies
        missing_details = "; ".join(
            f"{check} requires {', '.join(dependencies)}"
            for check, dependencies in sorted(missing_dependencies.items())
        )
        super().__init__(f"Missing required health check dependencies: {missing_details}")


class HealthCheckCycleError(HealthCheckPlanError):
    """Raised when a dependency cycle is detected."""

    def __init__(self, cycle: tuple[str, ...]) -> None:
        self.cycle = cycle
        cycle_text = " -> ".join(cycle) if cycle else "unknown cycle"
        super().__init__(f"Health check dependency cycle detected: {cycle_text}")


@dataclass(frozen=True)
class HealthCheckDescriptor:
    """Descriptor for a registered health check."""

    name: str
    mode: HealthCheckMode
    run: Callable[[], HealthCheckOutcome]
    dependencies: tuple[HealthCheckDependency, ...] = ()


class HealthCheckPlanner:
    """Build deterministic execution order for health checks."""

    def __init__(self, checks: tuple[HealthCheckDescriptor, ...]) -> None:
        self._checks = checks
        self._checks_by_name = {check.name: check for check in checks}

    def build_order(self) -> tuple[HealthCheckDescriptor, ...]:
        """Resolve dependencies and return ordered health checks."""
        if not self._checks_by_name:
            return ()

        dependency_graph, missing_required = self._build_dependency_graph()
        if missing_required:
            raise HealthCheckDependencyError(
                {name: tuple(sorted(missing)) for name, missing in sorted(missing_required.items())}
            )

        ordered_names = self._topological_sort(dependency_graph)
        return tuple(self._checks_by_name[name] for name in ordered_names)

    def _build_dependency_graph(
        self,
    ) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
        dependency_graph: dict[str, set[str]] = {name: set() for name in self._checks_by_name}
        missing_required: dict[str, list[str]] = {}

        for check in self._checks_by_name.values():
            for dependency in check.dependencies:
                if dependency.name in self._checks_by_name:
                    dependency_graph[check.name].add(dependency.name)
                elif dependency.required:
                    missing_required.setdefault(check.name, []).append(dependency.name)

        return dependency_graph, missing_required

    def _topological_sort(self, dependency_graph: dict[str, set[str]]) -> tuple[str, ...]:
        remaining_dependencies = {
            name: set(dependencies) for name, dependencies in dependency_graph.items()
        }
        ready = sorted(name for name, deps in remaining_dependencies.items() if not deps)
        ordered: list[str] = []

        while ready:
            name = ready.pop(0)
            ordered.append(name)
            for dependent, dependencies in remaining_dependencies.items():
                if name in dependencies:
                    dependencies.remove(name)
                    if not dependencies and dependent not in ordered and dependent not in ready:
                        ready.append(dependent)
                        ready.sort()

        if len(ordered) != len(remaining_dependencies):
            cycle = self._find_cycle(dependency_graph)
            raise HealthCheckCycleError(cycle)

        return tuple(ordered)

    def _find_cycle(self, dependency_graph: dict[str, set[str]]) -> tuple[str, ...]:
        visiting: set[str] = set()
        visited: set[str] = set()
        stack: list[str] = []

        def visit(node: str) -> tuple[str, ...] | None:
            visiting.add(node)
            stack.append(node)
            for neighbor in dependency_graph.get(node, set()):
                if neighbor in visiting:
                    if neighbor in stack:
                        start_index = stack.index(neighbor)
                        return tuple(stack[start_index:] + [neighbor])
                    return (neighbor, node, neighbor)
                if neighbor not in visited:
                    cycle_path = visit(neighbor)
                    if cycle_path:
                        return cycle_path
            visiting.remove(node)
            visited.add(node)
            stack.pop()
            return None

        for node in sorted(dependency_graph):
            if node not in visited:
                cycle = visit(node)
                if cycle:
                    return cycle

        return ()


__all__ = [
    "HealthCheckCycleError",
    "HealthCheckDependencyError",
    "HealthCheckDescriptor",
    "HealthCheckMode",
    "HealthCheckOutcome",
    "HealthCheckPlanError",
    "HealthCheckPlanner",
]
