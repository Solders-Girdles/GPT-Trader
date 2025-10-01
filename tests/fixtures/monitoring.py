"""Shared monitoring and guard fixtures for runtime protection.

Provides guard configurations, managers, and scenario builders for testing
the runtime guard system. These fixtures reduce per-file overhead while
maintaining the narrative intent of guard behavior testing.
"""

from __future__ import annotations

import pytest

from bot_v2.monitoring.alerts import AlertSeverity
from bot_v2.monitoring.runtime_guards import (
    GuardConfig,
    RuntimeGuard,
    RuntimeGuardManager,
)


@pytest.fixture
def basic_guard_config() -> GuardConfig:
    """Create a basic guard configuration.

    Provides standard settings for testing guard behavior:
    - WARNING severity (non-critical by default)
    - 5-minute cooldown to prevent alert spam
    - Auto-shutdown disabled (requires explicit enabling)
    - 60-second evaluation window
    """
    return GuardConfig(
        name="test_guard",
        enabled=True,
        threshold=100.0,
        window_seconds=60,
        severity=AlertSeverity.WARNING,
        auto_shutdown=False,
        cooldown_seconds=300,
    )


@pytest.fixture
def guard(basic_guard_config: GuardConfig) -> RuntimeGuard:
    """Create a basic runtime guard.

    Uses the standard guard configuration. Starts in HEALTHY status,
    ready to monitor conditions and trigger alerts when breached.
    """
    return RuntimeGuard(basic_guard_config)


@pytest.fixture
def guard_manager() -> RuntimeGuardManager:
    """Create a runtime guard manager.

    Orchestrates multiple guards for coordinated monitoring.
    Starts empty - add guards as needed per test scenario.
    """
    return RuntimeGuardManager()


@pytest.fixture
def critical_guard_config() -> GuardConfig:
    """Create a critical guard configuration.

    For testing critical failures that require immediate shutdown:
    - CRITICAL severity
    - Auto-shutdown enabled
    - Shorter cooldown (1 minute)
    - Tighter evaluation window
    """
    return GuardConfig(
        name="critical_guard",
        enabled=True,
        threshold=50.0,
        window_seconds=30,
        severity=AlertSeverity.CRITICAL,
        auto_shutdown=True,
        cooldown_seconds=60,
    )


@pytest.fixture
def disabled_guard_config() -> GuardConfig:
    """Create a disabled guard configuration.

    For testing guards that are configured but not actively monitoring.
    Useful for testing guard enable/disable workflows.
    """
    return GuardConfig(
        name="disabled_guard",
        enabled=False,
        threshold=100.0,
        window_seconds=60,
        severity=AlertSeverity.WARNING,
        auto_shutdown=False,
        cooldown_seconds=300,
    )
