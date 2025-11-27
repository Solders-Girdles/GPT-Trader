"""
Simplified Health Checks.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class HealthCheckResult:
    healthy: bool
    details: dict[str, Any]


__all__ = [
    "HealthCheckResult",
]
