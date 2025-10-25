"""Production logger assembled from dedicated mixins."""

from __future__ import annotations

from .auth import AuthLoggingMixin
from .base import BaseProductionLogger
from .errors import ErrorLoggingMixin
from .monitoring import MonitoringLoggingMixin
from .network import NetworkLoggingMixin
from .trading import TradingLoggingMixin


class ProductionLogger(
    ErrorLoggingMixin,
    AuthLoggingMixin,
    NetworkLoggingMixin,
    MonitoringLoggingMixin,
    TradingLoggingMixin,
    BaseProductionLogger,
):
    """High-performance production logger with structured JSON output."""

    pass


__all__ = ["ProductionLogger"]
