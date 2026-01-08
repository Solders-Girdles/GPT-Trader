"""
DEPRECATED: MetricsPublisher has moved to gpt_trader.monitoring.system.metrics

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.monitoring.system import MetricsPublisher
"""

import warnings

from gpt_trader.monitoring.system.metrics import MetricsPublisher

warnings.warn(
    "gpt_trader.orchestration.system_monitor_metrics is deprecated. "
    "Import from gpt_trader.monitoring.system instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MetricsPublisher"]
