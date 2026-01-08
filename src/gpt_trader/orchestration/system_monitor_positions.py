"""
DEPRECATED: PositionReconciler has moved to gpt_trader.monitoring.system.positions

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.monitoring.system import PositionReconciler
"""

import warnings

from gpt_trader.monitoring.system.positions import PositionReconciler

warnings.warn(
    "gpt_trader.orchestration.system_monitor_positions is deprecated. "
    "Import from gpt_trader.monitoring.system instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PositionReconciler"]
