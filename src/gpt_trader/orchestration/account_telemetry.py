"""
DEPRECATED: AccountTelemetryService has moved to gpt_trader.features.live_trade.telemetry.account

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.telemetry import AccountTelemetryService
"""

import warnings

from gpt_trader.features.live_trade.telemetry.account import AccountTelemetryService

warnings.warn(
    "gpt_trader.orchestration.account_telemetry is deprecated. "
    "Import from gpt_trader.features.live_trade.telemetry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AccountTelemetryService"]
