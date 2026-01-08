"""
DEPRECATED: DeterministicBroker has moved to gpt_trader.features.brokerages.mock.deterministic

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.brokerages.mock import DeterministicBroker
"""

import warnings

from gpt_trader.features.brokerages.mock.deterministic import DeterministicBroker

warnings.warn(
    "gpt_trader.orchestration.deterministic_broker is deprecated. "
    "Import from gpt_trader.features.brokerages.mock instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DeterministicBroker"]
