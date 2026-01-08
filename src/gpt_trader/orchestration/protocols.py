"""
DEPRECATED: Protocol definitions have moved to gpt_trader.app.protocols

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.app.protocols import EventStoreProtocol
    from gpt_trader.app.protocols import AccountManagerProtocol
    from gpt_trader.app.protocols import RuntimeStateProtocol
"""

import warnings

from gpt_trader.app.protocols import (
    AccountManagerProtocol,
    EventStoreProtocol,
    RuntimeStateProtocol,
)

warnings.warn(
    "gpt_trader.orchestration.protocols is deprecated. "
    "Import from gpt_trader.app.protocols instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AccountManagerProtocol",
    "EventStoreProtocol",
    "RuntimeStateProtocol",
]
