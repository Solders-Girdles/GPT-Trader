"""
DEPRECATED: HybridPaperBroker has moved to gpt_trader.features.brokerages.paper.hybrid

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.brokerages.paper import HybridPaperBroker
"""

import warnings

from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker

warnings.warn(
    "gpt_trader.orchestration.hybrid_paper_broker is deprecated. "
    "Import from gpt_trader.features.brokerages.paper instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HybridPaperBroker"]
