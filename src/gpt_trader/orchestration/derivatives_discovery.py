"""
DEPRECATED: derivatives_discovery has moved to gpt_trader.features.brokerages.coinbase.derivatives_discovery

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.brokerages.coinbase.derivatives_discovery import (
        DerivativesEligibility, discover_derivatives_eligibility, ...
    )
"""

import warnings

from gpt_trader.features.brokerages.coinbase.derivatives_discovery import (
    DerivativesEligibility,
    DerivativesMarket,
    discover_derivatives_eligibility,
)

warnings.warn(
    "gpt_trader.orchestration.derivatives_discovery is deprecated. "
    "Import from gpt_trader.features.brokerages.coinbase.derivatives_discovery instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DerivativesEligibility",
    "DerivativesMarket",
    "discover_derivatives_eligibility",
]
