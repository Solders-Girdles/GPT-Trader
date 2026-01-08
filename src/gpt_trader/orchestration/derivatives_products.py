"""
DEPRECATED: derivatives_products has moved to gpt_trader.features.brokerages.coinbase.derivatives_products

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.brokerages.coinbase.derivatives_products import (
        DerivativesProductCache, discover_derivatives_products, ...
    )
"""

import warnings

from gpt_trader.features.brokerages.coinbase.derivatives_products import (
    DerivativesProductCache,
    DerivativesProductDiscoveryResult,
    DerivativesProductSpec,
    discover_derivatives_products,
)

warnings.warn(
    "gpt_trader.orchestration.derivatives_products is deprecated. "
    "Import from gpt_trader.features.brokerages.coinbase.derivatives_products instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DerivativesProductCache",
    "DerivativesProductDiscoveryResult",
    "DerivativesProductSpec",
    "discover_derivatives_products",
]
