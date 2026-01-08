"""
DEPRECATED: symbols module has moved to gpt_trader.features.live_trade.symbols

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.symbols import (
        normalize_symbols, derivatives_enabled, get_cfm_symbol, ...
    )
"""

import warnings

from gpt_trader.features.live_trade.symbols import (
    CFM_SYMBOL_MAPPING,
    PERPS_ALLOWLIST,
    US_FUTURES_ALLOWLIST,
    SymbolNormalizationLog,
    TradingMode,
    cfm_enabled,
    derivatives_enabled,
    get_cfm_symbol,
    get_spot_symbol,
    get_symbol_pairs_for_hybrid,
    intx_perpetuals_enabled,
    logger,
    normalize_symbol_for_mode,
    normalize_symbol_list,
    normalize_symbols,
    us_futures_enabled,
)

warnings.warn(
    "gpt_trader.orchestration.symbols is deprecated. "
    "Import from gpt_trader.features.live_trade.symbols instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CFM_SYMBOL_MAPPING",
    "PERPS_ALLOWLIST",
    "US_FUTURES_ALLOWLIST",
    "SymbolNormalizationLog",
    "TradingMode",
    "cfm_enabled",
    "derivatives_enabled",
    "get_cfm_symbol",
    "get_spot_symbol",
    "get_symbol_pairs_for_hybrid",
    "intx_perpetuals_enabled",
    "logger",
    "normalize_symbol_for_mode",
    "normalize_symbol_list",
    "normalize_symbols",
    "us_futures_enabled",
]
