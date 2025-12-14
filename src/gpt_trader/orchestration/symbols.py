"""Shared helpers for normalizing trading symbols across profiles."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from gpt_trader.orchestration.configuration import BotConfig, Profile

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="symbols")

PERPS_ALLOWLIST = frozenset({"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"})
US_FUTURES_ALLOWLIST = frozenset({"BTC-FUTURES", "ETH-FUTURES", "SOL-FUTURES", "XRP-FUTURES"})

# CFM (Coinbase Financial Markets) symbol mapping
# Maps base asset to active CFM contract symbol
# Note: These need to be updated when contracts roll/expire
CFM_SYMBOL_MAPPING: dict[str, str] = {
    "BTC": "BTC-20DEC30-CDE",
    "ETH": "ETH-20DEC30-CDE",
    "SOL": "SLP-20DEC30-CDE",  # Note: SOL uses SLP contract code
}

TradingMode = Literal["spot", "cfm"]


@dataclass(frozen=True)
class SymbolNormalizationLog:
    """Captured log emitted during symbol normalization."""

    level: int
    message: str
    args: tuple[object, ...] = ()


def derivatives_enabled(profile: Profile, *, config: BotConfig) -> bool:
    """Determine whether derivatives trading should be enabled for the profile."""

    try:  # Local import to avoid circular at module load time.
        from gpt_trader.orchestration.configuration import Profile as _Profile

        _ProfileClass: type[Profile] | None = _Profile
    except Exception:  # pragma: no cover - defensive fallback
        _ProfileClass = None

    if _ProfileClass is not None and isinstance(profile, _ProfileClass):
        if profile == _ProfileClass.SPOT:
            return False
    else:
        profile_value = str(getattr(profile, "value", profile or "")).lower()
        if profile_value == "spot":
            return False

    # Use strict config toggle
    return config.derivatives_enabled


def us_futures_enabled(profile: Profile, *, config: BotConfig) -> bool:
    """Determine whether US futures trading should be enabled for the profile."""

    # Check if derivatives are enabled at all
    if not derivatives_enabled(profile, config=config):
        return False

    # Check US futures specific flag
    if config.coinbase_us_futures_enabled:
        return True

    # Check derivatives type
    if config.coinbase_derivatives_type == "us_futures":
        return True

    return False


def intx_perpetuals_enabled(profile: Profile, *, config: BotConfig) -> bool:
    """Determine whether INTX perpetuals trading should be enabled for the profile."""

    # Check if derivatives are enabled at all
    if not derivatives_enabled(profile, config=config):
        return False

    # Check INTX perpetuals specific flag
    if config.coinbase_intx_perpetuals_enabled:
        return True

    # Check derivatives type (default to INTX)
    if config.coinbase_derivatives_type in ("intx_perps", "perpetuals"):
        return True

    return True


def normalize_symbol_list(
    symbols: Sequence[str] | None,
    *,
    allow_derivatives: bool,
    quote: str,
    allowed_perps: Iterable[str] | None = None,
    allowed_us_futures: Iterable[str] | None = None,
    fallback_bases: Sequence[str] | None = None,
) -> tuple[list[str], list[SymbolNormalizationLog]]:
    """Produce a normalised symbol list and captured log records."""

    logs: list[SymbolNormalizationLog] = []
    allowed_perps_set = set(allowed_perps) if allowed_perps is not None else set(PERPS_ALLOWLIST)
    allowed_us_futures_set = (
        set(allowed_us_futures) if allowed_us_futures is not None else set(US_FUTURES_ALLOWLIST)
    )
    normalized: list[str] = []

    for raw in symbols or []:
        token = (raw or "").strip().upper()
        if not token:
            continue

        if allow_derivatives:
            if token.endswith("-PERP"):
                if token in allowed_perps_set:
                    normalized.append(token)
                else:
                    logs.append(
                        SymbolNormalizationLog(
                            logging.WARNING,
                            "Filtering unsupported perpetual symbol %s. Allowed perps: %s",
                            (token, sorted(allowed_perps_set)),
                        )
                    )
            elif token.endswith("-FUTURES"):
                if token in allowed_us_futures_set:
                    normalized.append(token)
                else:
                    logs.append(
                        SymbolNormalizationLog(
                            logging.WARNING,
                            "Filtering unsupported US futures symbol %s. Allowed US futures: %s",
                            (token, sorted(allowed_us_futures_set)),
                        )
                    )
            else:
                normalized.append(token)
            continue

        if token.endswith("-PERP"):
            base = token.split("-", 1)[0]
            replacement = f"{base}-{quote}"
            logs.append(
                SymbolNormalizationLog(
                    logging.WARNING,
                    "Derivatives disabled. Replacing %s with spot symbol %s",
                    (token, replacement),
                )
            )
            token = replacement

        normalized.append(token)

    normalized = list(dict.fromkeys(normalized))
    if normalized:
        return normalized, logs

    if allow_derivatives:
        fallback = ["BTC-PERP", "ETH-PERP", "BTC-FUTURES", "ETH-FUTURES"]
    else:
        if fallback_bases is None:
            from gpt_trader.orchestration.configuration import TOP_VOLUME_BASES

            fallback_bases = TOP_VOLUME_BASES
        fallback = [f"{base}-{quote}" for base in fallback_bases]

    logs.append(
        SymbolNormalizationLog(
            logging.INFO,
            "No valid symbols provided. Falling back to %s",
            (fallback,),
        )
    )
    return fallback, logs


def normalize_symbols(
    profile: Profile,
    symbols: Sequence[str] | None,
    *,
    config: BotConfig,
    quote: str | None = None,
) -> tuple[list[str], bool]:
    """Normalize configured symbols, applying per-profile defaults and gating."""

    quote_currency = (quote or config.coinbase_default_quote).upper()
    allow_derivatives = derivatives_enabled(profile, config=config)
    normalized, logs = normalize_symbol_list(
        symbols,
        allow_derivatives=allow_derivatives,
        quote=quote_currency,
    )

    for record in logs:
        logger.log(record.level, record.message, *record.args)

    return normalized, allow_derivatives


# =============================================================================
# CFM (Coinbase Financial Markets) Symbol Helpers
# =============================================================================


def cfm_enabled(config: BotConfig) -> bool:
    """Check if CFM futures trading is enabled.

    Args:
        config: Bot configuration.

    Returns:
        True if CFM is enabled via config flag or trading modes.
    """
    if config.cfm_enabled:
        return True
    if "cfm" in config.trading_modes:
        return True
    return False


def get_cfm_symbol(base_or_spot: str) -> str | None:
    """Map a base asset or spot symbol to its active CFM contract.

    Args:
        base_or_spot: Base asset (e.g., "BTC") or spot symbol (e.g., "BTC-USD").

    Returns:
        CFM contract symbol (e.g., "BTC-20DEC30-CDE") or None if not mapped.

    Examples:
        >>> get_cfm_symbol("BTC")
        'BTC-20DEC30-CDE'
        >>> get_cfm_symbol("BTC-USD")
        'BTC-20DEC30-CDE'
        >>> get_cfm_symbol("DOGE")
        None
    """
    # Extract base asset from spot symbol if needed
    base = base_or_spot.upper().split("-")[0]
    return CFM_SYMBOL_MAPPING.get(base)


def get_spot_symbol(cfm_or_base: str, quote: str = "USD") -> str:
    """Map a CFM symbol or base asset to its spot equivalent.

    Args:
        cfm_or_base: CFM contract symbol (e.g., "BTC-20DEC30-CDE") or base asset.
        quote: Quote currency (default: "USD").

    Returns:
        Spot symbol (e.g., "BTC-USD").

    Examples:
        >>> get_spot_symbol("BTC-20DEC30-CDE")
        'BTC-USD'
        >>> get_spot_symbol("BTC")
        'BTC-USD'
    """
    # CFM symbols have format: BASE-EXPIRY-CODE (e.g., BTC-20DEC30-CDE)
    # Extract base asset
    parts = cfm_or_base.upper().split("-")
    base = parts[0]

    # Handle SOL special case (SLP contract code)
    if base == "SLP":
        base = "SOL"

    return f"{base}-{quote}"


def normalize_symbol_for_mode(
    symbol: str,
    mode: TradingMode,
    quote: str = "USD",
) -> str:
    """Normalize a symbol for the specified trading mode.

    Converts between spot and CFM symbols as needed.

    Args:
        symbol: Input symbol (spot or CFM format).
        mode: Target trading mode ("spot" or "cfm").
        quote: Quote currency for spot symbols (default: "USD").

    Returns:
        Symbol normalized for the target mode.

    Examples:
        >>> normalize_symbol_for_mode("BTC-USD", "cfm")
        'BTC-20DEC30-CDE'
        >>> normalize_symbol_for_mode("BTC-20DEC30-CDE", "spot")
        'BTC-USD'
    """
    symbol = symbol.upper()

    if mode == "spot":
        # Convert CFM symbol to spot
        return get_spot_symbol(symbol, quote)
    elif mode == "cfm":
        # Convert spot symbol to CFM
        cfm_symbol = get_cfm_symbol(symbol)
        if cfm_symbol:
            return cfm_symbol
        # If no mapping exists, return original
        logger.warning(
            f"No CFM mapping for symbol {symbol}, using original"
        )
        return symbol
    else:
        return symbol


def get_symbol_pairs_for_hybrid(
    symbols: Sequence[str],
    quote: str = "USD",
) -> dict[str, dict[str, str]]:
    """Get spot/CFM symbol pairs for hybrid trading.

    Maps base assets to their corresponding spot and CFM symbols.

    Args:
        symbols: List of symbols (can be spot or CFM format).
        quote: Quote currency (default: "USD").

    Returns:
        Dictionary mapping base asset to {spot: symbol, cfm: symbol}.

    Example:
        >>> get_symbol_pairs_for_hybrid(["BTC-USD", "ETH-USD"])
        {
            'BTC': {'spot': 'BTC-USD', 'cfm': 'BTC-20DEC30-CDE'},
            'ETH': {'spot': 'ETH-USD', 'cfm': 'ETH-20DEC30-CDE'},
        }
    """
    pairs: dict[str, dict[str, str]] = {}

    for symbol in symbols:
        symbol = symbol.upper()
        base = symbol.split("-")[0]

        # Handle SLP special case
        if base == "SLP":
            base = "SOL"

        if base in pairs:
            continue

        spot = f"{base}-{quote}"
        cfm = CFM_SYMBOL_MAPPING.get(base)

        pairs[base] = {
            "spot": spot,
            "cfm": cfm if cfm else spot,  # Fallback to spot if no CFM mapping
        }

    return pairs
