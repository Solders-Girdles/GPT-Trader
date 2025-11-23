"""Derivatives product discovery with explicit filtering and spec caching.

This module implements product discovery for futures and perpetuals with explicit
filtering by product_type and contract_expiry_type. It caches symbol specs including
max leverage and min notional values.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from bot_v2.features.brokerages.core.interfaces import IBrokerage, Product

logger = get_logger(__name__, component="derivatives_products")


@dataclass(frozen=True)
class DerivativesProductSpec:
    """Cached specification for a derivatives product."""

    symbol: str
    product_type: str  # "FUTURE" or "SPOT" (perpetuals are SPOT with -PERP suffix)
    contract_expiry_type: str | None  # "PERPETUAL" or None for futures with expiry
    max_leverage: Decimal
    min_notional: Decimal
    base_increment: Decimal | None
    quote_increment: Decimal | None
    base_min_size: Decimal | None
    base_max_size: Decimal | None
    raw_product_data: dict[str, Any]


@dataclass(frozen=True)
class DerivativesProductDiscoveryResult:
    """Result of derivatives product discovery."""

    perpetuals: list[DerivativesProductSpec]
    futures: list[DerivativesProductSpec]
    total_count: int
    error_message: str | None


class DerivativesProductCache:
    """Cache for derivatives product specifications."""

    def __init__(self) -> None:
        self._specs: dict[str, DerivativesProductSpec] = {}
        self._last_refresh: float | None = None

    def get(self, symbol: str) -> DerivativesProductSpec | None:
        """Get cached spec for a symbol."""
        return self._specs.get(symbol)

    def set(self, symbol: str, spec: DerivativesProductSpec) -> None:
        """Cache a product spec."""
        self._specs[symbol] = spec

    def bulk_set(self, specs: list[DerivativesProductSpec]) -> None:
        """Cache multiple product specs."""
        for spec in specs:
            self._specs[spec.symbol] = spec

    def get_all_perpetuals(self) -> list[DerivativesProductSpec]:
        """Get all cached perpetual specs."""
        return [spec for spec in self._specs.values() if spec.contract_expiry_type == "PERPETUAL"]

    def get_all_futures(self) -> list[DerivativesProductSpec]:
        """Get all cached future specs (non-perpetual derivatives)."""
        return [
            spec
            for spec in self._specs.values()
            if spec.product_type == "FUTURE" and spec.contract_expiry_type != "PERPETUAL"
        ]

    def clear(self) -> None:
        """Clear the cache."""
        self._specs.clear()
        self._last_refresh = None


def discover_derivatives_products(
    broker: IBrokerage,
    *,
    cache: DerivativesProductCache | None = None,
) -> DerivativesProductDiscoveryResult:
    """Discover futures and perpetuals with explicit filtering.

    This function queries products with explicit filtering for:
    - product_type=FUTURE for futures contracts
    - contract_expiry_type=PERPETUAL for perpetuals

    It does NOT rely on implicit filtering and caches specs including
    max leverage and min notional (10 USDC for INTX perps).

    Args:
        broker: The brokerage instance to query
        cache: Optional cache to store discovered specs

    Returns:
        DerivativesProductDiscoveryResult with discovered products
    """
    logger.info(
        "Discovering derivatives products with explicit filtering",
        operation="derivatives_product_discovery",
        stage="start",
    )

    perpetuals: list[DerivativesProductSpec] = []
    futures: list[DerivativesProductSpec] = []
    error_message: str | None = None

    try:
        # List all products
        all_products = broker.list_products()  # type: ignore[attr-defined]

        if not all_products or not isinstance(all_products, list):
            error_message = "No products returned from broker"
            logger.warning(
                "No products returned from broker",
                operation="derivatives_product_discovery",
                stage="list_products",
            )
            return DerivativesProductDiscoveryResult(
                perpetuals=[],
                futures=[],
                total_count=0,
                error_message=error_message,
            )

        # Explicitly filter for perpetuals and futures
        for product in all_products:
            if not hasattr(product, "symbol"):
                continue

            symbol = str(product.symbol)

            # Check if this is a perpetual (explicit check)
            is_perpetual = False
            contract_expiry_type: str | None = None

            # Check for perpetual indicators
            if symbol.endswith("-PERP") or symbol.endswith("-PERPETUAL"):
                is_perpetual = True
                contract_expiry_type = "PERPETUAL"
            elif hasattr(product, "product_type") and hasattr(product, "contract_expiry_type"):
                product_type = str(getattr(product, "product_type", "")).upper()
                expiry_type = str(getattr(product, "contract_expiry_type", "")).upper()
                if expiry_type == "PERPETUAL":
                    is_perpetual = True
                    contract_expiry_type = "PERPETUAL"
                elif product_type == "FUTURE":
                    # Future with expiry (not perpetual)
                    is_perpetual = False
                    contract_expiry_type = expiry_type if expiry_type else None

            # Skip non-derivatives
            if not is_perpetual and not (
                hasattr(product, "product_type")
                and str(getattr(product, "product_type", "")).upper() == "FUTURE"
            ):
                continue

            # Extract specs
            spec = _extract_product_spec(product, is_perpetual, contract_expiry_type)

            if spec:
                if is_perpetual:
                    perpetuals.append(spec)
                else:
                    futures.append(spec)

                # Cache the spec
                if cache is not None:
                    cache.set(symbol, spec)

        logger.info(
            "Derivatives product discovery complete",
            operation="derivatives_product_discovery",
            stage="complete",
            perpetuals=len(perpetuals),
            futures=len(futures),
            total=len(perpetuals) + len(futures),
        )

    except Exception as exc:
        error_message = f"Failed to discover derivatives products: {exc}"
        logger.error(
            "Failed to discover derivatives products",
            operation="derivatives_product_discovery",
            stage="error",
            error=str(exc),
            exc_info=True,
        )

    return DerivativesProductDiscoveryResult(
        perpetuals=perpetuals,
        futures=futures,
        total_count=len(perpetuals) + len(futures),
        error_message=error_message,
    )


def _extract_product_spec(
    product: Product,
    is_perpetual: bool,
    contract_expiry_type: str | None,
) -> DerivativesProductSpec | None:
    """Extract product specification from a product object.

    Args:
        product: The product object from the broker
        is_perpetual: Whether this is a perpetual contract
        contract_expiry_type: The contract expiry type

    Returns:
        DerivativesProductSpec or None if extraction fails
    """
    try:
        symbol = str(product.symbol)

        # Extract product type
        product_type = "FUTURE"
        if hasattr(product, "product_type"):
            product_type = str(getattr(product, "product_type", "FUTURE")).upper()

        # Extract leverage (default to 1 if not available)
        max_leverage = Decimal("1")
        if hasattr(product, "max_leverage"):
            max_leverage = Decimal(str(getattr(product, "max_leverage", "1")))

        # Extract min notional (default to 10 USDC for INTX perps per spec)
        min_notional = Decimal("10")  # Default INTX perps min notional
        if hasattr(product, "min_notional"):
            min_notional = Decimal(str(getattr(product, "min_notional", "10")))
        elif hasattr(product, "quote_min_size"):
            # Some brokers use quote_min_size for min notional
            min_notional = Decimal(str(getattr(product, "quote_min_size", "10")))

        # Extract increments and sizes
        base_increment: Decimal | None = None
        if hasattr(product, "base_increment"):
            base_increment = Decimal(str(getattr(product, "base_increment", "0")))

        quote_increment: Decimal | None = None
        if hasattr(product, "quote_increment"):
            quote_increment = Decimal(str(getattr(product, "quote_increment", "0")))

        base_min_size: Decimal | None = None
        if hasattr(product, "base_min_size"):
            base_min_size = Decimal(str(getattr(product, "base_min_size", "0")))

        base_max_size: Decimal | None = None
        if hasattr(product, "base_max_size"):
            base_max_size = Decimal(str(getattr(product, "base_max_size", "0")))

        # Build raw product data dict
        raw_product_data: dict[str, Any] = {}
        if hasattr(product, "__dict__"):
            raw_product_data = dict(product.__dict__)
        elif hasattr(product, "model_dump"):
            raw_product_data = product.model_dump()  # type: ignore[attr-defined]

        return DerivativesProductSpec(
            symbol=symbol,
            product_type=product_type,
            contract_expiry_type=contract_expiry_type,
            max_leverage=max_leverage,
            min_notional=min_notional,
            base_increment=base_increment,
            quote_increment=quote_increment,
            base_min_size=base_min_size,
            base_max_size=base_max_size,
            raw_product_data=raw_product_data,
        )

    except Exception as exc:
        logger.warning(
            "Failed to extract product spec",
            operation="derivatives_product_discovery",
            stage="extract_spec",
            symbol=getattr(product, "symbol", "unknown"),
            error=str(exc),
        )
        return None


__all__ = [
    "DerivativesProductCache",
    "DerivativesProductDiscoveryResult",
    "DerivativesProductSpec",
    "discover_derivatives_products",
]
