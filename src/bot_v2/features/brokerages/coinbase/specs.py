"""
Product specifications and quantization for Coinbase perpetuals.

Handles:
- Product spec loading and override management
- Side-aware price quantization (BUY=floor, SELL=ceil)
- Safe position sizing with buffers
- Pre-flight order validation
"""

import logging
import math
import os
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path
from typing import Any

import yaml

from bot_v2.utilities.quantization import quantize_price_side_aware

logger = logging.getLogger(__name__)


class ProductSpec:
    """Product specification with quantization rules."""

    def __init__(self, product_id: str, spec_data: dict[str, Any]) -> None:
        self.product_id = product_id
        self.min_size = Decimal(str(spec_data.get("min_size", "0.001")))
        self.step_size = Decimal(str(spec_data.get("step_size", "0.001")))
        self.price_increment = Decimal(str(spec_data.get("price_increment", "0.01")))
        self.min_notional = Decimal(str(spec_data.get("min_notional", "10")))
        self.max_size = Decimal(str(spec_data.get("max_size", "1000000")))
        self.last_verified = spec_data.get("last_verified", "unknown")
        self.source = spec_data.get("source", "api")

        # Optional fields
        self.slippage_multiplier = Decimal(str(spec_data.get("slippage_multiplier", "1.0")))
        self.safe_buffer = Decimal(str(spec_data.get("safe_buffer", "0.1")))  # 10% default


class SpecsService:
    """Manages product specifications with YAML overrides."""

    def __init__(self, config_path: str | None = None) -> None:
        self.specs_cache: dict[str, ProductSpec] = {}
        self.overrides: dict[str, dict] = {}

        # Load overrides from YAML if available
        if config_path is None:
            config_path = os.getenv("PERPS_SPECS_PATH", "config/brokers/coinbase_perp_specs.yaml")

        self.load_overrides(config_path)

    def load_overrides(self, config_path: str) -> None:
        """Load spec overrides from YAML configuration."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Specs config not found: {config_path}")
            return

        try:
            with open(path) as f:
                config = yaml.safe_load(f)
                if config and "products" in config:
                    self.overrides = config["products"]
                    logger.info(f"Loaded {len(self.overrides)} product overrides")
        except Exception as e:
            logger.error(f"Failed to load specs config: {e}")

    def build_spec(self, product_id: str, api_data: dict | None = None) -> ProductSpec:
        """Build spec from API data merged with overrides."""
        # Check cache first
        if product_id in self.specs_cache:
            return self.specs_cache[product_id]

        # Start with API data or defaults
        spec_data = api_data.copy() if api_data else {}

        # Apply overrides if available
        if product_id in self.overrides:
            spec_data.update(self.overrides[product_id])
            logger.debug(f"Applied overrides for {product_id}")

        # Create and cache spec
        spec = ProductSpec(product_id, spec_data)
        self.specs_cache[product_id] = spec
        return spec

    def quantize_price_side_aware(self, product_id: str, side: str, price: float) -> Decimal:
        """
        Quantize price based on side for better fills.
        BUY orders: floor to price increment (more aggressive)
        SELL orders: ceil to price increment (more aggressive)
        """
        spec = self.get_spec(product_id)
        price_decimal = Decimal(str(price))
        increment = spec.price_increment

        if increment <= 0:
            return price_decimal

        # Calculate number of increments
        num_increments = price_decimal / increment

        if side.upper() == "BUY":
            # Floor for buy orders (lower price = more aggressive)
            quantized = math.floor(num_increments) * increment
        else:  # SELL
            # Ceil for sell orders (higher price = more aggressive)
            quantized = math.ceil(num_increments) * increment

        return Decimal(str(quantized))

    def quantize_size(self, product_id: str, size: float) -> Decimal:
        """Quantize size to step size (always floor for safety)."""
        spec = self.get_spec(product_id)
        size_decimal = Decimal(str(size))
        step = spec.step_size

        if step <= 0:
            return size_decimal

        # Always floor size to avoid exceeding limits
        num_steps = size_decimal / step
        quantized = math.floor(num_steps) * step

        return Decimal(str(quantized))

    def calculate_safe_position_size(
        self, product_id: str, target_notional: float, mark_price: float
    ) -> tuple[Decimal, str]:
        """
        Calculate safe position size with buffers.
        Returns (safe_size, reason) tuple.
        """
        spec = self.get_spec(product_id)

        # Calculate raw size from notional
        raw_size = Decimal(str(target_notional)) / Decimal(str(mark_price))

        # Apply safety buffer (e.g., 10% reduction)
        buffer_factor = Decimal("1") - spec.safe_buffer
        safe_size = raw_size * buffer_factor

        # Quantize to step size
        quantized_size = self.quantize_size(product_id, float(safe_size))

        # Check minimum size
        if quantized_size < spec.min_size:
            # Try to meet minimum with buffer
            buffered_min = spec.min_size * (Decimal("1") + spec.safe_buffer)
            if buffered_min * Decimal(str(mark_price)) <= Decimal(str(target_notional)):
                return buffered_min, "adjusted_to_min_with_buffer"
            else:
                return Decimal("0"), "below_minimum_notional"

        # Check maximum size
        if quantized_size > spec.max_size:
            return spec.max_size, "capped_at_maximum"

        # Check minimum notional
        notional = quantized_size * Decimal(str(mark_price))
        if notional < spec.min_notional:
            # Try to meet minimum notional
            min_size = (spec.min_notional * (Decimal("1") + spec.safe_buffer)) / Decimal(
                str(mark_price)
            )
            min_size = self.quantize_size(product_id, float(min_size))
            if min_size <= spec.max_size:
                return min_size, "adjusted_for_min_notional"
            else:
                return Decimal("0"), "cannot_meet_min_notional"

        return quantized_size, "within_limits"

    def validate_order(
        self, product_id: str, side: str, order_type: str, size: float, price: float | None = None
    ) -> dict[str, Any]:
        """
        Validate and adjust order parameters.
        Returns dict with adjusted values or error details.
        """
        spec = self.get_spec(product_id)
        result = {"valid": True, "adjusted_size": None, "adjusted_price": None, "reasons": []}

        # Validate and adjust size
        size_decimal = Decimal(str(size))
        quantized_size = self.quantize_size(product_id, size)

        if quantized_size != size_decimal:
            result["adjusted_size"] = float(quantized_size)
            result["reasons"].append(f"size_quantized_to_{quantized_size}")
        else:
            result["adjusted_size"] = size

        # Check size limits
        if quantized_size < spec.min_size:
            result["valid"] = False
            result["reasons"].append(f"size_below_minimum_{spec.min_size}")
            return result

        if quantized_size > spec.max_size:
            result["valid"] = False
            result["reasons"].append(f"size_above_maximum_{spec.max_size}")
            return result

        # Validate price if provided (for limit/stop orders)
        if price is not None and order_type.upper() in ["LIMIT", "STOP_LIMIT"]:
            price_decimal = Decimal(str(price))
            quantized_price = self.quantize_price_side_aware(product_id, side, price)

            if quantized_price != price_decimal:
                result["adjusted_price"] = float(quantized_price)
                result["reasons"].append(f"price_quantized_to_{quantized_price}")
            else:
                result["adjusted_price"] = price

            # Check minimum notional
            notional = quantized_size * quantized_price
            if notional < spec.min_notional:
                result["valid"] = False
                result["reasons"].append(f"notional_below_minimum_{spec.min_notional}")
                return result
        else:
            result["adjusted_price"] = price

        return result

    def get_spec(self, product_id: str) -> ProductSpec:
        """Get spec for product, building if necessary."""
        if product_id not in self.specs_cache:
            return self.build_spec(product_id)
        return self.specs_cache[product_id]

    def get_slippage_multiplier(self, product_id: str) -> float:
        """Get slippage multiplier for product."""
        spec = self.get_spec(product_id)
        return float(spec.slippage_multiplier)


# Global instance
_specs_service: SpecsService | None = None


def get_specs_service() -> SpecsService:
    """Get or create global specs service."""
    global _specs_service
    if _specs_service is None:
        _specs_service = SpecsService()
    return _specs_service


# === Lightweight module-level helpers for engine preflight ===


@dataclass
class ValidationResult:
    ok: bool
    reason: str | None = None
    adjusted_quantity: Decimal | None = None
    adjusted_price: Decimal | None = None


def quantize_size(size: Decimal, step_size: Decimal) -> Decimal:
    """Floor size to the exchange step size."""
    if step_size is None or step_size == 0:
        return size
    q = (size / step_size).to_integral_value(rounding=ROUND_DOWN)
    return (q * step_size).quantize(step_size)


def quantize_size_up(size: Decimal, step_size: Decimal) -> Decimal:
    """Ceil size to the next exchange step size."""
    if step_size is None or step_size == 0:
        return size
    q = (size / step_size).to_integral_value(rounding=ROUND_UP)
    return (q * step_size).quantize(step_size)


def validate_order(
    *,
    product,
    side: str,
    quantity: Decimal | None = None,
    order_type: str,
    price: Decimal | None = None,
    overrides: dict[str, dict[str, str]] | None = None,
) -> ValidationResult:
    """Validate an order against product/spec requirements.

    Provides adjusted fields on success or reasoned failure on violations.
    """
    # Step size and min size
    step = Decimal(str(getattr(product, "step_size", Decimal("0.001"))))
    min_size = Decimal(str(getattr(product, "min_size", Decimal("0.001"))))
    price_inc = Decimal(str(getattr(product, "price_increment", Decimal("0.01"))))
    min_notional = getattr(product, "min_notional", None)
    min_notional = Decimal(str(min_notional)) if min_notional is not None else None

    if quantity is None:
        return ValidationResult(ok=False, reason="quantity_missing")

    base_quantity = quantity if isinstance(quantity, Decimal) else Decimal(str(quantity))

    # Quantize quantity and auto-bump to the minimum tradable size when undershooting
    adj_quantity = quantize_size(base_quantity, step)
    if adj_quantity < min_size:
        adj_quantity = min_size

    # Price handling
    adj_price: Decimal | None = None
    if order_type.lower() in ("limit", "stop_limit"):
        if price is None:
            return ValidationResult(ok=False, reason="price_required")
        adj_price = quantize_price_side_aware(Decimal(str(price)), price_inc, side)
    else:
        adj_price = price

    # Notional check
    if min_notional is not None and adj_price is not None:
        notional = adj_quantity * adj_price
        if notional < min_notional:
            # Suggest quantity to clear notional threshold (no buffer in validator)
            needed = quantize_size_up(min_notional / adj_price, step)
            if needed < min_size:
                needed = min_size
            return ValidationResult(
                ok=False,
                reason="min_notional",
                adjusted_quantity=needed,
                adjusted_price=adj_price,
            )

    return ValidationResult(ok=True, adjusted_quantity=adj_quantity, adjusted_price=adj_price)


def calculate_safe_position_size(
    *,
    product,
    side: str,
    intended_quantity: Decimal,
    ref_price: Decimal,
    overrides: dict[str, dict[str, str]] | None = None,
) -> Decimal:
    """Return a size that safely clears minimum thresholds with +10% buffer."""
    step = Decimal(str(getattr(product, "step_size", Decimal("0.001"))))
    min_size = Decimal(str(getattr(product, "min_size", Decimal("0.001"))))
    min_notional = getattr(product, "min_notional", None)
    min_notional = Decimal(str(min_notional)) if min_notional is not None else None

    safe_quantity = quantize_size(max(intended_quantity, min_size * Decimal("1.10")), step)
    if min_notional:
        notional = safe_quantity * ref_price
        target = min_notional * Decimal("1.10")
        if notional < target and ref_price > 0:
            needed = quantize_size(target / ref_price, step)
            if needed < min_size:
                needed = min_size
            safe_quantity = needed
    return safe_quantity
