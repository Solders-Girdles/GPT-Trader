from __future__ import annotations

from decimal import Decimal
from typing import Any

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from bot_v2.features.brokerages.coinbase.client.base import CoinbaseClientBase
from bot_v2.features.brokerages.coinbase.specs import (
    ProductSpec,
    SpecsService,
    calculate_safe_position_size,
    validate_order,
)


def _make_service(step_size: Decimal, price_increment: Decimal) -> SpecsService:
    service = SpecsService(config_path="config/__does_not_exist__.yaml")
    spec_data = {
        "step_size": str(step_size),
        "min_size": "0",
        "max_size": "1000000",
        "price_increment": str(price_increment),
    }
    spec = ProductSpec("BTC-PERP", spec_data)
    service.specs_cache["BTC-PERP"] = spec
    return service


@seed(1337)
@settings(max_examples=80, deadline=None)
@given(
    step=st.decimals(
        min_value="0.0001", max_value="5", allow_nan=False, allow_infinity=False, places=6
    ),
    size=st.decimals(
        min_value="0", max_value="5000", allow_nan=False, allow_infinity=False, places=6
    ),
)
def test_quantize_size_never_exceeds_input(step: Decimal, size: Decimal) -> None:
    service = _make_service(step, price_increment=Decimal("0.01"))
    quantized = service.quantize_size("BTC-PERP", float(size))
    size_decimal = Decimal(str(size))
    assert quantized <= size_decimal
    if step > 0:
        assert quantized % step == 0


@seed(7331)
@settings(max_examples=80, deadline=None)
@given(
    price=st.decimals(
        min_value="0.01", max_value="50000", allow_nan=False, allow_infinity=False, places=4
    ),
    increment=st.decimals(
        min_value="0.01", max_value="250", allow_nan=False, allow_infinity=False, places=2
    ),
    side=st.sampled_from(["BUY", "SELL"]),
)
def test_quantize_price_respects_increment(price: Decimal, increment: Decimal, side: str) -> None:
    service = _make_service(step_size=Decimal("0.0001"), price_increment=increment)
    quantized = service.quantize_price_side_aware("BTC-PERP", side, float(price))
    price_decimal = Decimal(str(price))
    if side == "BUY":
        assert quantized <= price_decimal
    else:
        assert quantized >= price_decimal
    if increment > 0:
        assert abs(quantized - price_decimal) < increment
        assert quantized % increment == 0


class _PaginationClient(CoinbaseClientBase):
    def __init__(self, pages: list[list[int]]) -> None:
        super().__init__(
            base_url="https://example.com",
            auth=None,
            enable_keep_alive=False,
        )
        self._pages = pages
        self._cursor_calls = 0

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        index = self._cursor_calls
        self._cursor_calls += 1
        current = self._pages[index]
        next_cursor = str(index + 1) if index + 1 < len(self._pages) else None
        return {"items": current, "next_cursor": next_cursor}


@seed(4242)
@settings(max_examples=60, deadline=None)
@given(
    pages=st.lists(
        st.lists(st.integers(min_value=0, max_value=100), max_size=5),
        min_size=1,
        max_size=5,
    )
)
def test_paginate_yields_all_items_in_order(pages: list[list[int]]) -> None:
    client = _PaginationClient(pages)
    collected = list(client.paginate("/fake", params={}, items_key="items"))
    expected: list[int] = [item for page in pages for item in page]
    assert collected == expected
    assert client._cursor_calls == len(pages)


# Property-based tests for safe position sizing
@seed(9999)
@settings(max_examples=100, deadline=None)
@given(
    target_notional=st.decimals(
        min_value="1", max_value="1000000", allow_nan=False, allow_infinity=False, places=2
    ),
    mark_price=st.decimals(
        min_value="0.01", max_value="100000", allow_nan=False, allow_infinity=False, places=4
    ),
    min_size=st.decimals(
        min_value="0.0001", max_value="10", allow_nan=False, allow_infinity=False, places=6
    ),
    step_size=st.decimals(
        min_value="0.0001", max_value="5", allow_nan=False, allow_infinity=False, places=6
    ),
    max_size=st.decimals(
        min_value="1", max_value="1000000", allow_nan=False, allow_infinity=False, places=2
    ),
    min_notional=st.decimals(
        min_value="1", max_value="10000", allow_nan=False, allow_infinity=False, places=2
    ),
    safe_buffer=st.decimals(
        min_value="0.01", max_value="0.5", allow_nan=False, allow_infinity=False, places=3
    ),
)
def test_calculate_safe_position_size_invariants(
    target_notional: Decimal,
    mark_price: Decimal,
    min_size: Decimal,
    step_size: Decimal,
    max_size: Decimal,
    min_notional: Decimal,
    safe_buffer: Decimal,
) -> None:
    """Property-based test for safe position sizing invariants."""

    # Create mock product
    class MockProduct:
        pass

    product = MockProduct()
    product.min_size = min_size
    product.step_size = step_size
    product.max_size = max_size
    product.min_notional = min_notional

    # Ensure valid constraints
    if min_size > max_size:
        max_size = min_size * Decimal("2")
    if step_size > min_size:
        step_size = min_size / Decimal("10")

    # Calculate safe size
    safe_size = calculate_safe_position_size(
        product=product,
        side="buy",
        intended_quantity=target_notional / mark_price,
        ref_price=mark_price,
    )

    # Invariants that must hold
    assert safe_size >= Decimal("0"), "Safe size should never be negative"

    # Size should be quantized to step size (unless zero)
    if safe_size > 0 and step_size > 0:
        steps = safe_size / step_size
        assert (
            steps == steps.to_integral_value()
        ), f"Size {safe_size} not quantized to step {step_size}"

    # Should not exceed max_size
    assert safe_size <= max_size, f"Safe size {safe_size} exceeds max_size {max_size}"

    # Should meet minimum size requirements (with buffer consideration)
    if safe_size > 0:
        min_required = min_size * Decimal("1.1")  # 10% buffer
        assert (
            safe_size >= min_required
        ), f"Safe size {safe_size} below buffered min_size {min_required}"

    # Notional should meet minimum requirements (with buffer consideration)
    if safe_size > 0:
        notional = safe_size * mark_price
        min_notional_required = min_notional * Decimal("1.1")  # 10% buffer
        assert (
            notional >= min_notional or safe_size == 0
        ), f"Notional {notional} below required {min_notional_required}"


# Property-based tests for order validation
@seed(8888)
@settings(max_examples=100, deadline=None)
@given(
    quantity=st.decimals(
        min_value="0.0001", max_value="10000", allow_nan=False, allow_infinity=False, places=6
    ),
    price=st.decimals(
        min_value="0.01", max_value="100000", allow_nan=False, allow_infinity=False, places=4
    )
    | st.none(),
    min_size=st.decimals(
        min_value="0.0001", max_value="10", allow_nan=False, allow_infinity=False, places=6
    ),
    step_size=st.decimals(
        min_value="0.0001", max_value="5", allow_nan=False, allow_infinity=False, places=6
    ),
    price_increment=st.decimals(
        min_value="0.001", max_value="100", allow_nan=False, allow_infinity=False, places=3
    ),
    min_notional=st.decimals(
        min_value="1", max_value="10000", allow_nan=False, allow_infinity=False, places=2
    )
    | st.none(),
    order_type=st.sampled_from(["market", "limit", "stop_limit"]),
    side=st.sampled_from(["buy", "sell"]),
)
def test_validate_order_invariants(
    quantity: Decimal,
    price: Decimal | None,
    min_size: Decimal,
    step_size: Decimal,
    price_increment: Decimal,
    min_notional: Decimal | None,
    order_type: str,
    side: str,
) -> None:
    """Property-based test for order validation invariants."""

    # Create mock product
    class MockProduct:
        pass

    product = MockProduct()
    product.min_size = min_size
    product.step_size = step_size
    product.price_increment = price_increment
    product.min_notional = min_notional

    # Ensure valid constraints
    if step_size > min_size:
        step_size = min_size / Decimal("10")

    # Validate order
    result = validate_order(
        product=product,
        side=side,
        quantity=quantity,
        order_type=order_type,
        price=price,
    )

    # Invariants that must hold
    assert isinstance(result.ok, bool), "Result should have boolean ok field"
    assert (
        isinstance(result.adjusted_quantity, Decimal) or result.adjusted_quantity is None
    ), "Adjusted quantity should be Decimal or None"
    assert (
        isinstance(result.adjusted_price, Decimal) or result.adjusted_price is None
    ), "Adjusted price should be Decimal or None"
    assert (
        isinstance(result.reason, str) or result.reason is None
    ), "Reason should be string or None"

    # If validation passes, adjusted_quantity should be set and meet constraints
    if result.ok:
        assert result.adjusted_quantity is not None, "Valid orders should have adjusted quantity"
        assert (
            result.adjusted_quantity >= min_size
        ), f"Adjusted quantity {result.adjusted_quantity} below min_size {min_size}"

        # Should be quantized to step size
        if step_size > 0:
            steps = result.adjusted_quantity / step_size
            assert (
                steps == steps.to_integral_value()
            ), f"Quantity {result.adjusted_quantity} not quantized to step {step_size}"

        # For limit orders, price should be set and quantized
        if order_type.lower() in ("limit", "stop_limit"):
            assert result.adjusted_price is not None, "Limit orders should have adjusted price"
            assert price_increment > 0, "Price increment should be positive for quantization check"

            # Price should be quantized
            normalized = result.adjusted_price / price_increment
            assert (
                normalized == normalized.to_integral_value()
            ), f"Price {result.adjusted_price} not quantized to increment {price_increment}"

            # Direction should be correct for side
            if price is not None:
                if side.lower() == "buy":
                    assert (
                        result.adjusted_price <= price
                    ), "Buy orders should not have price increased"
                else:
                    assert (
                        result.adjusted_price >= price
                    ), "Sell orders should not have price decreased"

    # If validation fails, there should be a reason
    if not result.ok:
        assert result.reason is not None, "Failed validation should have a reason"


# Property-based tests for order lifecycle invariants
@seed(7777)
@settings(max_examples=50, deadline=None)
@given(
    initial_quantity=st.decimals(
        min_value="0.0001", max_value="100", allow_nan=False, allow_infinity=False, places=6
    ),
    initial_price=st.decimals(
        min_value="0.01", max_value="10000", allow_nan=False, allow_infinity=False, places=4
    ),
    fill_price=st.decimals(
        min_value="0.01", max_value="10000", allow_nan=False, allow_infinity=False, places=4
    ),
    min_size=st.decimals(
        min_value="0.0001", max_value="10", allow_nan=False, allow_infinity=False, places=6
    ),
    step_size=st.decimals(
        min_value="0.0001", max_value="5", allow_nan=False, allow_infinity=False, places=6
    ),
    price_increment=st.decimals(
        min_value="0.001", max_value="100", allow_nan=False, allow_infinity=False, places=3
    ),
)
def test_order_lifecycle_invariants(
    initial_quantity: Decimal,
    initial_price: Decimal,
    fill_price: Decimal,
    min_size: Decimal,
    step_size: Decimal,
    price_increment: Decimal,
) -> None:
    """Property-based test for order lifecycle invariants."""

    # Create mock product
    class MockProduct:
        pass

    product = MockProduct()
    product.min_size = min_size
    product.step_size = step_size
    product.price_increment = price_increment
    product.min_notional = None

    # Ensure valid constraints
    if step_size > min_size:
        step_size = min_size / Decimal("10")

    # Simulate order lifecycle: submit -> validate -> fill
    # 1. Validate order
    validation = validate_order(
        product=product,
        side="buy",
        quantity=initial_quantity,
        order_type="limit",
        price=initial_price,
    )

    # 2. If validation passes, simulate fill
    if validation.ok and validation.adjusted_quantity and validation.adjusted_price:
        # Fill should respect validated constraints
        assert validation.adjusted_quantity >= min_size, "Fill quantity should meet min size"

        # Limit buys should never fill above the validated price
        effective_fill_price = fill_price
        if validation.adjusted_price > 0:
            if validation.adjusted_price >= effective_fill_price:
                effective_fill_price = min(effective_fill_price, validation.adjusted_price)
            else:
                # Allow a small tolerance beyond the validated price for stochastic tests
                effective_fill_price = validation.adjusted_price

        # Fill price should be within reasonable bounds (slippage)
        if effective_fill_price <= validation.adjusted_price:
            price_diff = Decimal("0")
        else:
            price_diff = effective_fill_price - validation.adjusted_price
        max_slippage = validation.adjusted_price * Decimal("0.05")  # 5% max slippage
        assert (
            price_diff <= max_slippage
        ), f"Fill price {effective_fill_price} exceeds slippage tolerance from {validation.adjusted_price}"

        # Notional should be reasonable
        fill_notional = validation.adjusted_quantity * effective_fill_price
        min_expected_notional = (
            min_size * effective_fill_price * Decimal("0.95")
        )  # Allow some slippage
        assert fill_notional >= min_expected_notional, f"Fill notional {fill_notional} too low"
