from __future__ import annotations

from decimal import Decimal
from typing import Any

from hypothesis import given, seed, settings
from hypothesis import strategies as st
from src.bot_v2.features.brokerages.coinbase.client.base import CoinbaseClientBase
from src.bot_v2.features.brokerages.coinbase.specs import ProductSpec, SpecsService


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
