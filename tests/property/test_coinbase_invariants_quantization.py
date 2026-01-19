from __future__ import annotations

from decimal import Decimal

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from gpt_trader.features.brokerages.coinbase.specs import ProductSpec, SpecsService


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
