from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import Product, MarketType
from bot_v2.features.brokerages.coinbase.specs import (
    quantize_price_side_aware,
    validate_order,
    calculate_safe_position_size,
)


def make_product(symbol: str) -> Product:
    return Product(
        symbol=symbol,
        base_asset=symbol.split("-")[0],
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
    )


def test_side_aware_price_quantization_buy_sell():
    inc = Decimal("0.01")
    # BUY rounds down
    assert quantize_price_side_aware(Decimal("123.4567"), inc, "buy") == Decimal("123.45")
    # SELL rounds up
    assert quantize_price_side_aware(Decimal("123.4567"), inc, "sell") == Decimal("123.46")


def test_validate_order_enforces_min_size_with_buffer():
    p = make_product("BTC-PERP")
    # Intend size below min -> expect min_size buffer suggestion or rejection
    vr = validate_order(product=p, side="buy", qty=Decimal("0.0001"), order_type="limit", price=Decimal("50000"))
    # Validator does not auto-bump; it fails on min_size
    assert vr.ok is False
    assert vr.reason == "min_size"


def test_validate_order_enforces_min_notional():
    p = make_product("ETH-PERP")
    # Price * qty just below min_notional * 1.1 should be rejected with adjusted suggestion
    vr = validate_order(product=p, side="buy", qty=Decimal("0.001"), order_type="limit", price=Decimal("9000"))
    # Validator will suggest adjusted qty; does not auto-bump
    assert vr.ok is False
    assert vr.reason == "min_notional"
    assert vr.adjusted_qty is not None


def test_calculate_safe_position_size_bumps_to_clear_thresholds():
    p = make_product("SOL-PERP")
    # Intended tiny size should be bumped to clear min_size and min_notional with buffer
    safe = calculate_safe_position_size(product=p, side="buy", intended_qty=Decimal("0.0001"), ref_price=Decimal("100"))
    assert safe >= p.min_size
