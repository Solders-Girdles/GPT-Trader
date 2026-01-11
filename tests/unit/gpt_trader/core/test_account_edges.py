from __future__ import annotations

from decimal import Decimal

from gpt_trader.core.account import CFMBalance, Position, UnifiedBalance


def _make_position(**overrides) -> Position:
    data = {
        "symbol": "BTC-USD",
        "quantity": Decimal("1"),
        "entry_price": Decimal("100"),
        "mark_price": Decimal("100"),
        "unrealized_pnl": Decimal("0"),
        "realized_pnl": Decimal("0"),
        "side": "long",
    }
    data.update(overrides)
    return Position(**data)


def test_liquidation_buffer_pct_none_cases() -> None:
    assert _make_position().liquidation_buffer_pct is None
    assert (
        _make_position(mark_price=None, liquidation_price=Decimal("80")).liquidation_buffer_pct
        is None
    )
    assert (
        _make_position(
            mark_price=Decimal("0"), liquidation_price=Decimal("80")
        ).liquidation_buffer_pct
        is None
    )


def test_liquidation_buffer_pct_long_and_short() -> None:
    long_pos = _make_position(
        side="long", mark_price=Decimal("100"), liquidation_price=Decimal("80")
    )
    short_pos = _make_position(
        side="short", mark_price=Decimal("100"), liquidation_price=Decimal("120")
    )

    assert long_pos.liquidation_buffer_pct == 20.0
    assert short_pos.liquidation_buffer_pct == 20.0


def test_position_is_futures_flag() -> None:
    assert _make_position().is_futures is False
    assert _make_position(product_type="FUTURE").is_futures is True


def test_cfm_balance_risk_and_utilization() -> None:
    balance = CFMBalance(
        futures_buying_power=Decimal("0"),
        total_usd_balance=Decimal("200"),
        available_margin=Decimal("0"),
        initial_margin=Decimal("50"),
        unrealized_pnl=Decimal("0"),
        daily_realized_pnl=Decimal("0"),
        liquidation_threshold=Decimal("0"),
        liquidation_buffer_amount=Decimal("0"),
        liquidation_buffer_percentage=49.9,
    )
    assert balance.is_at_risk is True
    assert balance.margin_utilization_pct == 25.0

    safe_balance = CFMBalance(
        futures_buying_power=Decimal("0"),
        total_usd_balance=Decimal("0"),
        available_margin=Decimal("0"),
        initial_margin=Decimal("50"),
        unrealized_pnl=Decimal("0"),
        daily_realized_pnl=Decimal("0"),
        liquidation_threshold=Decimal("0"),
        liquidation_buffer_amount=Decimal("0"),
        liquidation_buffer_percentage=50.0,
    )
    assert safe_balance.is_at_risk is False
    assert safe_balance.margin_utilization_pct == 0.0


def test_unified_balance_has_cfm() -> None:
    without_cfm = UnifiedBalance(
        spot_balance=Decimal("100"),
        cfm_balance=Decimal("0"),
        cfm_available_margin=Decimal("0"),
        cfm_buying_power=Decimal("0"),
        total_equity=Decimal("100"),
    )
    with_cfm = UnifiedBalance(
        spot_balance=Decimal("100"),
        cfm_balance=Decimal("1"),
        cfm_available_margin=Decimal("0"),
        cfm_buying_power=Decimal("0"),
        total_equity=Decimal("101"),
    )

    assert without_cfm.has_cfm is False
    assert with_cfm.has_cfm is True
