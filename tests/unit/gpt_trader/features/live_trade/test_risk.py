"""Tests for LiveRiskManager validation: leverage caps, liquidation buffer, and exposure limits."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import MarketType, Product
from gpt_trader.features.live_trade.risk import LiveRiskManager, ValidationError
from gpt_trader.features.live_trade.risk.config import RiskConfig


def make_product(symbol: str = "BTC-PERP", leverage_max: int = 10) -> Product:
    return Product(
        symbol=symbol,
        base_asset=symbol.split("-")[0],
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        step_size=Decimal("0.001"),
        min_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
        leverage_max=leverage_max,
    )


class TestLeverageCap:
    """Tests for maximum leverage enforcement."""

    @pytest.mark.perps
    def test_pre_trade_rejects_above_max_leverage(self) -> None:
        risk = LiveRiskManager(config=RiskConfig(max_leverage=5))
        product = make_product()

        equity = Decimal("10000")
        price = Decimal("51000")
        quantity = Decimal("20")  # notional 1,020,000 => 102x > 5x

        with pytest.raises(ValidationError):
            risk.pre_trade_validate(
                symbol="BTC-PERP",
                side="buy",
                quantity=quantity,
                price=price,
                product=product,
                equity=equity,
                current_positions=None,
            )


class TestLiquidationBuffer:
    """Tests for liquidation price buffer monitoring."""

    def test_buffer_breach_triggers_reduce_only(self) -> None:
        config = RiskConfig(min_liquidation_buffer_pct=0.15)
        rm = LiveRiskManager(config=config)

        equity = Decimal("20000")
        pos = {
            "quantity": Decimal("1"),
            "mark": Decimal("100"),
            "liquidation_price": Decimal("90"),  # 10% from mark < 15% threshold
        }

        triggered = rm.check_liquidation_buffer("BTC-PERP", pos, equity)
        assert triggered is True
        assert rm.positions["BTC-PERP"]["reduce_only"] is True

    def test_buffer_ok_when_above_minimum(self) -> None:
        config = RiskConfig(min_liquidation_buffer_pct=0.10)
        rm = LiveRiskManager(config=config)

        equity = Decimal("20000")
        pos = {
            "quantity": Decimal("2"),
            "mark": Decimal("100"),
            "liquidation_price": Decimal("85"),  # 15% buffer > 10% threshold
        }

        triggered = rm.check_liquidation_buffer("BTC-PERP", pos, equity)
        assert triggered is False


class TestTotalExposure:
    """Tests for portfolio exposure limits."""

    def test_total_exposure_breach_while_symbol_cap_ok(self) -> None:
        config = RiskConfig(
            max_leverage=10,
            min_liquidation_buffer_pct=0.1,
            max_position_pct_per_symbol=0.5,  # 50% per symbol allowed
            max_exposure_pct=0.8,  # 80% portfolio cap
            slippage_guard_bps=1_000_000,
        )
        rm = LiveRiskManager(config=config)
        product = make_product("BTC-PERP", leverage_max=20)

        equity = Decimal("100000")
        price = Decimal("50000")

        current_positions = {
            "ETH-PERP": {
                "quantity": Decimal("10"),
                "mark": Decimal("4000"),
                "price": Decimal("4000"),
            },  # 40k
            "SOL-PERP": {
                "quantity": Decimal("500"),
                "mark": Decimal("60"),
                "price": Decimal("60"),
            },  # 30k
        }  # total 70k (70%)

        # New order 15k (15%) keeps per-symbol under 50%, but total → 85% > 80%
        quantity = Decimal("0.3")  # 0.3 * 50k = 15k

        with pytest.raises(ValidationError, match="Total exposure .* would exceed cap"):
            rm.pre_trade_validate(
                symbol="BTC-PERP",
                side="buy",
                quantity=quantity,
                price=price,
                product=product,
                equity=equity,
                current_positions=current_positions,
            )
