from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import MarketType
from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
)


class StubEventStore:
    def __init__(self) -> None:
        self.metrics: list[tuple] = []

    def append_metric(self, *args, **kwargs) -> None:
        self.metrics.append((args, kwargs))


class ProductStub:
    def __init__(self, market_type: MarketType) -> None:
        self.market_type = market_type


class ImpactStub:
    def __init__(self, estimated_impact_bps: Decimal, liquidity_sufficient: bool = True) -> None:
        self.estimated_impact_bps = estimated_impact_bps
        self.slippage_cost = Decimal("0")
        self.liquidity_sufficient = liquidity_sufficient
        self.recommended_slicing = None
        self.max_slice_size = None


def test_pre_trade_validator_kill_switch_blocks_trade():
    config = RiskConfig(kill_switch_enabled=True, enable_pre_trade_liq_projection=False)
    event_store = StubEventStore()
    validator = PreTradeValidator(config, event_store)

    product = ProductStub(MarketType.SPOT)

    with pytest.raises(ValidationError) as exc_info:
        validator.pre_trade_validate(
            "BTC-USD",
            "buy",
            quantity=Decimal("1"),
            price=Decimal("100"),
            product=product,
            equity=Decimal("10000"),
            current_positions={},
        )

    assert "Kill switch" in str(exc_info.value)
    assert event_store.metrics  # metric emitted


def test_pre_trade_validator_reduce_only_blocks_increase():
    config = RiskConfig(enable_pre_trade_liq_projection=False)
    event_store = StubEventStore()
    validator = PreTradeValidator(
        config,
        event_store,
        is_reduce_only_mode=lambda: True,
    )

    product = ProductStub(MarketType.SPOT)
    positions = {"BTC-USD": {"side": "long", "quantity": "1", "price": "20000"}}

    with pytest.raises(ValidationError) as exc_info:
        validator.pre_trade_validate(
            "BTC-USD",
            "buy",
            quantity=Decimal("0.5"),
            price=Decimal("200"),
            product=product,
            equity=Decimal("10000"),
            current_positions=positions,
        )

    assert "Reduce-only mode" in str(exc_info.value)
    assert event_store.metrics


def test_market_impact_guard_blocks_when_threshold_exceeded():
    config = RiskConfig(
        enable_market_impact_guard=True,
        max_market_impact_bps=5,
        enable_pre_trade_liq_projection=False,
    )
    event_store = StubEventStore()

    def impact_estimator(request):
        return ImpactStub(estimated_impact_bps=Decimal("10"))

    validator = PreTradeValidator(
        config,
        event_store,
        impact_estimator=impact_estimator,
    )

    product = ProductStub(MarketType.SPOT)

    with pytest.raises(ValidationError):
        validator.pre_trade_validate(
            "BTC-USD",
            "buy",
            quantity=Decimal("0.1"),
            price=Decimal("100"),
            product=product,
            equity=Decimal("10000"),
            current_positions={},
        )

    assert event_store.metrics
    args, kwargs = event_store.metrics[-1]
    if kwargs:
        payload = kwargs.get("metrics") or kwargs.get("metrics_payload")
    else:
        payload = args[1]
    assert payload["event_type"] == "market_impact_guard"


def test_validate_leverage_raises_when_cap_exceeded():
    config = RiskConfig(
        leverage_max_per_symbol={"BTC-USD": 1},
        enable_pre_trade_liq_projection=False,
    )
    validator = PreTradeValidator(config, StubEventStore())
    product = ProductStub(MarketType.PERPETUAL)

    with pytest.raises(ValidationError):
        validator.validate_leverage(
            "BTC-USD",
            quantity=Decimal("2"),
            price=Decimal("10000"),
            product=product,
            equity=Decimal("5000"),
        )


def test_validate_exposure_limits_enforces_symbol_cap():
    config = RiskConfig(
        max_position_pct_per_symbol=0.1,
        enable_pre_trade_liq_projection=False,
    )
    validator = PreTradeValidator(config, StubEventStore())

    with pytest.raises(ValidationError):
        validator.validate_exposure_limits(
            "BTC-USD",
            notional=Decimal("5000"),
            equity=Decimal("10000"),
            current_positions=None,
        )
