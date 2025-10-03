"""Unit tests for OrderValidationPipeline."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from bot_v2.errors import ValidationError
from bot_v2.features.brokerages.core.interfaces import (
    MarketType,
    OrderSide,
    OrderType,
    Product,
    Quote,
    TimeInForce,
)
from bot_v2.features.live_trade.advanced_execution_models.models import (
    NormalizedOrderRequest,
    OrderConfig,
)
from bot_v2.features.live_trade.order_validation_pipeline import (
    OrderValidationPipeline,
    ValidationResult,
)
from bot_v2.features.live_trade.risk import PositionSizingAdvice
from bot_v2.features.live_trade.stop_trigger_manager import StopTriggerManager
from bot_v2.features.live_trade.dynamic_sizing_helper import DynamicSizingHelper


@pytest.fixture
def product() -> Product:
    """Baseline product for validation tests."""
    return Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=5,
        contract_size=Decimal("1"),
        funding_rate=Decimal("0.0001"),
        next_funding_time=None,
    )


@pytest.fixture
def quote() -> Quote:
    """Market quote fixture."""
    return Quote(
        symbol="BTC-PERP",
        bid=Decimal("100"),
        ask=Decimal("101"),
        last=Decimal("100.5"),
        ts=datetime.utcnow(),
    )


@pytest.fixture
def sizing_helper() -> DynamicSizingHelper:
    """Create sizing helper stub with predictable behaviour."""
    helper = MagicMock(spec=DynamicSizingHelper)
    helper.maybe_apply_position_sizing.return_value = None
    helper.determine_reference_price.return_value = Decimal("100")
    helper.estimate_equity.return_value = Decimal("10000")
    return helper  # type: ignore[return-value]


@pytest.fixture
def stop_manager() -> StopTriggerManager:
    """Use real stop trigger manager for validation hooks."""
    return StopTriggerManager(config=OrderConfig())


def build_request(
    *,
    product: Product | None,
    quote: Quote | None = None,
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    quantity: Decimal = Decimal("1"),
    limit_price: Decimal | None = Decimal("100"),
    stop_price: Decimal | None = None,
    post_only: bool = False,
    reduce_only: bool = False,
) -> NormalizedOrderRequest:
    """Helper to create normalized requests for tests."""
    return NormalizedOrderRequest(
        client_id="client-123",
        symbol="BTC-PERP",
        side=side,
        quantity=quantity,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        time_in_force=TimeInForce.GTC,
        reduce_only=reduce_only,
        post_only=post_only,
        leverage=None,
        product=product,
        quote=quote,
    )


class TestOrderValidationPipeline:
    """Validate pipeline behaviour across scenarios."""

    def test_validate_success_without_adjustments(
        self,
        product: Product,
        sizing_helper: DynamicSizingHelper,
        stop_manager: StopTriggerManager,
    ) -> None:
        pipeline = OrderValidationPipeline(
            config=OrderConfig(),
            sizing_helper=sizing_helper,
            stop_trigger_manager=stop_manager,
            risk_manager=None,
        )
        request = build_request(product=product)

        result = pipeline.validate(request)

        assert result.ok
        assert request.quantity == Decimal("1")
        assert request.limit_price == Decimal("100")

    def test_dynamic_sizing_rejects_zero_quantity(
        self,
        product: Product,
        sizing_helper: DynamicSizingHelper,
        stop_manager: StopTriggerManager,
    ) -> None:
        sizing_helper.maybe_apply_position_sizing.return_value = PositionSizingAdvice(
            symbol="BTC-PERP",
            side="buy",
            target_notional=Decimal("0"),
            target_quantity=Decimal("0"),
            reduce_only=False,
            reason="no_notional",
        )
        pipeline = OrderValidationPipeline(
            config=OrderConfig(),
            sizing_helper=sizing_helper,
            stop_trigger_manager=stop_manager,
            risk_manager=None,
        )
        request = build_request(product=product)

        result = pipeline.validate(request)

        assert result.failed
        assert result.rejection_reason == "position_sizing"

    def test_dynamic_sizing_updates_quantity_and_reduce_only(
        self,
        product: Product,
        sizing_helper: DynamicSizingHelper,
        stop_manager: StopTriggerManager,
    ) -> None:
        sizing_helper.maybe_apply_position_sizing.return_value = PositionSizingAdvice(
            symbol="BTC-PERP",
            side="buy",
            target_notional=Decimal("2000"),
            target_quantity=Decimal("0.2"),
            reduce_only=True,
            reason="dynamic_sizing",
        )
        pipeline = OrderValidationPipeline(
            config=OrderConfig(),
            sizing_helper=sizing_helper,
            stop_trigger_manager=stop_manager,
            risk_manager=None,
        )
        request = build_request(product=product)

        result = pipeline.validate(request)

        assert result.ok
        assert request.quantity == Decimal("0.2")
        assert request.reduce_only is True

    def test_post_only_cross_rejected(
        self,
        product: Product,
        quote: Quote,
        sizing_helper: DynamicSizingHelper,
        stop_manager: StopTriggerManager,
    ) -> None:
        pipeline = OrderValidationPipeline(
            config=OrderConfig(reject_on_cross=True),
            sizing_helper=sizing_helper,
            stop_trigger_manager=stop_manager,
            risk_manager=None,
        )
        request = build_request(
            product=product,
            quote=quote,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("101"),
            post_only=True,
        )

        result = pipeline.validate(request)

        assert result.failed
        assert result.rejection_reason == "post_only_cross"
        assert result.post_only_rejection is True

    def test_quantization_adjusts_limit_price(
        self,
        product: Product,
        sizing_helper: DynamicSizingHelper,
        stop_manager: StopTriggerManager,
    ) -> None:
        pipeline = OrderValidationPipeline(
            config=OrderConfig(),
            sizing_helper=sizing_helper,
            stop_trigger_manager=stop_manager,
            risk_manager=None,
        )
        request = build_request(
            product=product,
            limit_price=Decimal("100.1234"),
            order_type=OrderType.LIMIT,
        )

        result = pipeline.validate(request)

        assert result.ok
        assert request.limit_price == Decimal("100.12")

    def test_quantization_failure_returns_reason(
        self,
        sizing_helper: DynamicSizingHelper,
        stop_manager: StopTriggerManager,
    ) -> None:
        high_min_product = Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("1000"),
            price_increment=Decimal("0.01"),
            leverage_max=5,
            contract_size=Decimal("1"),
            funding_rate=Decimal("0.0001"),
            next_funding_time=None,
        )
        pipeline = OrderValidationPipeline(
            config=OrderConfig(),
            sizing_helper=sizing_helper,
            stop_trigger_manager=stop_manager,
            risk_manager=None,
        )
        request = build_request(
            product=high_min_product,
            limit_price=Decimal("50"),
            quantity=Decimal("0.1"),
        )

        result = pipeline.validate(request)

        assert result.failed
        assert result.rejection_reason == "min_notional"

    def test_risk_validation_failure(
        self,
        product: Product,
        sizing_helper: DynamicSizingHelper,
        stop_manager: StopTriggerManager,
    ) -> None:
        risk_manager = MagicMock()
        risk_manager.pre_trade_validate.side_effect = ValidationError("Max exposure exceeded")
        risk_manager.positions = {}
        pipeline = OrderValidationPipeline(
            config=OrderConfig(),
            sizing_helper=sizing_helper,
            stop_trigger_manager=stop_manager,
            risk_manager=risk_manager,
        )
        request = build_request(product=product, order_type=OrderType.MARKET, limit_price=None)

        result = pipeline.validate(request)

        assert result.failed
        assert result.rejection_reason == "risk"

    def test_stop_validation_failure(
        self,
        product: Product,
        sizing_helper: DynamicSizingHelper,
        stop_manager: StopTriggerManager,
    ) -> None:
        pipeline = OrderValidationPipeline(
            config=OrderConfig(),
            sizing_helper=sizing_helper,
            stop_trigger_manager=stop_manager,
            risk_manager=None,
        )
        request = build_request(
            product=product,
            order_type=OrderType.STOP,
            stop_price=None,
        )

        result = pipeline.validate(request)

        assert result.failed
        assert result.rejection_reason == "invalid_stop"

    def test_success_returns_final_values(
        self,
        product: Product,
        sizing_helper: DynamicSizingHelper,
        stop_manager: StopTriggerManager,
    ) -> None:
        sizing_helper.maybe_apply_position_sizing.return_value = PositionSizingAdvice(
            symbol="BTC-PERP",
            side="buy",
            target_notional=Decimal("2000"),
            target_quantity=Decimal("0.2"),
            reduce_only=False,
            reason=None,
        )
        pipeline = OrderValidationPipeline(
            config=OrderConfig(),
            sizing_helper=sizing_helper,
            stop_trigger_manager=stop_manager,
            risk_manager=None,
        )
        request = build_request(product=product)

        result = pipeline.validate(request)

        assert result.ok
        assert isinstance(result, ValidationResult)
        assert result.quantity == Decimal("0.2")
        assert result.limit_price == request.limit_price
        assert result.stop_price == request.stop_price
