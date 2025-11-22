"""
Market impact validation tests for pre-trade checks.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock
import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
)


class TestPreTradeValidatorMarketImpact:
    """Market impact validation tests for PreTradeValidator."""

    def test_market_impact_guard_disabled(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test market impact guard can be disabled."""
        config = RiskConfig(enable_market_impact_guard=False, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        # Should not call impact estimator or block trade
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("10"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError when market impact guard is disabled")

    def test_market_impact_guard_blocks_high_impact(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test market impact guard blocks high impact trades."""
        config = RiskConfig(
            enable_market_impact_guard=True,
            max_market_impact_bps=5,
            enable_pre_trade_liq_projection=False,
        )
        validator = PreTradeValidator(config, mock_event_store)

        # Mock impact estimator to return high impact
        def impact_estimator(request):
            assessment = Mock()
            assessment.estimated_impact_bps = Decimal("10")  # 10 bps > 5 bps limit
            assessment.liquidity_sufficient = False
            assessment.slippage_cost = Decimal("50.25")  # Mock slippage cost as Decimal
            assessment.recommended_slicing = None
            assessment.max_slice_size = None
            return assessment

        validator._impact_estimator = impact_estimator

        with pytest.raises(ValidationError, match="exceeds maximum market impact"):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("10"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )

        # Verify metric was emitted
        metric = mock_event_store.get_last_metric()
        assert metric.get("event_type") == "market_impact_guard"

    def test_market_impact_guard_allows_low_impact(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test market impact guard allows low impact trades."""
        config = RiskConfig(
            enable_market_impact_guard=True,
            max_market_impact_bps=10,
            enable_pre_trade_liq_projection=False,
        )
        validator = PreTradeValidator(config, mock_event_store)

        # Mock impact estimator to return low impact
        def impact_estimator(request):
            assessment = Mock()
            assessment.estimated_impact_bps = Decimal("5")  # 5 bps < 10 bps limit
            assessment.liquidity_sufficient = True
            assessment.slippage_cost = Decimal("25.50")  # Mock slippage cost as Decimal
            assessment.recommended_slicing = None
            assessment.max_slice_size = None
            return assessment

        validator._impact_estimator = impact_estimator

        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for low impact trade")

    def test_market_impact_guard_insufficient_liquidity(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test market impact guard blocks trades with insufficient liquidity."""
        config = RiskConfig(
            enable_market_impact_guard=True,
            max_market_impact_bps=20,
            enable_pre_trade_liq_projection=False,
        )
        validator = PreTradeValidator(config, mock_event_store)

        # Mock impact estimator to return insufficient liquidity
        def impact_estimator(request):
            assessment = Mock()
            assessment.estimated_impact_bps = Decimal("5")  # Within impact limit
            assessment.liquidity_sufficient = False  # But insufficient liquidity
            assessment.slippage_cost = Decimal("75.00")  # Mock slippage cost as Decimal
            assessment.recommended_slicing = None
            assessment.max_slice_size = None
            return assessment

        validator._impact_estimator = impact_estimator

        with pytest.raises(ValidationError, match="insufficient liquidity"):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("10"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )

    def test_pre_trade_validate_emits_appropriate_metrics(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test pre-trade validation emits appropriate metrics."""
        config = RiskConfig(
            enable_market_impact_guard=True,
            max_market_impact_bps=5,
            enable_pre_trade_liq_projection=False,
        )
        validator = PreTradeValidator(config, mock_event_store)

        # Mock impact estimator to trigger metric
        def impact_estimator(request):
            return Mock(
                estimated_impact_bps=Decimal("10"), liquidity_sufficient=False  # Triggers guard
            )

        validator._impact_estimator = impact_estimator

        with pytest.raises(ValidationError):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("10"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )

        # Should emit market impact guard metric
        metric = mock_event_store.get_last_metric()
        assert metric.get("event_type") == "market_impact_guard"
        assert "estimated_impact_bps" in metric or "impact" in str(metric).lower()
