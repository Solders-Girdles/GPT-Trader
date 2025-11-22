"""
Scenario and configuration tests for pre-trade validation.
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


class TestPreTradeValidatorScenarios:
    """Test scenarios and configuration variations for PreTradeValidator."""

    def test_pre_trade_validate_comprehensive_success(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test comprehensive pre-trade validation success scenario."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Mock impact estimator to return acceptable values
        def impact_estimator(request):
            return Mock(estimated_impact_bps=Decimal("1"), liquidity_sufficient=True)

        validator._impact_estimator = impact_estimator

        # All validations should pass
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("0.05"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )
        except ValidationError:
            pytest.fail("Comprehensive validation should pass for valid trade")

    def test_pre_trade_validate_multiple_failure_modes(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test pre-trade validation catches multiple failure modes."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Enable kill switch - should fail immediately
        config = RiskConfig(kill_switch_enabled=True, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        with pytest.raises(ValidationError, match="Kill switch"):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("10"),  # Also excessive leverage
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("100"),
                current_positions={},  # Low equity
            )

        # Kill switch should be checked first, regardless of other issues
        metric = mock_event_store.get_last_metric()
        assert metric.get("event_type") == "kill_switch"

    def test_pre_trade_validate_spot_vs_perpetual_differences(
        self, conservative_risk_config, mock_event_store, btc_perpetual_product, eth_spot_product
    ):
        """Test pre-trade validation handles spot vs perpetual differences."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Mock impact estimator
        def impact_estimator(request):
            return Mock(estimated_impact_bps=Decimal("1"), liquidity_sufficient=True)

        validator._impact_estimator = impact_estimator

        # Both should be valid
        try:
            # Perpetual trade
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("0.05"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )

            # Spot trade
            validator.pre_trade_validate(
                "ETH-USD",
                "buy",
                quantity=Decimal("1"),
                price=Decimal("3000"),
                product=eth_spot_product,
                equity=Decimal("10000"),
                current_positions={},
            )
        except ValidationError:
            pytest.fail("Both spot and perpetual trades should be valid")

    def test_pre_trade_validate_aggressive_config(
        self, aggressive_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test pre-trade validation with aggressive risk configuration."""
        validator = PreTradeValidator(aggressive_risk_config, mock_event_store)

        # Mock impact estimator for aggressive config
        def impact_estimator(request):
            return Mock(
                estimated_impact_bps=Decimal("8"), liquidity_sufficient=True  # Within 10 bps limit
            )

        validator._impact_estimator = impact_estimator

        # Should allow larger positions
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("0.3"),  # 3x leverage allowed
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )
        except ValidationError:
            pytest.fail("Aggressive config should allow larger positions")

    def test_pre_trade_validate_emergency_config(
        self, emergency_risk_config, mock_event_store, btc_perpetual_product
    ):
        """Test pre-trade validation with emergency risk configuration."""
        validator = PreTradeValidator(emergency_risk_config, mock_event_store)

        # Even with kill switch disabled, should be very restrictive
        config = RiskConfig(
            kill_switch_enabled=False,
            leverage_max_per_symbol={"BTC-USD": 0.5},
            max_position_pct_per_symbol=0.01,
            enable_pre_trade_liq_projection=False,
        )
        validator = PreTradeValidator(config, mock_event_store)

        # Should block even moderate positions
        with pytest.raises(ValidationError):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                quantity=Decimal("0.2"),  # Exceeds 0.5x leverage
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={},
            )
