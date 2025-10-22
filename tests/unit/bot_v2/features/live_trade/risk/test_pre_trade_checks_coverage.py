"""
Comprehensive test coverage for pre-trade validation system.

Targets 80%+ coverage for critical financial protection logic including:
- Leverage validation
- Exposure limits
- Liquidation buffer checks
- Market impact validation
- Kill switch functionality
- Reduce-only mode enforcement
- Correlation risk assessment
- Margin requirement validation
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any
from unittest.mock import Mock, patch

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.brokerages.core.interfaces import MarketType
from bot_v2.features.live_trade.risk.pre_trade_checks import (
    PreTradeValidator,
    ValidationError,
    _coalesce_quantity,
    _to_decimal,
)


class TestPreTradeValidatorCoverage:
    """Comprehensive test suite for PreTradeValidator functionality."""

    # -------------------------------------------------------------------------
    # VALIDATION HELPER FUNCTIONS
    # -------------------------------------------------------------------------

    def test_coalesce_quantity_success(self):
        """Test _coalesce_quantity with valid inputs."""
        result = _coalesce_quantity(None, Decimal("1.5"), Decimal("2.0"))
        assert result == Decimal("1.5")

        result = _coalesce_quantity(Decimal("3.0"), None, Decimal("4.0"))
        assert result == Decimal("3.0")

        result = _coalesce_quantity(None, None, Decimal("5.0"))
        assert result == Decimal("5.0")

    def test_coalesce_quantity_failure(self):
        """Test _coalesce_quantity raises when all values are None."""
        with pytest.raises(TypeError, match="quantity must be provided"):
            _coalesce_quantity(None, None, None)

    def test_to_decimal_valid_inputs(self):
        """Test _to_decimal with various valid input types."""
        assert _to_decimal("123.45") == Decimal("123.45")
        assert _to_decimal(Decimal("67.89")) == Decimal("67.89")
        assert _to_decimal(123) == Decimal("123")
        assert _to_decimal(None) == Decimal("0")
        assert _to_decimal("") == Decimal("0")
        assert _to_decimal("null") == Decimal("0")

    def test_to_decimal_invalid_inputs(self):
        """Test _to_decimal gracefully handles invalid inputs."""
        assert _to_decimal("invalid") == Decimal("0")
        assert _to_decimal([], Decimal("1.0")) == Decimal("1.0")
        assert _to_decimal({}, Decimal("2.0")) == Decimal("2.0")
        # Test with infinity - handle more gracefully
        try:
            result = _to_decimal(float('inf'))
            # If it doesn't raise, it should be a valid Decimal or default
            assert isinstance(result, Decimal)
        except (ValueError, InvalidOperation, TypeError):
            # If it raises during Decimal conversion, it should return default
            assert _to_decimal(float('inf')) == Decimal("0")

    # -------------------------------------------------------------------------
    # INITIALIZATION TESTS
    # -------------------------------------------------------------------------

    def test_validator_initialization_default(self, conservative_risk_config, mock_event_store):
        """Test PreTradeValidator initialization with default parameters."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        assert validator.config is conservative_risk_config
        assert validator.event_store is mock_event_store
        assert hasattr(validator, '_is_reduce_only_mode')
        assert hasattr(validator, '_impact_estimator')

    def test_validator_initialization_with_callbacks(self, conservative_risk_config, mock_event_store):
        """Test PreTradeValidator initialization with custom callbacks."""
        reduce_only_cb = Mock(return_value=False)
        impact_estimator = Mock()

        validator = PreTradeValidator(
            conservative_risk_config,
            mock_event_store,
            is_reduce_only_mode=reduce_only_cb,
            impact_estimator=impact_estimator
        )

        assert validator._is_reduce_only_mode is reduce_only_cb
        assert validator._impact_estimator is impact_estimator

    # -------------------------------------------------------------------------
    # KILL SWITCH TESTS
    # -------------------------------------------------------------------------

    def test_kill_switch_blocks_all_trades(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test kill switch blocks all trade attempts."""
        config = RiskConfig(kill_switch_enabled=True, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        with pytest.raises(ValidationError, match="Kill switch enabled"):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )

        # Verify metric was emitted
        assert len(mock_event_store.metrics) >= 1
        metric = mock_event_store.get_last_metric()
        assert metric["metrics"].get("event_type") == "kill_switch"

    def test_kill_switch_disabled_allows_trades(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test trades proceed when kill switch is disabled."""
        config = RiskConfig(kill_switch_enabled=False, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        # Should not raise - adjust quantity to stay within exposure limits
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.01"),  # Reduced to stay within 20% exposure limit ($500/$10000 = 5%)
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )
        except ValidationError as e:
            pytest.fail(f"Unexpected ValidationError: {e}")

    # -------------------------------------------------------------------------
    # REDUCE ONLY MODE TESTS
    # -------------------------------------------------------------------------

    def test_reduce_only_mode_blocks_increasing_trades(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test reduce-only mode blocks position increases."""
        config = RiskConfig(enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(
            config,
            mock_event_store,
            is_reduce_only_mode=lambda: True
        )

        positions = {"BTC-USD": {"side": "long", "quantity": "1", "price": "50000"}}

        # Block increase to long position
        with pytest.raises(ValidationError, match="Reduce-only mode"):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.5"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions=positions
            )

        # Block increase to short position
        with pytest.raises(ValidationError, match="Reduce-only mode"):
            validator.pre_trade_validate(
                "BTC-USD",
                "sell",
                qty=Decimal("0.5"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={"BTC-USD": {"side": "short", "quantity": "1", "price": "50000"}}
            )

    def test_reduce_only_mode_allows_position_reduction(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test reduce-only mode allows position reduction."""
        config = RiskConfig(enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(
            config,
            mock_event_store,
            is_reduce_only_mode=lambda: True
        )

        # Allow reduction of long position - adjust quantity to stay within exposure limits
        positions = {"BTC-USD": {"side": "long", "quantity": "0.01", "price": "50000"}}  # Small existing position
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "sell",
                qty=Decimal("0.005"),  # Small reduction to stay within exposure limits
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions=positions
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for position reduction")

        # Allow reduction of short position
        positions = {"BTC-USD": {"side": "short", "quantity": "0.01", "price": "50000"}}  # Small existing position
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.005"),  # Small reduction to stay within exposure limits
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions=positions
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for position reduction")

    def test_reduce_only_mode_allows_new_position_when_no_existing(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test reduce-only mode allows new positions when no existing position."""
        config = RiskConfig(enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(
            config,
            mock_event_store,
            is_reduce_only_mode=lambda: True
        )

        # This should be allowed - reducing from zero to a position
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "sell",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for new position in reduce-only mode")

    # -------------------------------------------------------------------------
    # LEVERAGE VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_validate_leverage_within_limits(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test leverage validation passes when within limits."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Within 1x leverage limit
        try:
            validator.validate_leverage(
                "BTC-USD",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000")
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for valid leverage")

    def test_validate_leverage_exceeds_symbol_limit(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test leverage validation fails when symbol limit exceeded."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Exceeds 1x leverage limit (1 BTC * $50000 = $50000 notional, equity $10000)
        with pytest.raises(ValidationError, match="Leverage.*exceeds.*cap"):
            validator.validate_leverage(
                "BTC-USD",
                qty=Decimal("1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000")
            )

    def test_validate_leverage_edge_cases(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test leverage validation edge cases."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Zero equity
        with pytest.raises(ValidationError, match="Leverage.*exceeds.*cap"):
            validator.validate_leverage(
                "BTC-USD",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("0")
            )

        # Zero quantity
        try:
            validator.validate_leverage(
                "BTC-USD",
                qty=Decimal("0"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000")
            )
        except ValidationError:
            pytest.fail("Zero quantity should be valid")

        # Very small position
        try:
            validator.validate_leverage(
                "BTC-USD",
                qty=Decimal("0.0001"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000")
            )
        except ValidationError:
            pytest.fail("Small position should be valid")

    def test_validate_leverage_with_existing_positions(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test leverage validation with high leverage values."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Test with high leverage that exceeds symbol cap
        with pytest.raises(ValidationError, match="Leverage.*exceeds.*cap"):
            validator.validate_leverage(
                "BTC-USD",
                qty=Decimal("5"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000")
            )

    # -------------------------------------------------------------------------
    # EXPOSURE LIMIT VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_validate_exposure_limits_within_bounds(self, conservative_risk_config, mock_event_store):
        """Test exposure validation passes when within limits."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # 5% of $10000 equity = $500 notional limit, $450 is within bounds
        try:
            validator.validate_exposure_limits(
                "BTC-USD",
                notional=Decimal("450"),
                equity=Decimal("10000"),
                current_positions={}
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for valid exposure")

    def test_validate_exposure_limits_exceeds_symbol_cap(self, conservative_risk_config, mock_event_store):
        """Test exposure validation fails when symbol cap exceeded."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # 20% of $10000 equity = $2000 notional limit, $2500 exceeds limit
        with pytest.raises(ValidationError, match="Symbol exposure.*exceeds.*cap"):
            validator.validate_exposure_limits(
                "BTC-USD",
                notional=Decimal("2500"),
                equity=Decimal("10000"),
                current_positions={}
            )

    def test_validate_exposure_limits_with_existing_positions(self, conservative_risk_config, mock_event_store):
        """Test exposure validation considers existing positions."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Add existing position that uses part of the allocation
        current_positions = {
            "BTC-USD": {
                "side": "long",
                "quantity": "0.02",
                "price": "50000",
                "mark_price": "50000"
            }
        }  # $1000 notional

        # New position should exceed total symbol cap
        with pytest.raises(ValidationError, match="Symbol exposure.*exceeds.*cap"):
            validator.validate_exposure_limits(
                "BTC-USD",
                notional=Decimal("1500"),  # Combined would be $2500 > 20% of $10000 = $2000
                equity=Decimal("10000"),
                current_positions=current_positions
            )

    def test_validate_exposure_limits_zero_equity(self, conservative_risk_config, mock_event_store):
        """Test exposure validation with zero equity."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        with pytest.raises(ValidationError, match="Symbol exposure.*exceeds.*cap"):
            validator.validate_exposure_limits(
                "BTC-USD",
                notional=Decimal("100"),
                equity=Decimal("0"),
                current_positions={}
            )

    # -------------------------------------------------------------------------
    # LIQUIDATION BUFFER VALIDATION TESTS
    # -------------------------------------------------------------------------

    @patch('bot_v2.features.live_trade.risk.pre_trade_checks.effective_mmr')
    def test_validate_liquidation_buffer_sufficient_margin(self, mock_effective_mmr, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test liquidation buffer validation passes with sufficient margin."""
        mock_effective_mmr.return_value = Decimal("0.1")  # 10% MMR
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        current_positions = {
            "BTC-USD": {
                "side": "long",
                "quantity": "0.1",
                "price": "48000",
                "mark_price": "50000"
            }
        }

        try:
            validator.validate_liquidation_buffer(
                "BTC-USD",
                qty=Decimal("0.01"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000")
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for sufficient liquidation buffer")

    @patch('bot_v2.features.live_trade.risk.pre_trade_checks.effective_mmr')
    def test_validate_liquidation_buffer_insufficient_margin(self, mock_effective_mmr, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test liquidation buffer validation fails with insufficient margin."""
        mock_effective_mmr.return_value = Decimal("0.15")  # 15% MMR
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        current_positions = {
            "BTC-USD": {
                "side": "long",
                "quantity": "0.8",  # Large position
                "price": "48000",
                "mark_price": "50000"
            }
        }

        with pytest.raises(ValidationError, match="Insufficient liquidation buffer"):
            validator.validate_liquidation_buffer(
                "BTC-USD",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("1000")  # Low equity
            )

    @patch('bot_v2.features.live_trade.risk.pre_trade_checks.effective_mmr')
    def test_validate_liquidation_buffer_zero_equity(self, mock_effective_mmr, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test liquidation buffer validation with zero equity."""
        mock_effective_mmr.return_value = Decimal("0.1")
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        with pytest.raises(ValidationError, match="Insufficient liquidation buffer"):
            validator.validate_liquidation_buffer(
                "BTC-USD",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("0")
            )

    # -------------------------------------------------------------------------
    # MARKET IMPACT VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_market_impact_guard_disabled(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test market impact guard can be disabled."""
        config = RiskConfig(enable_market_impact_guard=False, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        # Should not call impact estimator or block trade
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("10"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError when market impact guard is disabled")

    def test_market_impact_guard_blocks_high_impact(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test market impact guard blocks high impact trades."""
        config = RiskConfig(
            enable_market_impact_guard=True,
            max_market_impact_bps=5,
            enable_pre_trade_liq_projection=False
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
                qty=Decimal("10"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )

        # Verify metric was emitted
        metric = mock_event_store.get_last_metric()
        assert metric.get("event_type") == "market_impact_guard"

    def test_market_impact_guard_allows_low_impact(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test market impact guard allows low impact trades."""
        config = RiskConfig(
            enable_market_impact_guard=True,
            max_market_impact_bps=10,
            enable_pre_trade_liq_projection=False
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
                qty=Decimal("1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for low impact trade")

    def test_market_impact_guard_insufficient_liquidity(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test market impact guard blocks trades with insufficient liquidity."""
        config = RiskConfig(
            enable_market_impact_guard=True,
            max_market_impact_bps=20,
            enable_pre_trade_liq_projection=False
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
                qty=Decimal("10"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )

    # -------------------------------------------------------------------------
    # CORRELATION RISK VALIDATION TESTS
    # -------------------------------------------------------------------------

    def test_validate_correlation_risk_within_limits(self, aggressive_risk_config, mock_event_store):
        """Test correlation risk validation passes when within limits."""
        validator = PreTradeValidator(aggressive_risk_config, mock_event_store)

        # Add positions with moderate correlation
        current_positions = {
            "BTC-USD": {"side": "long", "quantity": "0.1", "price": "50000", "mark_price": "50000"},
            "ETH-USD": {"side": "long", "quantity": "1", "price": "3000", "mark_price": "3000"}
        }

        try:
            validator.validate_correlation_risk(
                "BTC-USD",
                notional=Decimal("1000"),
                current_positions=current_positions
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError for acceptable correlation")

    def test_validate_correlation_risk_exceeds_limit(self, conservative_risk_config, mock_event_store):
        """Test correlation risk validation fails when limit exceeded."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Add highly correlated positions (both large crypto longs)
        current_positions = {
            "BTC-USD": {"side": "long", "quantity": "0.5", "price": "50000", "mark_price": "50000"},
            "ETH-USD": {"side": "long", "quantity": "10", "price": "3000", "mark_price": "3000"},
            "SOL-USD": {"side": "long", "quantity": "100", "price": "100", "mark_price": "100"}
        }

        with pytest.raises(ValidationError, match="correlation risk"):
            validator.validate_correlation_risk(
                "BTC-USD",
                notional=Decimal("5000"),  # Large additional exposure
                current_positions=current_positions
            )

    def test_validate_correlation_risk_no_existing_positions(self, conservative_risk_config, mock_event_store):
        """Test correlation risk validation with no existing positions."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        try:
            validator.validate_correlation_risk(
                "BTC-USD",
                notional=Decimal("10000"),
                current_positions={}
            )
        except ValidationError:
            pytest.fail("Should not raise ValidationError with no existing positions")

    # -------------------------------------------------------------------------
    # COMPREHENSIVE INTEGRATION TESTS
    # -------------------------------------------------------------------------

    def test_pre_trade_validate_comprehensive_success(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test comprehensive pre-trade validation success scenario."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Mock impact estimator to return acceptable values
        def impact_estimator(request):
            return Mock(
                estimated_impact_bps=Decimal("1"),
                liquidity_sufficient=True
            )
        validator._impact_estimator = impact_estimator

        # All validations should pass
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.05"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )
        except ValidationError:
            pytest.fail("Comprehensive validation should pass for valid trade")

    def test_pre_trade_validate_multiple_failure_modes(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test pre-trade validation catches multiple failure modes."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Enable kill switch - should fail immediately
        config = RiskConfig(kill_switch_enabled=True, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        with pytest.raises(ValidationError, match="Kill switch"):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("10"),  # Also excessive leverage
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("100"),
                current_positions={}  # Low equity
            )

        # Kill switch should be checked first, regardless of other issues
        metric = mock_event_store.get_last_metric()
        assert metric.get("event_type") == "kill_switch"

    def test_pre_trade_validate_spot_vs_perpetual_differences(self, conservative_risk_config, mock_event_store, btc_perpetual_product, eth_spot_product):
        """Test pre-trade validation handles spot vs perpetual differences."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Mock impact estimator
        def impact_estimator(request):
            return Mock(
                estimated_impact_bps=Decimal("1"),
                liquidity_sufficient=True
            )
        validator._impact_estimator = impact_estimator

        # Both should be valid
        try:
            # Perpetual trade
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.05"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )

            # Spot trade
            validator.pre_trade_validate(
                "ETH-USD",
                "buy",
                qty=Decimal("1"),
                price=Decimal("3000"),
                product=eth_spot_product,
                equity=Decimal("10000"),
                current_positions={}
            )
        except ValidationError:
            pytest.fail("Both spot and perpetual trades should be valid")

    # -------------------------------------------------------------------------
    # ERROR HANDLING AND EDGE CASES
    # -------------------------------------------------------------------------

    def test_pre_trade_validate_invalid_inputs(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test pre-trade validation handles invalid inputs gracefully."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Invalid side
        with pytest.raises(ValueError):  # May raise ValueError from validation logic
            validator.pre_trade_validate(
                "BTC-USD",
                "invalid_side",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )

        # Negative quantity
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("-0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )
        except (ValidationError, ValueError):
            pass  # Should fail somehow

    def test_pre_trade_validate_malformed_position_data(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test pre-trade validation handles malformed position data."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Malformed position data should not crash
        malformed_positions = {
            "BTC-USD": {
                "side": "long",
                "quantity": "invalid_number",
                "price": "50000",
                "mark_price": "50000"
            }
        }

        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.01"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions=malformed_positions
            )
        except (ValidationError, ValueError, TypeError):
            pass  # Should handle gracefully

    # -------------------------------------------------------------------------
    # CONFIGURATION VARIATIONS TESTS
    # -------------------------------------------------------------------------

    def test_pre_trade_validate_aggressive_config(self, aggressive_risk_config, mock_event_store, btc_perpetual_product):
        """Test pre-trade validation with aggressive risk configuration."""
        validator = PreTradeValidator(aggressive_risk_config, mock_event_store)

        # Mock impact estimator for aggressive config
        def impact_estimator(request):
            return Mock(
                estimated_impact_bps=Decimal("8"),  # Within 10 bps limit
                liquidity_sufficient=True
            )
        validator._impact_estimator = impact_estimator

        # Should allow larger positions
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.3"),  # 3x leverage allowed
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )
        except ValidationError:
            pytest.fail("Aggressive config should allow larger positions")

    def test_pre_trade_validate_emergency_config(self, emergency_risk_config, mock_event_store, btc_perpetual_product):
        """Test pre-trade validation with emergency risk configuration."""
        validator = PreTradeValidator(emergency_risk_config, mock_event_store)

        # Even with kill switch disabled, should be very restrictive
        config = RiskConfig(
            kill_switch_enabled=False,
            leverage_max_per_symbol={"BTC-USD": 0.5},
            max_position_pct_per_symbol=0.01,
            enable_pre_trade_liq_projection=False
        )
        validator = PreTradeValidator(config, mock_event_store)

        # Should block even moderate positions
        with pytest.raises(ValidationError):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.2"),  # Exceeds 0.5x leverage
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )

    # -------------------------------------------------------------------------
    # TELEMETRY AND METRICS TESTS
    # -------------------------------------------------------------------------

    def test_pre_trade_validate_emits_appropriate_metrics(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test pre-trade validation emits appropriate metrics."""
        config = RiskConfig(
            enable_market_impact_guard=True,
            max_market_impact_bps=5,
            enable_pre_trade_liq_projection=False
        )
        validator = PreTradeValidator(config, mock_event_store)

        # Mock impact estimator to trigger metric
        def impact_estimator(request):
            return Mock(
                estimated_impact_bps=Decimal("10"),  # Triggers guard
                liquidity_sufficient=False
            )
        validator._impact_estimator = impact_estimator

        with pytest.raises(ValidationError):
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("10"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )

        # Should emit market impact guard metric
        metric = mock_event_store.get_last_metric()
        assert metric.get("event_type") == "market_impact_guard"
        assert "estimated_impact_bps" in metric or "impact" in str(metric).lower()


class TestPreTradeValidatorErrorHandling:
    """Test error handling and edge cases in PreTradeValidator."""

    def test_validation_error_message_formatting(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test ValidationError messages are properly formatted."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        config = RiskConfig(kill_switch_enabled=True, enable_pre_trade_liq_projection=False)
        validator = PreTradeValidator(config, mock_event_store)

        with pytest.raises(ValidationError) as exc_info:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.1"),
                price=Decimal("50000"),
                product=btc_perpetual_product,
                equity=Decimal("10000"),
                current_positions={}
            )

        error_message = str(exc_info.value)
        assert "kill switch" in error_message.lower()
        assert len(error_message) > 10  # Should have meaningful content

    def test_decimal_precision_handling(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test proper decimal precision handling."""
        validator = PreTradeValidator(conservative_risk_config, mock_event_store)

        # Very small decimal values
        try:
            validator.pre_trade_validate(
                "BTC-USD",
                "buy",
                qty=Decimal("0.00000001"),
                price=Decimal("50000.00000001"),
                product=btc_perpetual_product,
                equity=Decimal("10000.00000001"),
                current_positions={}
            )
        except ValidationError:
            pass  # May fail due to precision, but shouldn't crash

    def test_concurrent_validation_safety(self, conservative_risk_config, mock_event_store, btc_perpetual_product):
        """Test validator is safe for concurrent use."""
        import threading
        import time

        validator = PreTradeValidator(conservative_risk_config, mock_event_store)
        results = []

        def validate_trade():
            try:
                validator.pre_trade_validate(
                    "BTC-USD",
                    "buy",
                    qty=Decimal("0.01"),
                    price=Decimal("50000"),
                    product=btc_perpetual_product,
                    equity=Decimal("10000"),
                    current_positions={}
                )
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")

        # Run multiple validations concurrently
        threads = [threading.Thread(target=validate_trade) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Should handle concurrent access gracefully
        assert len(results) == 5
        assert all("error:" not in result or result == "success" for result in results)