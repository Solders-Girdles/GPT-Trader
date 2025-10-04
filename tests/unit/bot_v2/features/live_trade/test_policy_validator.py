"""
Unit tests for PolicyValidator.

Tests order validation logic against trading policies.
"""

from decimal import Decimal

import pytest

from bot_v2.features.live_trade.capability_registry import OrderCapability, OrderTypeSupport
from bot_v2.features.live_trade.policy_validator import PolicyValidator


class TestTradingEnabledValidation:
    """Tests for trading enabled/disabled validation."""

    def test_trading_enabled_passes(self):
        """Test validation passes when trading is enabled."""
        allowed, reason = PolicyValidator.validate_trading_enabled(trading_enabled=True)

        assert allowed is True
        assert reason == ""

    def test_trading_disabled_fails(self):
        """Test validation fails when trading is disabled."""
        allowed, reason = PolicyValidator.validate_trading_enabled(trading_enabled=False)

        assert allowed is False
        assert "disabled" in reason.lower()


class TestCapabilitySupportValidation:
    """Tests for capability support level validation."""

    def test_supported_capability_passes(self):
        """Test validation passes for SUPPORTED capability."""
        cap = OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED)

        allowed, reason = PolicyValidator.validate_capability_support(cap, "LIMIT", "GTC")

        assert allowed is True
        assert reason == ""

    def test_gated_capability_fails(self):
        """Test validation fails for GATED capability."""
        cap = OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED)

        allowed, reason = PolicyValidator.validate_capability_support(cap, "LIMIT", "GTD")

        assert allowed is False
        assert "gated" in reason.lower()

    def test_unsupported_capability_fails(self):
        """Test validation fails for UNSUPPORTED capability."""
        cap = OrderCapability("BRACKET", "GTC", OrderTypeSupport.UNSUPPORTED)

        allowed, reason = PolicyValidator.validate_capability_support(cap, "BRACKET", "GTC")

        assert allowed is False
        assert "unsupported" in reason.lower()

    def test_missing_capability_fails(self):
        """Test validation fails when capability is None."""
        allowed, reason = PolicyValidator.validate_capability_support(None, "BRACKET", "GTC")

        assert allowed is False
        assert "not supported" in reason


class TestQuantityLimitsValidation:
    """Tests for quantity limit validation."""

    def test_quantity_within_limits(self):
        """Test validation passes for quantity within limits."""
        allowed, reason = PolicyValidator.validate_quantity_limits(
            quantity=Decimal("10"),
            min_order_size=Decimal("1"),
            max_order_size=Decimal("100"),
            capability=None,
        )

        assert allowed is True

    def test_quantity_below_minimum(self):
        """Test validation fails when quantity below minimum."""
        allowed, reason = PolicyValidator.validate_quantity_limits(
            quantity=Decimal("0.5"),
            min_order_size=Decimal("1"),
            max_order_size=None,
            capability=None,
        )

        assert allowed is False
        assert "below minimum" in reason

    def test_quantity_above_maximum(self):
        """Test validation fails when quantity above maximum."""
        allowed, reason = PolicyValidator.validate_quantity_limits(
            quantity=Decimal("200"),
            min_order_size=Decimal("1"),
            max_order_size=Decimal("100"),
            capability=None,
        )

        assert allowed is False
        assert "exceeds maximum" in reason

    def test_quantity_no_maximum_allowed(self):
        """Test validation passes when no maximum is set."""
        allowed, reason = PolicyValidator.validate_quantity_limits(
            quantity=Decimal("1000000"),
            min_order_size=Decimal("1"),
            max_order_size=None,
            capability=None,
        )

        assert allowed is True

    def test_capability_min_quantity_enforced(self):
        """Test capability min quantity overrides symbol minimum."""
        cap = OrderCapability(
            "LIMIT", "GTC", OrderTypeSupport.SUPPORTED, min_quantity=Decimal("10")
        )

        allowed, reason = PolicyValidator.validate_quantity_limits(
            quantity=Decimal("5"),
            min_order_size=Decimal("1"),  # Symbol min
            max_order_size=None,
            capability=cap,
        )

        assert allowed is False
        assert "capability minimum" in reason

    def test_capability_max_quantity_enforced(self):
        """Test capability max quantity overrides symbol maximum."""
        cap = OrderCapability(
            "LIMIT", "GTC", OrderTypeSupport.SUPPORTED, max_quantity=Decimal("50")
        )

        allowed, reason = PolicyValidator.validate_quantity_limits(
            quantity=Decimal("75"),
            min_order_size=Decimal("1"),
            max_order_size=Decimal("100"),  # Symbol max
            capability=cap,
        )

        assert allowed is False
        assert "capability maximum" in reason


class TestQuantityIncrementValidation:
    """Tests for quantity increment alignment."""

    def test_quantity_aligned_to_increment(self):
        """Test validation passes when quantity aligns to increment."""
        allowed, reason = PolicyValidator.validate_quantity_increment(
            quantity=Decimal("10.5"), size_increment=Decimal("0.5")
        )

        assert allowed is True

    def test_quantity_not_aligned_to_increment(self):
        """Test validation fails when quantity doesn't align."""
        allowed, reason = PolicyValidator.validate_quantity_increment(
            quantity=Decimal("10.3"), size_increment=Decimal("0.5")
        )

        assert allowed is False
        assert "not aligned" in reason

    def test_whole_number_increment(self):
        """Test validation with whole number increment."""
        allowed, reason = PolicyValidator.validate_quantity_increment(
            quantity=Decimal("100"), size_increment=Decimal("10")
        )

        assert allowed is True

    def test_very_small_increment(self):
        """Test validation with very small increment."""
        allowed, reason = PolicyValidator.validate_quantity_increment(
            quantity=Decimal("1.001"), size_increment=Decimal("0.001")
        )

        assert allowed is True


class TestPriceIncrementValidation:
    """Tests for price increment alignment."""

    def test_price_aligned_to_increment(self):
        """Test validation passes when price aligns."""
        allowed, reason = PolicyValidator.validate_price_increment(
            price=Decimal("50000"), price_increment=Decimal("1")
        )

        assert allowed is True

    def test_price_not_aligned_to_increment(self):
        """Test validation fails when price doesn't align."""
        allowed, reason = PolicyValidator.validate_price_increment(
            price=Decimal("50000.5"), price_increment=Decimal("1")
        )

        assert allowed is False
        assert "not aligned" in reason

    def test_none_price_skips_validation(self):
        """Test None price (market orders) skips validation."""
        allowed, reason = PolicyValidator.validate_price_increment(
            price=None, price_increment=Decimal("0.01")
        )

        assert allowed is True

    def test_decimal_price_increment(self):
        """Test validation with decimal price increment."""
        allowed, reason = PolicyValidator.validate_price_increment(
            price=Decimal("3000.10"), price_increment=Decimal("0.10")
        )

        assert allowed is True


class TestNotionalLimitValidation:
    """Tests for notional (quantity * price) limit validation."""

    def test_notional_within_limit(self):
        """Test validation passes when notional within limit."""
        cap = OrderCapability(
            "LIMIT", "GTC", OrderTypeSupport.SUPPORTED, max_notional=Decimal("100000")
        )

        allowed, reason = PolicyValidator.validate_notional_limit(
            quantity=Decimal("1"), price=Decimal("50000"), capability=cap
        )

        assert allowed is True

    def test_notional_exceeds_limit(self):
        """Test validation fails when notional exceeds limit."""
        cap = OrderCapability(
            "LIMIT", "GTC", OrderTypeSupport.SUPPORTED, max_notional=Decimal("100000")
        )

        allowed, reason = PolicyValidator.validate_notional_limit(
            quantity=Decimal("3"), price=Decimal("50000"), capability=cap  # $150k notional
        )

        assert allowed is False
        assert "Notional" in reason
        assert "exceeds maximum" in reason

    def test_none_price_skips_notional_check(self):
        """Test None price skips notional validation."""
        cap = OrderCapability(
            "MARKET", "IOC", OrderTypeSupport.SUPPORTED, max_notional=Decimal("100000")
        )

        allowed, reason = PolicyValidator.validate_notional_limit(
            quantity=Decimal("1000"), price=None, capability=cap
        )

        assert allowed is True

    def test_no_notional_limit_passes(self):
        """Test validation passes when no notional limit set."""
        cap = OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED, max_notional=None)

        allowed, reason = PolicyValidator.validate_notional_limit(
            quantity=Decimal("1000"), price=Decimal("50000"), capability=cap
        )

        assert allowed is True

    def test_none_capability_skips_notional_check(self):
        """Test None capability skips notional validation."""
        allowed, reason = PolicyValidator.validate_notional_limit(
            quantity=Decimal("100"), price=Decimal("50000"), capability=None
        )

        assert allowed is True


class TestPostOnlySupportValidation:
    """Tests for post-only flag validation."""

    def test_post_only_supported(self):
        """Test validation passes when post-only is supported."""
        cap = OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED, post_only_supported=True)

        allowed, reason = PolicyValidator.validate_post_only_support(post_only=True, capability=cap)

        assert allowed is True

    def test_post_only_not_supported(self):
        """Test validation fails when post-only not supported."""
        cap = OrderCapability(
            "MARKET", "IOC", OrderTypeSupport.SUPPORTED, post_only_supported=False
        )

        allowed, reason = PolicyValidator.validate_post_only_support(post_only=True, capability=cap)

        assert allowed is False
        assert "Post-only not supported" in reason

    def test_post_only_not_requested(self):
        """Test validation passes when post-only not requested."""
        cap = OrderCapability(
            "MARKET", "IOC", OrderTypeSupport.SUPPORTED, post_only_supported=False
        )

        allowed, reason = PolicyValidator.validate_post_only_support(
            post_only=False, capability=cap
        )

        assert allowed is True

    def test_post_only_with_none_capability(self):
        """Test post-only validation with None capability passes."""
        allowed, reason = PolicyValidator.validate_post_only_support(
            post_only=True, capability=None
        )

        assert allowed is True


class TestReduceOnlySupportValidation:
    """Tests for reduce-only flag validation."""

    def test_reduce_only_supported(self):
        """Test validation passes when reduce-only is supported."""
        cap = OrderCapability(
            "LIMIT", "GTC", OrderTypeSupport.SUPPORTED, reduce_only_supported=True
        )

        allowed, reason = PolicyValidator.validate_reduce_only_support(
            reduce_only=True, capability=cap
        )

        assert allowed is True

    def test_reduce_only_not_supported(self):
        """Test validation fails when reduce-only not supported."""
        cap = OrderCapability(
            "MARKET", "IOC", OrderTypeSupport.SUPPORTED, reduce_only_supported=False
        )

        allowed, reason = PolicyValidator.validate_reduce_only_support(
            reduce_only=True, capability=cap
        )

        assert allowed is False
        assert "Reduce-only not supported" in reason

    def test_reduce_only_not_requested(self):
        """Test validation passes when reduce-only not requested."""
        cap = OrderCapability(
            "MARKET", "IOC", OrderTypeSupport.SUPPORTED, reduce_only_supported=False
        )

        allowed, reason = PolicyValidator.validate_reduce_only_support(
            reduce_only=False, capability=cap
        )

        assert allowed is True


class TestEnvironmentRulesValidation:
    """Tests for environment-specific validation."""

    def test_paper_trading_gtd_stop_rejected(self):
        """Test paper trading rejects GTD stop orders."""
        allowed, reason = PolicyValidator.validate_environment_rules(
            environment="paper", order_type="STOP", tif="GTD"
        )

        assert allowed is False
        assert "paper trading" in reason.lower()

    def test_paper_trading_gtd_stop_limit_rejected(self):
        """Test paper trading rejects GTD stop-limit orders."""
        allowed, reason = PolicyValidator.validate_environment_rules(
            environment="paper", order_type="STOP_LIMIT", tif="GTD"
        )

        assert allowed is False
        assert "paper trading" in reason.lower()

    def test_paper_trading_normal_orders_allowed(self):
        """Test paper trading allows normal orders."""
        allowed, reason = PolicyValidator.validate_environment_rules(
            environment="paper", order_type="LIMIT", tif="GTC"
        )

        assert allowed is True

    def test_sandbox_allows_all_orders(self):
        """Test sandbox environment allows all order types."""
        allowed, reason = PolicyValidator.validate_environment_rules(
            environment="sandbox", order_type="STOP", tif="GTD"
        )

        assert allowed is True

    def test_live_allows_all_orders(self):
        """Test live environment allows all order types."""
        allowed, reason = PolicyValidator.validate_environment_rules(
            environment="live", order_type="STOP_LIMIT", tif="GTD"
        )

        assert allowed is True


class TestFullValidationPipeline:
    """Tests for complete validation pipeline."""

    @pytest.fixture
    def valid_capability(self):
        """Create valid LIMIT/GTC capability."""
        return OrderCapability(
            "LIMIT",
            "GTC",
            OrderTypeSupport.SUPPORTED,
            min_quantity=Decimal("0.001"),
            max_quantity=Decimal("1000"),
            post_only_supported=True,
            reduce_only_supported=True,
        )

    def test_complete_valid_order(self, valid_capability):
        """Test complete validation passes for valid order."""
        allowed, reason = PolicyValidator.validate_order(
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            trading_enabled=True,
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("100"),
            size_increment=Decimal("0.001"),
            price_increment=Decimal("1"),
            capability=valid_capability,
            post_only=False,
            reduce_only=False,
            environment="sandbox",
        )

        assert allowed is True
        assert reason == "Order allowed"

    def test_validation_fails_at_first_error(self):
        """Test validation stops at first error."""
        allowed, reason = PolicyValidator.validate_order(
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            trading_enabled=False,  # This should fail first
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("100"),
            size_increment=Decimal("0.5"),  # This would also fail
            price_increment=Decimal("1"),
            capability=None,  # This would also fail
        )

        assert allowed is False
        assert "disabled" in reason.lower()

    def test_validation_with_market_order(self):
        """Test validation for market order (no price)."""
        market_cap = OrderCapability(
            "MARKET", "IOC", OrderTypeSupport.SUPPORTED, post_only_supported=False
        )

        allowed, reason = PolicyValidator.validate_order(
            order_type="MARKET",
            tif="IOC",
            quantity=Decimal("1.0"),
            price=None,  # Market order
            trading_enabled=True,
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("100"),
            size_increment=Decimal("0.001"),
            price_increment=Decimal("1"),
            capability=market_cap,
        )

        assert allowed is True

    def test_validation_with_post_only(self, valid_capability):
        """Test validation with post-only flag."""
        allowed, reason = PolicyValidator.validate_order(
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            trading_enabled=True,
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("100"),
            size_increment=Decimal("0.001"),
            price_increment=Decimal("1"),
            capability=valid_capability,
            post_only=True,
            reduce_only=False,
        )

        assert allowed is True

    def test_validation_in_paper_environment(self, valid_capability):
        """Test validation respects paper environment rules."""
        gtd_cap = OrderCapability("STOP", "GTD", OrderTypeSupport.SUPPORTED)

        allowed, reason = PolicyValidator.validate_order(
            order_type="STOP",
            tif="GTD",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            trading_enabled=True,
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("100"),
            size_increment=Decimal("0.001"),
            price_increment=Decimal("1"),
            capability=gtd_cap,
            environment="paper",
        )

        assert allowed is False
        assert "paper trading" in reason.lower()
