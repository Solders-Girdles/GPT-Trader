"""
Unit tests for CapabilityRegistry.

Tests capability management, lookup, filtering, and validation logic.
"""

from decimal import Decimal

import pytest

from bot_v2.features.live_trade.capability_registry import CapabilityRegistry
from bot_v2.features.live_trade.order_policy import OrderCapability, OrderTypeSupport


class TestCoinbaseCapabilities:
    """Tests for Coinbase perpetuals capability definitions."""

    def test_get_coinbase_perp_capabilities_returns_list(self):
        """Test that method returns a list of capabilities."""
        caps = CapabilityRegistry.get_coinbase_perp_capabilities()

        assert isinstance(caps, list)
        assert len(caps) > 0
        assert all(isinstance(c, OrderCapability) for c in caps)

    def test_coinbase_capabilities_count(self):
        """Test expected number of Coinbase capabilities."""
        caps = CapabilityRegistry.get_coinbase_perp_capabilities()

        # Should have 11 default capabilities
        # MARKET/IOC, 3x LIMIT, 2x STOP, 2x STOP_LIMIT, 3x GTD
        assert len(caps) == 11

    def test_coinbase_has_market_ioc(self):
        """Test MARKET/IOC capability exists."""
        caps = CapabilityRegistry.get_coinbase_perp_capabilities()

        market_ioc = CapabilityRegistry.find_capability(caps, "MARKET", "IOC")

        assert market_ioc is not None
        assert market_ioc.order_type == "MARKET"
        assert market_ioc.tif == "IOC"
        assert market_ioc.support_level == OrderTypeSupport.SUPPORTED
        assert market_ioc.post_only_supported is False

    def test_coinbase_has_limit_gtc(self):
        """Test LIMIT/GTC capability exists."""
        caps = CapabilityRegistry.get_coinbase_perp_capabilities()

        limit_gtc = CapabilityRegistry.find_capability(caps, "LIMIT", "GTC")

        assert limit_gtc is not None
        assert limit_gtc.support_level == OrderTypeSupport.SUPPORTED

    def test_coinbase_gtd_orders_gated(self):
        """Test GTD orders start as GATED."""
        caps = CapabilityRegistry.get_coinbase_perp_capabilities()

        limit_gtd = CapabilityRegistry.find_capability(caps, "LIMIT", "GTD")
        stop_gtd = CapabilityRegistry.find_capability(caps, "STOP", "GTD")
        stop_limit_gtd = CapabilityRegistry.find_capability(caps, "STOP_LIMIT", "GTD")

        assert limit_gtd is not None
        assert limit_gtd.support_level == OrderTypeSupport.GATED
        assert stop_gtd.support_level == OrderTypeSupport.GATED
        assert stop_limit_gtd.support_level == OrderTypeSupport.GATED

    def test_coinbase_capabilities_are_independent(self):
        """Test that multiple calls return independent capability lists."""
        caps1 = CapabilityRegistry.get_coinbase_perp_capabilities()
        caps2 = CapabilityRegistry.get_coinbase_perp_capabilities()

        # Should be different list instances
        assert caps1 is not caps2

        # Modifying one shouldn't affect the other
        caps1[0].support_level = OrderTypeSupport.UNSUPPORTED
        assert caps2[0].support_level == OrderTypeSupport.SUPPORTED


class TestCapabilityLookup:
    """Tests for capability lookup functionality."""

    @pytest.fixture
    def sample_capabilities(self):
        """Sample capability list for testing."""
        return [
            OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "IOC", OrderTypeSupport.SUPPORTED),
            OrderCapability("MARKET", "IOC", OrderTypeSupport.SUPPORTED),
            OrderCapability("STOP", "GTD", OrderTypeSupport.GATED),
        ]

    def test_find_capability_success(self, sample_capabilities):
        """Test finding existing capability."""
        result = CapabilityRegistry.find_capability(sample_capabilities, "LIMIT", "GTC")

        assert result is not None
        assert result.order_type == "LIMIT"
        assert result.tif == "GTC"

    def test_find_capability_not_found(self, sample_capabilities):
        """Test finding non-existent capability returns None."""
        result = CapabilityRegistry.find_capability(sample_capabilities, "BRACKET", "GTC")

        assert result is None

    def test_find_capability_wrong_tif(self, sample_capabilities):
        """Test that order type match alone isn't enough."""
        result = CapabilityRegistry.find_capability(sample_capabilities, "LIMIT", "FOK")

        assert result is None

    def test_find_capability_empty_list(self):
        """Test finding capability in empty list."""
        result = CapabilityRegistry.find_capability([], "LIMIT", "GTC")

        assert result is None

    def test_find_capability_case_sensitive(self, sample_capabilities):
        """Test that lookup is case-sensitive."""
        result = CapabilityRegistry.find_capability(sample_capabilities, "limit", "gtc")

        # Should not find lowercase
        assert result is None


class TestGTDEnablement:
    """Tests for GTD order enablement."""

    def test_enable_gtd_orders_changes_gated_to_supported(self):
        """Test enabling GTD changes status."""
        caps = [
            OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),
            OrderCapability("STOP", "GTD", OrderTypeSupport.GATED),
        ]

        enabled = CapabilityRegistry.enable_gtd_orders(caps)

        assert enabled is True
        assert caps[0].support_level == OrderTypeSupport.SUPPORTED
        assert caps[1].support_level == OrderTypeSupport.SUPPORTED

    def test_enable_gtd_returns_false_if_no_gtd(self):
        """Test enabling returns False when no GTD capabilities."""
        caps = [
            OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("MARKET", "IOC", OrderTypeSupport.SUPPORTED),
        ]

        enabled = CapabilityRegistry.enable_gtd_orders(caps)

        assert enabled is False

    def test_enable_gtd_idempotent(self):
        """Test enabling GTD multiple times is safe."""
        caps = [
            OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),
        ]

        # First enable
        enabled1 = CapabilityRegistry.enable_gtd_orders(caps)
        assert enabled1 is True

        # Second enable (already supported)
        enabled2 = CapabilityRegistry.enable_gtd_orders(caps)
        assert enabled2 is False  # Nothing changed

    def test_enable_gtd_only_affects_gated(self):
        """Test that only GATED GTD orders are changed."""
        caps = [
            OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),
            OrderCapability("STOP", "GTD", OrderTypeSupport.SUPPORTED),  # Already supported
            OrderCapability("STOP_LIMIT", "GTD", OrderTypeSupport.UNSUPPORTED),
        ]

        enabled = CapabilityRegistry.enable_gtd_orders(caps)

        assert enabled is True
        assert caps[0].support_level == OrderTypeSupport.SUPPORTED  # Changed
        assert caps[1].support_level == OrderTypeSupport.SUPPORTED  # Unchanged
        assert caps[2].support_level == OrderTypeSupport.UNSUPPORTED  # Not changed

    def test_enable_gtd_empty_list(self):
        """Test enabling on empty list."""
        caps: list[OrderCapability] = []

        enabled = CapabilityRegistry.enable_gtd_orders(caps)

        assert enabled is False


class TestCapabilityFiltering:
    """Tests for capability filtering operations."""

    @pytest.fixture
    def mixed_capabilities(self):
        """Capabilities with different support levels."""
        return [
            OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),
            OrderCapability("MARKET", "IOC", OrderTypeSupport.SUPPORTED),
            OrderCapability("BRACKET", "GTC", OrderTypeSupport.UNSUPPORTED),
            OrderCapability("STOP", "GTC", OrderTypeSupport.SUPPORTED),
        ]

    def test_filter_supported(self, mixed_capabilities):
        """Test filtering to only supported capabilities."""
        supported = CapabilityRegistry.filter_supported(mixed_capabilities)

        assert len(supported) == 3
        assert all(c.support_level == OrderTypeSupport.SUPPORTED for c in supported)

    def test_filter_supported_preserves_order(self, mixed_capabilities):
        """Test that filtering preserves original order."""
        supported = CapabilityRegistry.filter_supported(mixed_capabilities)

        # First supported should be LIMIT/GTC
        assert supported[0].order_type == "LIMIT"
        assert supported[0].tif == "GTC"

    def test_filter_supported_empty_list(self):
        """Test filtering empty list."""
        supported = CapabilityRegistry.filter_supported([])

        assert supported == []

    def test_filter_supported_none_supported(self):
        """Test when no capabilities are supported."""
        caps = [
            OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),
            OrderCapability("BRACKET", "GTC", OrderTypeSupport.UNSUPPORTED),
        ]

        supported = CapabilityRegistry.filter_supported(caps)

        assert supported == []

    def test_get_supported_order_types(self, mixed_capabilities):
        """Test extracting unique supported order types."""
        types = CapabilityRegistry.get_supported_order_types(mixed_capabilities)

        assert types == {"LIMIT", "MARKET", "STOP"}

    def test_get_supported_order_types_deduplicates(self):
        """Test that duplicate order types are deduplicated."""
        caps = [
            OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "IOC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "FOK", OrderTypeSupport.SUPPORTED),
        ]

        types = CapabilityRegistry.get_supported_order_types(caps)

        assert types == {"LIMIT"}

    def test_get_supported_order_types_empty(self):
        """Test getting order types from empty list."""
        types = CapabilityRegistry.get_supported_order_types([])

        assert types == set()


class TestCapabilityValidation:
    """Tests for capability validation."""

    def test_validate_basic_capability(self):
        """Test validating basic valid capability."""
        cap = OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED)

        valid, error = CapabilityRegistry.validate_capability(cap)

        assert valid is True
        assert error == ""

    def test_validate_missing_order_type(self):
        """Test validation fails for missing order_type."""
        cap = OrderCapability("", "GTC", OrderTypeSupport.SUPPORTED)

        valid, error = CapabilityRegistry.validate_capability(cap)

        assert valid is False
        assert "order_type" in error

    def test_validate_missing_tif(self):
        """Test validation fails for missing TIF."""
        cap = OrderCapability("LIMIT", "", OrderTypeSupport.SUPPORTED)

        valid, error = CapabilityRegistry.validate_capability(cap)

        assert valid is False
        assert "tif" in error

    def test_validate_min_exceeds_max_quantity(self):
        """Test validation fails when min > max."""
        cap = OrderCapability(
            "LIMIT",
            "GTC",
            OrderTypeSupport.SUPPORTED,
            min_quantity=Decimal("100"),
            max_quantity=Decimal("10"),
        )

        valid, error = CapabilityRegistry.validate_capability(cap)

        assert valid is False
        assert "min_quantity" in error or "max_quantity" in error

    def test_validate_negative_quantity_increment(self):
        """Test validation fails for negative increment."""
        cap = OrderCapability(
            "LIMIT",
            "GTC",
            OrderTypeSupport.SUPPORTED,
            quantity_increment=Decimal("-0.1"),
        )

        valid, error = CapabilityRegistry.validate_capability(cap)

        assert valid is False
        assert "quantity_increment" in error

    def test_validate_zero_quantity_increment(self):
        """Test validation fails for zero increment."""
        cap = OrderCapability(
            "LIMIT",
            "GTC",
            OrderTypeSupport.SUPPORTED,
            quantity_increment=Decimal("0"),
        )

        valid, error = CapabilityRegistry.validate_capability(cap)

        assert valid is False
        assert "quantity_increment" in error

    def test_validate_negative_price_increment(self):
        """Test validation fails for negative price increment."""
        cap = OrderCapability(
            "LIMIT",
            "GTC",
            OrderTypeSupport.SUPPORTED,
            price_increment=Decimal("-0.01"),
        )

        valid, error = CapabilityRegistry.validate_capability(cap)

        assert valid is False
        assert "price_increment" in error

    def test_validate_zero_rate_limit(self):
        """Test validation fails for zero rate limit."""
        cap = OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED, rate_limit_per_minute=0)

        valid, error = CapabilityRegistry.validate_capability(cap)

        assert valid is False
        assert "rate_limit" in error

    def test_validate_capability_with_all_optional_fields(self):
        """Test validating capability with all fields set."""
        cap = OrderCapability(
            "LIMIT",
            "GTC",
            OrderTypeSupport.SUPPORTED,
            min_quantity=Decimal("0.001"),
            max_quantity=Decimal("1000"),
            quantity_increment=Decimal("0.001"),
            price_increment=Decimal("0.01"),
            max_notional=Decimal("100000"),
            rate_limit_per_minute=30,
            post_only_supported=True,
            reduce_only_supported=True,
        )

        valid, error = CapabilityRegistry.validate_capability(cap)

        assert valid is True


class TestCapabilityCounting:
    """Tests for capability counting by support level."""

    def test_count_by_support_level(self):
        """Test counting capabilities by support level."""
        caps = [
            OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "IOC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),
            OrderCapability("BRACKET", "GTC", OrderTypeSupport.UNSUPPORTED),
        ]

        counts = CapabilityRegistry.count_by_support_level(caps)

        assert counts[OrderTypeSupport.SUPPORTED] == 2
        assert counts[OrderTypeSupport.GATED] == 1
        assert counts[OrderTypeSupport.UNSUPPORTED] == 1

    def test_count_empty_list(self):
        """Test counting empty capability list."""
        counts = CapabilityRegistry.count_by_support_level([])

        assert counts[OrderTypeSupport.SUPPORTED] == 0
        assert counts[OrderTypeSupport.GATED] == 0
        assert counts[OrderTypeSupport.UNSUPPORTED] == 0

    def test_count_all_same_level(self):
        """Test counting when all at same level."""
        caps = [
            OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("MARKET", "IOC", OrderTypeSupport.SUPPORTED),
            OrderCapability("STOP", "GTC", OrderTypeSupport.SUPPORTED),
        ]

        counts = CapabilityRegistry.count_by_support_level(caps)

        assert counts[OrderTypeSupport.SUPPORTED] == 3
        assert counts[OrderTypeSupport.GATED] == 0
        assert counts[OrderTypeSupport.UNSUPPORTED] == 0
