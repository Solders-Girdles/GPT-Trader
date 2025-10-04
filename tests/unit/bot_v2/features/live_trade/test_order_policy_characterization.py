"""
Characterization tests for OrderPolicy module.

This test locks in the behavior of OrderPolicy components before refactoring:
- OrderCapability: Order type capability definitions
- SymbolPolicy: Trading policy validation logic
- OrderPolicyMatrix: Policy enforcement and recommendations

The test verifies complete flows:
capability definition → policy validation → rate limiting → recommendations
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.features.live_trade.capability_registry import OrderCapability, OrderTypeSupport
from bot_v2.features.live_trade.order_policy import (
    OrderPolicyMatrix,
    SymbolPolicy,
    create_standard_policy_matrix,
)


class TestSymbolPolicyCharacterization:
    """Characterization tests for SymbolPolicy validation logic."""

    @pytest.fixture
    def btc_policy(self):
        """Create BTC policy with standard capabilities."""
        capabilities = [
            OrderCapability("MARKET", "IOC", OrderTypeSupport.SUPPORTED, post_only_supported=False),
            OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "IOC", OrderTypeSupport.SUPPORTED),
            OrderCapability("STOP", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),  # Gated capability
        ]

        return SymbolPolicy(
            symbol="BTC-USD",
            environment="sandbox",
            capabilities=capabilities,
            min_order_size=Decimal("0.001"),
            max_order_size=Decimal("100"),
            size_increment=Decimal("0.001"),
            price_increment=Decimal("1"),
            trading_enabled=True,
            reduce_only_mode=False,
            spread_threshold_bps=Decimal("20"),
        )

    @pytest.fixture
    def eth_policy_disabled(self):
        """Create ETH policy with trading disabled."""
        capabilities = [
            OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
        ]

        return SymbolPolicy(
            symbol="ETH-USD",
            environment="sandbox",
            capabilities=capabilities,
            min_order_size=Decimal("0.01"),
            size_increment=Decimal("0.01"),
            price_increment=Decimal("0.1"),
            trading_enabled=False,  # Trading disabled
        )

    def test_basic_validation_trading_enabled(self, btc_policy):
        """Test basic validation succeeds when trading enabled."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="LIMIT", tif="GTC", quantity=Decimal("1.0"), price=Decimal("50000")
        )

        assert allowed is True
        assert reason == "Order allowed"

    def test_basic_validation_trading_disabled(self, eth_policy_disabled):
        """Test validation fails when trading disabled."""
        allowed, reason = eth_policy_disabled.is_order_allowed(
            order_type="LIMIT", tif="GTC", quantity=Decimal("1.0"), price=Decimal("3000")
        )

        assert allowed is False
        assert reason == "Trading disabled for symbol"

    def test_capability_lookup_success(self, btc_policy):
        """Test capability lookup finds matching order type and TIF."""
        capability = btc_policy.get_capability("LIMIT", "GTC")

        assert capability is not None
        assert capability.order_type == "LIMIT"
        assert capability.tif == "GTC"
        assert capability.support_level == OrderTypeSupport.SUPPORTED

    def test_capability_lookup_missing(self, btc_policy):
        """Test capability lookup returns None for unsupported combination."""
        capability = btc_policy.get_capability("BRACKET", "GTC")

        assert capability is None

    def test_validation_unsupported_capability(self, btc_policy):
        """Test validation fails for missing capability."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="BRACKET", tif="GTC", quantity=Decimal("1.0")
        )

        assert allowed is False
        assert "not supported" in reason

    def test_validation_gated_capability(self, btc_policy):
        """Test validation fails for gated capability."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="LIMIT", tif="GTD", quantity=Decimal("1.0")
        )

        assert allowed is False
        assert "gated" in reason.lower()

    def test_quantity_below_minimum(self, btc_policy):
        """Test validation fails when quantity below minimum."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="LIMIT", tif="GTC", quantity=Decimal("0.0001")  # Below 0.001 min
        )

        assert allowed is False
        assert "below minimum" in reason

    def test_quantity_above_maximum(self, btc_policy):
        """Test validation fails when quantity above maximum."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="LIMIT", tif="GTC", quantity=Decimal("200")  # Above 100 max
        )

        assert allowed is False
        assert "exceeds maximum" in reason

    def test_quantity_not_aligned_to_increment(self, btc_policy):
        """Test validation fails when quantity not aligned to increment."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("1.0015"),  # Not aligned to 0.001 increment
        )

        assert allowed is False
        assert "not aligned to increment" in reason

    def test_quantity_aligned_to_increment(self, btc_policy):
        """Test validation succeeds when quantity aligned to increment."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("1.001"),  # Aligned to 0.001 increment
            price=Decimal("50000"),
        )

        assert allowed is True

    def test_price_not_aligned_to_increment(self, btc_policy):
        """Test validation fails when price not aligned to increment."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("1.0"),
            price=Decimal("50000.5"),  # Not aligned to $1 increment
        )

        assert allowed is False
        assert "Price" in reason
        assert "not aligned to increment" in reason

    def test_price_aligned_to_increment(self, btc_policy):
        """Test validation succeeds when price aligned to increment."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("1.0"),
            price=Decimal("50001"),  # Aligned to $1 increment
        )

        assert allowed is True

    def test_price_validation_skipped_when_none(self, btc_policy):
        """Test price validation skipped when price is None (market orders)."""
        allowed, reason = btc_policy.is_order_allowed(
            order_type="MARKET", tif="IOC", quantity=Decimal("1.0"), price=None
        )

        assert allowed is True

    def test_capability_min_quantity_enforced(self):
        """Test capability-specific min quantity is enforced."""
        capabilities = [
            OrderCapability(
                "LIMIT",
                "GTC",
                OrderTypeSupport.SUPPORTED,
                min_quantity=Decimal("10"),  # Capability min
            ),
        ]

        policy = SymbolPolicy(
            symbol="TEST-USD",
            environment="sandbox",
            capabilities=capabilities,
            min_order_size=Decimal("1"),  # Policy min (lower)
            size_increment=Decimal("1"),
        )

        # Below capability min but above policy min
        allowed, reason = policy.is_order_allowed(
            order_type="LIMIT", tif="GTC", quantity=Decimal("5")
        )

        assert allowed is False
        assert "capability minimum" in reason

    def test_capability_max_quantity_enforced(self):
        """Test capability-specific max quantity is enforced."""
        capabilities = [
            OrderCapability(
                "LIMIT",
                "GTC",
                OrderTypeSupport.SUPPORTED,
                max_quantity=Decimal("50"),  # Capability max
            ),
        ]

        policy = SymbolPolicy(
            symbol="TEST-USD",
            environment="sandbox",
            capabilities=capabilities,
            max_order_size=Decimal("100"),  # Policy max (higher)
            size_increment=Decimal("1"),
        )

        # Above capability max but below policy max
        allowed, reason = policy.is_order_allowed(
            order_type="LIMIT", tif="GTC", quantity=Decimal("75")
        )

        assert allowed is False
        assert "capability maximum" in reason

    def test_notional_limit_enforced(self):
        """Test notional (quantity * price) limit is enforced."""
        capabilities = [
            OrderCapability(
                "LIMIT",
                "GTC",
                OrderTypeSupport.SUPPORTED,
                max_notional=Decimal("100000"),  # $100k max notional
            ),
        ]

        policy = SymbolPolicy(
            symbol="BTC-USD",
            environment="sandbox",
            capabilities=capabilities,
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
            price_increment=Decimal("1"),
        )

        # 3 BTC @ $50k = $150k notional (exceeds $100k limit)
        allowed, reason = policy.is_order_allowed(
            order_type="LIMIT", tif="GTC", quantity=Decimal("3"), price=Decimal("50000")
        )

        assert allowed is False
        assert "Notional" in reason
        assert "exceeds maximum" in reason

    def test_notional_limit_within_bounds(self):
        """Test validation succeeds when notional within limit."""
        capabilities = [
            OrderCapability(
                "LIMIT",
                "GTC",
                OrderTypeSupport.SUPPORTED,
                max_notional=Decimal("100000"),  # $100k max notional
            ),
        ]

        policy = SymbolPolicy(
            symbol="BTC-USD",
            environment="sandbox",
            capabilities=capabilities,
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
            price_increment=Decimal("1"),
        )

        # 1 BTC @ $50k = $50k notional (within $100k limit)
        allowed, reason = policy.is_order_allowed(
            order_type="LIMIT", tif="GTC", quantity=Decimal("1"), price=Decimal("50000")
        )

        assert allowed is True

    def test_to_dict_serialization(self, btc_policy):
        """Test SymbolPolicy serializes to dict correctly."""
        policy_dict = btc_policy.to_dict()

        assert policy_dict["symbol"] == "BTC-USD"
        assert policy_dict["environment"] == "sandbox"
        assert policy_dict["min_order_size"] == 0.001
        assert policy_dict["max_order_size"] == 100.0
        assert policy_dict["size_increment"] == 0.001
        assert policy_dict["price_increment"] == 1.0
        assert policy_dict["trading_enabled"] is True
        assert policy_dict["reduce_only_mode"] is False
        assert policy_dict["spread_threshold_bps"] == 20.0
        assert isinstance(policy_dict["capabilities"], list)
        assert len(policy_dict["capabilities"]) == 5

    def test_capability_to_dict_serialization(self):
        """Test OrderCapability serializes to dict correctly."""
        capability = OrderCapability(
            order_type="LIMIT",
            tif="GTC",
            support_level=OrderTypeSupport.SUPPORTED,
            min_quantity=Decimal("1"),
            max_quantity=Decimal("100"),
            quantity_increment=Decimal("0.1"),
            price_increment=Decimal("0.01"),
            max_notional=Decimal("10000"),
            rate_limit_per_minute=30,
            post_only_supported=True,
            reduce_only_supported=False,
        )

        cap_dict = capability.to_dict()

        assert cap_dict["order_type"] == "LIMIT"
        assert cap_dict["tif"] == "GTC"
        assert cap_dict["support_level"] == "supported"
        assert cap_dict["min_quantity"] == 1.0
        assert cap_dict["max_quantity"] == 100.0
        assert cap_dict["quantity_increment"] == 0.1
        assert cap_dict["price_increment"] == 0.01
        assert cap_dict["max_notional"] == 10000.0
        assert cap_dict["rate_limit_per_minute"] == 30
        assert cap_dict["post_only_supported"] is True
        assert cap_dict["reduce_only_supported"] is False


class TestOrderPolicyMatrixCharacterization:
    """Characterization tests for OrderPolicyMatrix enforcement."""

    @pytest.fixture
    def matrix(self):
        """Create policy matrix with standard symbols."""
        return create_standard_policy_matrix(environment="sandbox")

    def test_symbol_addition(self):
        """Test adding symbol creates policy with capabilities."""
        matrix = OrderPolicyMatrix(environment="sandbox")

        policy = matrix.add_symbol(
            symbol="BTC-USD",
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
            price_increment=Decimal("1"),
        )

        assert policy.symbol == "BTC-USD"
        assert policy.environment == "sandbox"
        assert len(policy.capabilities) > 0  # Gets default Coinbase capabilities

    def test_get_symbol_policy_exists(self, matrix):
        """Test retrieving existing symbol policy."""
        policy = matrix.get_symbol_policy("BTC-USD")

        assert policy is not None
        assert policy.symbol == "BTC-USD"

    def test_get_symbol_policy_missing(self, matrix):
        """Test retrieving missing symbol returns None."""
        policy = matrix.get_symbol_policy("DOGE-USD")

        assert policy is None

    def test_validate_order_success(self, matrix):
        """Test order validation succeeds for valid order."""
        allowed, reason = matrix.validate_order(
            symbol="BTC-USD",
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )

        assert allowed is True
        assert "validated" in reason.lower()

    def test_validate_order_missing_symbol(self, matrix):
        """Test validation fails for symbol without policy."""
        allowed, reason = matrix.validate_order(
            symbol="DOGE-USD", order_type="LIMIT", tif="GTC", quantity=Decimal("100")
        )

        assert allowed is False
        assert "No policy" in reason

    def test_validate_order_post_only_supported(self, matrix):
        """Test post-only flag validation for supported order type."""
        allowed, reason = matrix.validate_order(
            symbol="BTC-USD",
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            post_only=True,  # Post-only supported for LIMIT
        )

        assert allowed is True

    def test_validate_order_post_only_unsupported(self, matrix):
        """Test post-only flag validation fails for market orders."""
        allowed, reason = matrix.validate_order(
            symbol="BTC-USD",
            order_type="MARKET",
            tif="IOC",
            quantity=Decimal("0.1"),
            post_only=True,  # Post-only NOT supported for MARKET
        )

        assert allowed is False
        assert "Post-only not supported" in reason

    def test_validate_order_reduce_only_supported(self, matrix):
        """Test reduce-only flag validation for supported order type."""
        allowed, reason = matrix.validate_order(
            symbol="BTC-USD",
            order_type="LIMIT",
            tif="GTC",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            reduce_only=True,
        )

        assert allowed is True

    def test_validate_order_paper_environment_restriction(self):
        """Test paper environment restricts GTD stop orders."""
        matrix = OrderPolicyMatrix(environment="paper")
        matrix.add_symbol(
            symbol="BTC-USD",
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
            price_increment=Decimal("1"),
        )

        # Enable GTD first
        matrix.enable_gtd_orders("BTC-USD")

        # Try GTD stop order in paper environment
        allowed, reason = matrix.validate_order(
            symbol="BTC-USD",
            order_type="STOP",
            tif="GTD",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )

        assert allowed is False
        assert "paper trading" in reason.lower()

    def test_get_supported_order_types(self, matrix):
        """Test retrieving supported order types for symbol."""
        supported = matrix.get_supported_order_types("BTC-USD")

        assert len(supported) > 0

        # Verify structure
        for config in supported:
            assert "order_type" in config
            assert "tif" in config
            assert "post_only" in config
            assert "reduce_only" in config

        # Should include LIMIT/GTC
        limit_gtc = [c for c in supported if c["order_type"] == "LIMIT" and c["tif"] == "GTC"]
        assert len(limit_gtc) == 1

        # Should NOT include gated GTD orders
        gtd_orders = [c for c in supported if c["tif"] == "GTD"]
        assert len(gtd_orders) == 0  # GTD is gated by default

    def test_recommend_order_config_normal_urgency(self, matrix):
        """Test order recommendation with normal urgency."""
        config = matrix.recommend_order_config(
            symbol="BTC-USD", side="buy", quantity=Decimal("0.1"), urgency="normal"
        )

        assert config["order_type"] == "LIMIT"
        assert config["tif"] == "GTC"
        assert config["post_only"] is False
        assert config["use_market"] is False

    def test_recommend_order_config_urgent_good_liquidity(self, matrix):
        """Test urgent recommendation with good liquidity uses market order."""
        config = matrix.recommend_order_config(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            urgency="urgent",
            market_conditions={"liquidity_condition": "good", "spread_bps": 5},
        )

        assert config["order_type"] == "MARKET"
        assert config["tif"] == "IOC"
        assert config["use_market"] is True

    def test_recommend_order_config_urgent_poor_liquidity(self, matrix):
        """Test urgent recommendation with poor liquidity uses limit IOC."""
        config = matrix.recommend_order_config(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            urgency="urgent",
            market_conditions={"liquidity_condition": "poor", "spread_bps": 50},
        )

        # Should use LIMIT instead of MARKET due to poor liquidity
        assert config["tif"] == "IOC"
        # Should be limit, not market
        assert config["order_type"] == "LIMIT" or config["use_market"] is False

    def test_recommend_order_config_patient(self, matrix):
        """Test patient recommendation uses post-only."""
        config = matrix.recommend_order_config(
            symbol="BTC-USD", side="buy", quantity=Decimal("0.1"), urgency="patient"
        )

        assert config["post_only"] is True

    def test_recommend_order_config_wide_spread_forces_post_only(self, matrix):
        """Test wide spread forces post-only regardless of urgency."""
        config = matrix.recommend_order_config(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            urgency="normal",
            market_conditions={"spread_bps": 25},  # Above 20bps threshold
        )

        assert config["post_only"] is True
        assert config["order_type"] == "LIMIT"
        assert config["use_market"] is False

    def test_recommend_order_config_high_volatility_uses_ioc(self, matrix):
        """Test high volatility uses IOC time-in-force."""
        config = matrix.recommend_order_config(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("0.1"),
            urgency="normal",
            market_conditions={"volatility_percentile": 95},  # High volatility
        )

        assert config["tif"] == "IOC"

    def test_recommend_order_config_fallback_on_invalid(self, matrix):
        """Test recommendation falls back to basic config if validation fails."""
        # Create scenario where recommended config would be invalid
        # (e.g., excessive quantity)
        config = matrix.recommend_order_config(
            symbol="BTC-USD",
            side="buy",
            quantity=Decimal("10000"),  # Exceeds max_order_size
            urgency="urgent",
        )

        # Should fallback to basic LIMIT/GTC
        assert config["order_type"] == "LIMIT"
        assert config["tif"] == "GTC"
        assert "fallback_reason" in config

    def test_recommend_order_config_missing_symbol(self, matrix):
        """Test recommendation returns error for missing symbol."""
        config = matrix.recommend_order_config(
            symbol="DOGE-USD", side="buy", quantity=Decimal("100")
        )

        assert "error" in config
        assert "No policy" in config["error"]

    def test_enable_gtd_orders(self):
        """Test enabling gated GTD orders."""
        # Create isolated matrix to avoid test pollution
        matrix = create_standard_policy_matrix(environment="sandbox")

        # GTD should be gated initially
        policy = matrix.get_symbol_policy("BTC-USD")
        gtd_caps = [c for c in policy.capabilities if c.tif == "GTD"]
        assert all(c.support_level == OrderTypeSupport.GATED for c in gtd_caps)

        # Enable GTD
        enabled = matrix.enable_gtd_orders("BTC-USD")

        assert enabled is True

        # GTD should now be supported
        policy = matrix.get_symbol_policy("BTC-USD")
        gtd_caps = [c for c in policy.capabilities if c.tif == "GTD"]
        assert all(c.support_level == OrderTypeSupport.SUPPORTED for c in gtd_caps)

    def test_enable_gtd_orders_missing_symbol(self, matrix):
        """Test enabling GTD for missing symbol returns False."""
        enabled = matrix.enable_gtd_orders("DOGE-USD")

        assert enabled is False

    def test_set_reduce_only_mode(self, matrix):
        """Test setting reduce-only mode."""
        policy = matrix.get_symbol_policy("BTC-USD")
        assert policy.reduce_only_mode is False

        # Enable reduce-only
        matrix.set_reduce_only_mode("BTC-USD", enabled=True)

        assert policy.reduce_only_mode is True

        # Disable reduce-only
        matrix.set_reduce_only_mode("BTC-USD", enabled=False)

        assert policy.reduce_only_mode is False

    def test_get_policy_summary(self, matrix):
        """Test policy summary generation."""
        summary = matrix.get_policy_summary()

        assert summary["environment"] == "sandbox"
        assert summary["symbols"] == 4  # BTC, ETH, SOL, XRP
        assert "policies" in summary

        # Check BTC policy
        btc_policy = summary["policies"]["BTC-USD"]
        assert btc_policy["trading_enabled"] is True
        assert btc_policy["reduce_only_mode"] is False
        assert "LIMIT" in btc_policy["supported_order_types"]
        assert "MARKET" in btc_policy["supported_order_types"]
        assert btc_policy["min_order_size"] == 0.001

    def test_rate_limit_enforcement(self):
        """Test rate limit prevents excessive orders."""
        matrix = OrderPolicyMatrix(environment="sandbox")

        # Add symbol with low rate limit for testing
        custom_capabilities = [
            OrderCapability(
                "LIMIT",
                "GTC",
                OrderTypeSupport.SUPPORTED,
                rate_limit_per_minute=3,  # Only 3 orders/min
            ),
        ]

        matrix.add_symbol(
            symbol="TEST-USD",
            capabilities=custom_capabilities,
            min_order_size=Decimal("1"),
            size_increment=Decimal("1"),
        )

        # First 3 orders should succeed
        for i in range(3):
            allowed, reason = matrix.validate_order(
                symbol="TEST-USD", order_type="LIMIT", tif="GTC", quantity=Decimal("10")
            )
            assert allowed is True, f"Order {i+1} should be allowed"

        # 4th order should fail due to rate limit
        allowed, reason = matrix.validate_order(
            symbol="TEST-USD", order_type="LIMIT", tif="GTC", quantity=Decimal("10")
        )

        assert allowed is False
        assert "Rate limit" in reason

    def test_rate_limit_resets_after_time(self):
        """Test rate limit resets after time window."""
        # Create mock time provider to simulate time passing
        current_time = [datetime(2025, 1, 1, 12, 0, 0)]

        def mock_time():
            return current_time[0]

        # Create tracker with mock time provider
        from bot_v2.features.live_trade.rate_limit_tracker import RateLimitTracker

        tracker = RateLimitTracker(window_minutes=1, time_provider=mock_time)
        matrix = OrderPolicyMatrix(environment="sandbox", rate_limit_tracker=tracker)

        custom_capabilities = [
            OrderCapability(
                "LIMIT",
                "GTC",
                OrderTypeSupport.SUPPORTED,
                rate_limit_per_minute=2,
            ),
        ]

        matrix.add_symbol(
            symbol="TEST-USD",
            capabilities=custom_capabilities,
            min_order_size=Decimal("1"),
            size_increment=Decimal("1"),
        )

        # Consume rate limit
        for i in range(2):
            matrix.validate_order(
                symbol="TEST-USD", order_type="LIMIT", tif="GTC", quantity=Decimal("10")
            )

        # Advance time by 2 minutes (outside 1-minute window)
        current_time[0] = datetime(2025, 1, 1, 12, 2, 0)

        # Next order should succeed (old timestamps cleaned)
        allowed, reason = matrix.validate_order(
            symbol="TEST-USD", order_type="LIMIT", tif="GTC", quantity=Decimal("10")
        )

        assert allowed is True


class TestStandardPolicyMatrixCharacterization:
    """Characterization tests for standard policy matrix creation."""

    def test_standard_matrix_creates_common_symbols(self):
        """Test standard matrix includes common perpetuals."""
        matrix = create_standard_policy_matrix(environment="sandbox")

        expected_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]

        for symbol in expected_symbols:
            policy = matrix.get_symbol_policy(symbol)
            assert policy is not None, f"{symbol} should be in standard matrix"
            assert policy.trading_enabled is True
            assert policy.reduce_only_mode is False

    def test_standard_matrix_symbol_increments(self):
        """Test standard matrix uses correct increments per symbol."""
        matrix = create_standard_policy_matrix(environment="sandbox")

        # BTC should have 0.001 size increment, $1 price increment
        btc = matrix.get_symbol_policy("BTC-USD")
        assert btc.size_increment == Decimal("0.001")
        assert btc.price_increment == Decimal("1")

        # ETH should have 0.01 size increment, $0.1 price increment
        eth = matrix.get_symbol_policy("ETH-USD")
        assert eth.size_increment == Decimal("0.01")
        assert eth.price_increment == Decimal("0.1")

        # SOL should have 0.1 size increment
        sol = matrix.get_symbol_policy("SOL-USD")
        assert sol.size_increment == Decimal("0.1")
        assert sol.price_increment == Decimal("0.001")

        # XRP should have 1 size increment
        xrp = matrix.get_symbol_policy("XRP-USD")
        assert xrp.size_increment == Decimal("1")
        assert xrp.price_increment == Decimal("0.0001")

    def test_standard_matrix_spread_threshold(self):
        """Test standard matrix sets spread threshold."""
        matrix = create_standard_policy_matrix(environment="sandbox")

        btc = matrix.get_symbol_policy("BTC-USD")
        assert btc.spread_threshold_bps == Decimal("20")  # 2bps threshold

    def test_standard_matrix_environment(self):
        """Test standard matrix uses specified environment."""
        matrix = create_standard_policy_matrix(environment="live")

        assert matrix.environment == "live"

        btc = matrix.get_symbol_policy("BTC-USD")
        assert btc.environment == "live"
