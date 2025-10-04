"""
Unit tests for OrderRecommender.

Tests order configuration recommendation logic based on
urgency and market conditions.
"""

from decimal import Decimal

import pytest

from bot_v2.features.live_trade.order_policy import SymbolPolicy
from bot_v2.features.live_trade.order_recommender import OrderRecommender


class TestDefaultConfiguration:
    """Tests for default configuration."""

    @pytest.fixture
    def symbol_policy(self):
        """Create basic symbol policy."""
        return SymbolPolicy(
            symbol="BTC-USD",
            environment="sandbox",
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
        )

    def test_normal_urgency_default_config(self, symbol_policy):
        """Test normal urgency returns default LIMIT/GTC config."""
        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="normal",
            market_conditions=None,
        )

        assert config["order_type"] == "LIMIT"
        assert config["tif"] == "GTC"
        assert config["post_only"] is False
        assert config["reduce_only"] is False
        assert config["use_market"] is False

    def test_no_urgency_specified(self, symbol_policy):
        """Test default urgency returns LIMIT/GTC."""
        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy, side="buy", quantity=Decimal("1.0")
        )

        assert config["order_type"] == "LIMIT"
        assert config["tif"] == "GTC"


class TestUrgentUrgency:
    """Tests for urgent urgency handling."""

    @pytest.fixture
    def symbol_policy(self):
        """Create basic symbol policy."""
        return SymbolPolicy(
            symbol="BTC-USD",
            environment="sandbox",
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
        )

    def test_urgent_with_good_liquidity_uses_market(self, symbol_policy):
        """Test urgent + good liquidity uses MARKET/IOC."""
        market_conditions = {"liquidity_condition": "good"}

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="urgent",
            market_conditions=market_conditions,
        )

        assert config["order_type"] == "MARKET"
        assert config["tif"] == "IOC"
        assert config["use_market"] is True

    def test_urgent_with_excellent_liquidity_uses_market(self, symbol_policy):
        """Test urgent + excellent liquidity uses MARKET/IOC."""
        market_conditions = {"liquidity_condition": "excellent"}

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="urgent",
            market_conditions=market_conditions,
        )

        assert config["order_type"] == "MARKET"
        assert config["tif"] == "IOC"
        assert config["use_market"] is True

    def test_urgent_with_poor_liquidity_uses_limit_ioc(self, symbol_policy):
        """Test urgent + poor liquidity uses LIMIT/IOC."""
        market_conditions = {"liquidity_condition": "poor"}

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="urgent",
            market_conditions=market_conditions,
        )

        assert config["order_type"] == "LIMIT"
        assert config["tif"] == "IOC"
        assert config["use_market"] is False

    def test_urgent_without_market_conditions_uses_limit_ioc(self, symbol_policy):
        """Test urgent without market conditions uses LIMIT/IOC."""
        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="urgent",
            market_conditions=None,
        )

        assert config["order_type"] == "LIMIT"
        assert config["tif"] == "IOC"


class TestPatientUrgency:
    """Tests for patient urgency handling."""

    @pytest.fixture
    def symbol_policy(self):
        """Create basic symbol policy."""
        return SymbolPolicy(
            symbol="BTC-USD",
            environment="sandbox",
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
        )

    def test_patient_urgency_sets_post_only(self, symbol_policy):
        """Test patient urgency sets post_only flag."""
        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="patient",
        )

        assert config["post_only"] is True
        assert config["order_type"] == "LIMIT"
        assert config["tif"] == "GTC"


class TestSpreadConditions:
    """Tests for spread-based adjustments."""

    @pytest.fixture
    def symbol_policy_with_threshold(self):
        """Create policy with spread threshold."""
        return SymbolPolicy(
            symbol="BTC-USD",
            environment="sandbox",
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
            spread_threshold_bps=Decimal("20"),  # 20 bps threshold
        )

    def test_wide_spread_forces_post_only(self, symbol_policy_with_threshold):
        """Test wide spread forces post-only LIMIT order."""
        market_conditions = {"spread_bps": 25}  # Above 20 bps threshold

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy_with_threshold,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="normal",
            market_conditions=market_conditions,
        )

        assert config["post_only"] is True
        assert config["order_type"] == "LIMIT"
        assert config["use_market"] is False

    def test_narrow_spread_no_adjustment(self, symbol_policy_with_threshold):
        """Test narrow spread doesn't force post-only."""
        market_conditions = {"spread_bps": 10}  # Below 20 bps threshold

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy_with_threshold,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="normal",
            market_conditions=market_conditions,
        )

        assert config["post_only"] is False

    def test_spread_at_threshold_no_adjustment(self, symbol_policy_with_threshold):
        """Test spread at exact threshold doesn't trigger post-only."""
        market_conditions = {"spread_bps": 20}  # At threshold

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy_with_threshold,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="normal",
            market_conditions=market_conditions,
        )

        assert config["post_only"] is False

    def test_invalid_spread_bps_ignored(self, symbol_policy_with_threshold):
        """Test invalid spread_bps is safely ignored."""
        market_conditions = {"spread_bps": "invalid"}

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy_with_threshold,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="normal",
            market_conditions=market_conditions,
        )

        # Should use default config
        assert config["post_only"] is False


class TestVolatilityConditions:
    """Tests for volatility-based adjustments."""

    @pytest.fixture
    def symbol_policy(self):
        """Create basic symbol policy."""
        return SymbolPolicy(
            symbol="BTC-USD",
            environment="sandbox",
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
        )

    def test_high_volatility_uses_ioc(self, symbol_policy):
        """Test high volatility (>90 percentile) uses IOC."""
        market_conditions = {"volatility_percentile": 95}

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="normal",
            market_conditions=market_conditions,
        )

        assert config["tif"] == "IOC"

    def test_volatility_at_threshold_uses_ioc(self, symbol_policy):
        """Test volatility at 90 doesn't trigger IOC."""
        market_conditions = {"volatility_percentile": 90}

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="normal",
            market_conditions=market_conditions,
        )

        assert config["tif"] == "GTC"

    def test_low_volatility_no_adjustment(self, symbol_policy):
        """Test low volatility keeps GTC."""
        market_conditions = {"volatility_percentile": 50}

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="normal",
            market_conditions=market_conditions,
        )

        assert config["tif"] == "GTC"

    def test_invalid_volatility_ignored(self, symbol_policy):
        """Test invalid volatility is safely ignored."""
        market_conditions = {"volatility_percentile": "invalid"}

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="normal",
            market_conditions=market_conditions,
        )

        assert config["tif"] == "GTC"


class TestCombinedConditions:
    """Tests for combined urgency + market conditions."""

    @pytest.fixture
    def symbol_policy_with_threshold(self):
        """Create policy with spread threshold."""
        return SymbolPolicy(
            symbol="BTC-USD",
            environment="sandbox",
            min_order_size=Decimal("0.001"),
            size_increment=Decimal("0.001"),
            spread_threshold_bps=Decimal("20"),
        )

    def test_urgent_overridden_by_wide_spread(self, symbol_policy_with_threshold):
        """Test wide spread overrides urgent market order preference."""
        market_conditions = {
            "liquidity_condition": "good",
            "spread_bps": 25,  # Wide spread
        }

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy_with_threshold,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="urgent",
            market_conditions=market_conditions,
        )

        # Should be LIMIT (not MARKET) due to wide spread
        assert config["order_type"] == "LIMIT"
        assert config["post_only"] is True
        assert config["use_market"] is False

    def test_patient_with_high_volatility(self, symbol_policy_with_threshold):
        """Test patient + high volatility combines post-only and IOC."""
        market_conditions = {"volatility_percentile": 95}

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy_with_threshold,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="patient",
            market_conditions=market_conditions,
        )

        # Patient sets post_only, volatility sets IOC
        assert config["post_only"] is True
        assert config["tif"] == "IOC"

    def test_all_conditions_combined(self, symbol_policy_with_threshold):
        """Test all conditions interact correctly."""
        market_conditions = {
            "liquidity_condition": "good",
            "spread_bps": 25,  # Wide spread
            "volatility_percentile": 95,  # High volatility
        }

        config = OrderRecommender.recommend_config(
            symbol_policy=symbol_policy_with_threshold,
            side="buy",
            quantity=Decimal("1.0"),
            urgency="urgent",
            market_conditions=market_conditions,
        )

        # Wide spread wins: LIMIT, post-only
        # Volatility sets IOC
        assert config["order_type"] == "LIMIT"
        assert config["post_only"] is True
        assert config["tif"] == "IOC"
        assert config["use_market"] is False
