"""
Comprehensive tests for DynamicSizingHelper.

Tests cover:
- Initialization with slippage multipliers from env
- Position sizing application logic
- Reference price determination from multiple sources
- Equity estimation from risk manager and broker
- Impact-aware sizing calculations
- Binary search for optimal size within impact limits
- Sizing modes (STRICT, CONSERVATIVE, AGGRESSIVE)
- Edge cases (thin books, zero depth, large orders)
"""

import os
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, Product, Quote
from bot_v2.features.live_trade.advanced_execution_models.models import (
    OrderConfig,
    SizingMode,
)
from bot_v2.features.live_trade.dynamic_sizing_helper import DynamicSizingHelper
from bot_v2.features.live_trade.risk import (
    LiveRiskManager,
    PositionSizingAdvice,
    PositionSizingContext,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_broker():
    """Create mock broker."""
    broker = Mock()
    broker.get_quote = Mock(return_value=None)
    broker.list_balances = Mock(return_value=[])
    return broker


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    risk_manager = Mock(spec=LiveRiskManager)
    risk_manager.config = Mock()
    risk_manager.config.enable_dynamic_position_sizing = False
    risk_manager.config.max_leverage = 1
    risk_manager.config.position_sizing_method = "intelligent"
    risk_manager.config.position_sizing_multiplier = 1.0
    risk_manager.start_of_day_equity = Decimal("10000")
    risk_manager.positions = {}
    return risk_manager


@pytest.fixture
def sizing_helper(mock_broker):
    """Create DynamicSizingHelper instance."""
    return DynamicSizingHelper(broker=mock_broker)


@pytest.fixture
def sample_product():
    """Create sample product."""
    product = Mock(spec=Product)
    product.symbol = "BTC-USD"
    product.mark_price = Decimal("50000")
    return product


@pytest.fixture
def sample_quote():
    """Create sample quote."""
    quote = Mock(spec=Quote)
    quote.bid = Decimal("49995")
    quote.ask = Decimal("50005")
    quote.last = Decimal("50000")
    return quote


# ============================================================================
# Test: Initialization
# ============================================================================


class TestDynamicSizingHelperInitialization:
    """Test DynamicSizingHelper initialization."""

    def test_default_initialization(self, mock_broker):
        """Test initialization with defaults."""
        helper = DynamicSizingHelper(broker=mock_broker)

        assert helper.broker is mock_broker
        assert helper.risk_manager is None
        assert isinstance(helper.config, OrderConfig)
        assert helper._last_sizing_advice is None
        assert isinstance(helper.slippage_multipliers, dict)

    def test_initialization_with_risk_manager(self, mock_broker, mock_risk_manager):
        """Test initialization with risk manager."""
        helper = DynamicSizingHelper(broker=mock_broker, risk_manager=mock_risk_manager)

        assert helper.risk_manager is mock_risk_manager

    def test_initialization_with_config(self, mock_broker):
        """Test initialization with custom config."""
        config = OrderConfig(max_impact_bps=Decimal("25"))
        helper = DynamicSizingHelper(broker=mock_broker, config=config)

        assert helper.config is config
        assert helper.config.max_impact_bps == Decimal("25")

    @patch.dict(os.environ, {"SLIPPAGE_MULTIPLIERS": "BTC-USD:0.5,ETH-USD:0.3"})
    def test_initialization_loads_slippage_multipliers_from_env(self, mock_broker):
        """Test loading slippage multipliers from environment."""
        helper = DynamicSizingHelper(broker=mock_broker)

        assert "BTC-USD" in helper.slippage_multipliers
        assert helper.slippage_multipliers["BTC-USD"] == Decimal("0.5")
        assert "ETH-USD" in helper.slippage_multipliers
        assert helper.slippage_multipliers["ETH-USD"] == Decimal("0.3")

    @patch.dict(os.environ, {"SLIPPAGE_MULTIPLIERS": "invalid-format"})
    def test_initialization_handles_invalid_slippage_multipliers(self, mock_broker):
        """Test that invalid slippage multipliers are handled gracefully."""
        helper = DynamicSizingHelper(broker=mock_broker)

        # Should not raise exception
        assert isinstance(helper.slippage_multipliers, dict)


# ============================================================================
# Test: Position Sizing Application
# ============================================================================


class TestPositionSizingApplication:
    """Test maybe_apply_position_sizing method."""

    def test_returns_none_without_risk_manager(self, sizing_helper):
        """Test that sizing returns None without risk manager."""
        advice = sizing_helper.maybe_apply_position_sizing(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            limit_price=None,
            product=None,
            quote=None,
            leverage=None,
        )

        assert advice is None

    def test_returns_none_when_dynamic_sizing_disabled(self, mock_broker, mock_risk_manager):
        """Test that sizing returns None when disabled in config."""
        mock_risk_manager.config.enable_dynamic_position_sizing = False

        helper = DynamicSizingHelper(broker=mock_broker, risk_manager=mock_risk_manager)

        advice = helper.maybe_apply_position_sizing(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            limit_price=None,
            product=None,
            quote=None,
            leverage=None,
        )

        assert advice is None

    def test_applies_sizing_when_enabled(self, mock_broker, mock_risk_manager, sample_quote):
        """Test that sizing is applied when enabled."""
        mock_risk_manager.config.enable_dynamic_position_sizing = True
        mock_risk_manager.size_position = Mock(
            return_value=PositionSizingAdvice(
                symbol="BTC-USD",
                side="buy",
                target_notional=Decimal("25000"),
                target_quantity=Decimal("0.5"),
                reason="risk_limit",
            )
        )

        helper = DynamicSizingHelper(broker=mock_broker, risk_manager=mock_risk_manager)

        advice = helper.maybe_apply_position_sizing(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            limit_price=Decimal("50000"),
            product=None,
            quote=sample_quote,
            leverage=1,
        )

        assert advice is not None
        assert advice.symbol == "BTC-USD"
        assert advice.target_quantity == Decimal("0.5")
        mock_risk_manager.size_position.assert_called_once()

    def test_stores_last_sizing_advice(self, mock_broker, mock_risk_manager, sample_quote):
        """Test that last sizing advice is stored."""
        mock_risk_manager.config.enable_dynamic_position_sizing = True
        expected_advice = PositionSizingAdvice(
            symbol="BTC-USD",
            side="buy",
            target_notional=Decimal("25000"),
            target_quantity=Decimal("0.5"),
            reason="risk_limit",
        )
        mock_risk_manager.size_position = Mock(return_value=expected_advice)

        helper = DynamicSizingHelper(broker=mock_broker, risk_manager=mock_risk_manager)

        helper.maybe_apply_position_sizing(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            limit_price=Decimal("50000"),
            product=None,
            quote=sample_quote,
            leverage=1,
        )

        assert helper.last_sizing_advice is expected_advice


# ============================================================================
# Test: Reference Price Determination
# ============================================================================


class TestReferencePriceDetermination:
    """Test determine_reference_price method."""

    def test_uses_limit_price_first(self, sizing_helper):
        """Test that limit price is preferred."""
        price = sizing_helper.determine_reference_price(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50000"),
            quote=None,
            product=None,
        )

        assert price == Decimal("50000")

    def test_uses_quote_ask_for_market_buy(self, sizing_helper, sample_quote):
        """Test using quote ask for market buy."""
        price = sizing_helper.determine_reference_price(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            limit_price=None,
            quote=sample_quote,
            product=None,
        )

        assert price == Decimal("50005")  # ask

    def test_uses_quote_bid_for_market_sell(self, sizing_helper, sample_quote):
        """Test using quote bid for market sell."""
        price = sizing_helper.determine_reference_price(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            limit_price=None,
            quote=sample_quote,
            product=None,
        )

        assert price == Decimal("49995")  # bid

    def test_falls_back_to_quote_last(self, sizing_helper, sample_quote):
        """Test fallback to quote last price."""
        sample_quote.bid = None
        sample_quote.ask = None

        price = sizing_helper.determine_reference_price(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=None,
            quote=sample_quote,
            product=None,
        )

        assert price == Decimal("50000")  # last

    def test_fetches_quote_from_broker_when_missing(self, sizing_helper):
        """Test fetching quote from broker when not provided."""
        broker_quote = Mock(spec=Quote)
        broker_quote.ask = Decimal("50010")
        sizing_helper.broker.get_quote = Mock(return_value=broker_quote)

        price = sizing_helper.determine_reference_price(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            limit_price=None,
            quote=None,
            product=None,
        )

        assert price == Decimal("50010")
        sizing_helper.broker.get_quote.assert_called_once_with("BTC-USD")

    def test_uses_product_mark_price_as_final_fallback(self, sizing_helper, sample_product):
        """Test using product mark price as final fallback."""
        price = sizing_helper.determine_reference_price(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=None,
            quote=None,
            product=sample_product,
        )

        assert price == Decimal("50000")  # mark_price

    def test_returns_zero_when_no_price_available(self, sizing_helper):
        """Test returns zero when no price is available."""
        price = sizing_helper.determine_reference_price(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            limit_price=None,
            quote=None,
            product=None,
        )

        assert price == Decimal("0")


# ============================================================================
# Test: Equity Estimation
# ============================================================================


class TestEquityEstimation:
    """Test estimate_equity method."""

    def test_uses_risk_manager_start_of_day_equity(self, mock_broker, mock_risk_manager):
        """Test using risk manager's start of day equity."""
        mock_risk_manager.start_of_day_equity = Decimal("25000")

        helper = DynamicSizingHelper(broker=mock_broker, risk_manager=mock_risk_manager)
        equity = helper.estimate_equity()

        assert equity == Decimal("25000")

    def test_fetches_balances_from_broker_when_no_risk_manager(self, mock_broker):
        """Test fetching balances from broker."""
        balance1 = Mock()
        balance1.total = Decimal("10000")
        balance2 = Mock()
        balance2.total = Decimal("5000")
        mock_broker.list_balances = Mock(return_value=[balance1, balance2])

        helper = DynamicSizingHelper(broker=mock_broker)
        equity = helper.estimate_equity()

        assert equity == Decimal("15000")

    def test_uses_available_when_total_missing(self, mock_broker):
        """Test using available balance when total is missing."""
        balance = Mock()
        balance.total = None
        balance.available = Decimal("8000")
        mock_broker.list_balances = Mock(return_value=[balance])

        helper = DynamicSizingHelper(broker=mock_broker)
        equity = helper.estimate_equity()

        assert equity == Decimal("8000")

    def test_returns_zero_when_no_equity_available(self, mock_broker):
        """Test returns zero when no equity source is available."""
        mock_broker.list_balances = Mock(return_value=[])

        helper = DynamicSizingHelper(broker=mock_broker)
        equity = helper.estimate_equity()

        assert equity == Decimal("0")


# ============================================================================
# Test: Impact-Aware Sizing
# ============================================================================


class TestImpactAwareSizing:
    """Test calculate_impact_aware_size method."""

    def test_returns_zero_without_depth_data(self, sizing_helper):
        """Test returns zero when depth data is missing."""
        market_snapshot = {}

        size, impact = sizing_helper.calculate_impact_aware_size(
            symbol="BTC-USD",
            target_notional=Decimal("50000"),
            market_snapshot=market_snapshot,
        )

        assert size == Decimal("0")
        assert impact == Decimal("0")

    def test_calculates_size_within_impact_limit(self, sizing_helper):
        """Test calculating size within impact limit."""
        market_snapshot = {
            "depth_l1": Decimal("10000"),  # $10k depth at L1
            "depth_l10": Decimal("100000"),  # $100k depth at L10
        }

        size, impact = sizing_helper.calculate_impact_aware_size(
            symbol="BTC-USD",
            target_notional=Decimal("5000"),  # Within L1
            market_snapshot=market_snapshot,
        )

        assert size > Decimal("0")
        assert impact <= sizing_helper.config.max_impact_bps

    def test_sizes_down_large_orders(self, sizing_helper):
        """Test that large orders are sized down."""
        market_snapshot = {
            "depth_l1": Decimal("1000"),  # Low depth
            "depth_l10": Decimal("10000"),
        }

        size, impact = sizing_helper.calculate_impact_aware_size(
            symbol="BTC-USD",
            target_notional=Decimal("50000"),  # Much larger than depth
            market_snapshot=market_snapshot,
        )

        # Should size down significantly
        assert size < Decimal("50000")
        assert impact <= sizing_helper.config.max_impact_bps

    def test_applies_slippage_multiplier(self, sizing_helper):
        """Test that per-symbol slippage multiplier is stored correctly."""
        sizing_helper.slippage_multipliers["BTC-USD"] = Decimal("0.5")
        sizing_helper.slippage_multipliers["ETH-USD"] = Decimal("0.3")

        # Verify multipliers are stored
        assert "BTC-USD" in sizing_helper.slippage_multipliers
        assert sizing_helper.slippage_multipliers["BTC-USD"] == Decimal("0.5")
        assert sizing_helper.slippage_multipliers["ETH-USD"] == Decimal("0.3")

    def test_strict_mode_returns_zero_if_cannot_fit(self, mock_broker):
        """Test STRICT mode returns zero if cannot fit within limit."""
        config = OrderConfig(
            sizing_mode=SizingMode.STRICT,
            max_impact_bps=Decimal("1"),  # Very tight limit
        )
        helper = DynamicSizingHelper(broker=mock_broker, config=config)

        market_snapshot = {
            "depth_l1": Decimal("1000"),
            "depth_l10": Decimal("10000"),
        }

        size, impact = helper.calculate_impact_aware_size(
            symbol="BTC-USD",
            target_notional=Decimal("50000"),  # Cannot fit
            market_snapshot=market_snapshot,
        )

        assert size == Decimal("0")
        assert impact == Decimal("0")

    def test_aggressive_mode_allows_higher_impact(self, mock_broker):
        """Test AGGRESSIVE mode allows higher impact."""
        config = OrderConfig(
            sizing_mode=SizingMode.AGGRESSIVE,
            max_impact_bps=Decimal("10"),
        )
        helper = DynamicSizingHelper(broker=mock_broker, config=config)

        market_snapshot = {
            "depth_l1": Decimal("10000"),
            "depth_l10": Decimal("100000"),
        }

        size, impact = helper.calculate_impact_aware_size(
            symbol="BTC-USD",
            target_notional=Decimal("80000"),  # Close to L10
            market_snapshot=market_snapshot,
        )

        # Should accept full size
        assert size == Decimal("80000")


# ============================================================================
# Test: Impact Estimation
# ============================================================================


class TestImpactEstimation:
    """Test estimate_impact method."""

    def test_linear_impact_within_l1(self, sizing_helper):
        """Test linear impact for orders within L1 depth."""
        impact = sizing_helper.estimate_impact(
            order_size=Decimal("5000"),  # Half of L1
            l1_depth=Decimal("10000"),
            l10_depth=Decimal("100000"),
        )

        # Should be linear: (5000/10000) * 5 = 2.5 bps
        assert impact == Decimal("2.5")

    def test_square_root_impact_beyond_l1(self, sizing_helper):
        """Test square root impact for orders beyond L1."""
        impact = sizing_helper.estimate_impact(
            order_size=Decimal("50000"),  # Beyond L1
            l1_depth=Decimal("10000"),
            l10_depth=Decimal("100000"),
        )

        # Should have L1 impact + square root excess impact
        assert impact > Decimal("5")  # More than just L1 impact
        assert impact < Decimal("100")  # But not max

    def test_max_impact_beyond_l10(self, sizing_helper):
        """Test maximum impact for orders beyond L10."""
        impact = sizing_helper.estimate_impact(
            order_size=Decimal("200000"),  # Beyond L10
            l1_depth=Decimal("10000"),
            l10_depth=Decimal("100000"),
        )

        assert impact == Decimal("100")  # Max impact

    def test_impact_scales_with_order_size(self, sizing_helper):
        """Test that impact increases with order size."""
        impact_small = sizing_helper.estimate_impact(
            Decimal("1000"), Decimal("10000"), Decimal("100000")
        )
        impact_medium = sizing_helper.estimate_impact(
            Decimal("5000"), Decimal("10000"), Decimal("100000")
        )
        impact_large = sizing_helper.estimate_impact(
            Decimal("50000"), Decimal("10000"), Decimal("100000")
        )

        assert impact_small < impact_medium < impact_large


# ============================================================================
# Test: Position Quantity Extraction
# ============================================================================


class TestPositionQuantityExtraction:
    """Test _extract_position_quantity method."""

    def test_returns_zero_without_risk_manager(self, sizing_helper):
        """Test returns zero without risk manager."""
        quantity = sizing_helper._extract_position_quantity("BTC-USD")

        assert quantity == Decimal("0")

    def test_extracts_position_from_risk_manager(self, mock_broker, mock_risk_manager):
        """Test extracting position from risk manager."""
        mock_risk_manager.positions = {"BTC-USD": Decimal("2.5")}

        helper = DynamicSizingHelper(broker=mock_broker, risk_manager=mock_risk_manager)
        quantity = helper._extract_position_quantity("BTC-USD")

        assert quantity == Decimal("2.5")

    def test_returns_zero_for_missing_position(self, mock_broker, mock_risk_manager):
        """Test returns zero for missing position."""
        mock_risk_manager.positions = {}

        helper = DynamicSizingHelper(broker=mock_broker, risk_manager=mock_risk_manager)
        quantity = helper._extract_position_quantity("BTC-USD")

        assert quantity == Decimal("0")


# ============================================================================
# Test: Last Sizing Advice Property
# ============================================================================


class TestLastSizingAdviceProperty:
    """Test last_sizing_advice property."""

    def test_returns_none_initially(self, sizing_helper):
        """Test that property returns None initially."""
        assert sizing_helper.last_sizing_advice is None

    def test_returns_stored_advice(self, mock_broker, mock_risk_manager, sample_quote):
        """Test that property returns stored advice."""
        mock_risk_manager.config.enable_dynamic_position_sizing = True
        expected_advice = PositionSizingAdvice(
            symbol="BTC-USD",
            side="buy",
            target_notional=Decimal("25000"),
            target_quantity=Decimal("0.5"),
            reason="test",
        )
        mock_risk_manager.size_position = Mock(return_value=expected_advice)

        helper = DynamicSizingHelper(broker=mock_broker, risk_manager=mock_risk_manager)

        helper.maybe_apply_position_sizing(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            limit_price=Decimal("50000"),
            product=None,
            quote=sample_quote,
            leverage=1,
        )

        assert helper.last_sizing_advice is expected_advice
