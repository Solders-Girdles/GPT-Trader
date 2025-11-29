"""Tests for orchestration/execution/state_collection.py."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.brokerages.core.interfaces import Balance, MarketType, Product
from gpt_trader.features.live_trade.risk import ValidationError
from gpt_trader.orchestration.execution.state_collection import StateCollector

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_broker() -> MagicMock:
    """Create a mock broker."""
    broker = MagicMock()
    broker.list_balances = MagicMock(return_value=[])
    broker.list_positions = MagicMock(return_value=[])
    broker.get_product = MagicMock(return_value=None)
    return broker


@pytest.fixture
def mock_config(bot_config_factory):
    """Create mock BotConfig for state collection tests."""
    return bot_config_factory()


@pytest.fixture
def collector(mock_broker: MagicMock, mock_config, monkeypatch) -> StateCollector:
    """Create a StateCollector instance."""
    monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "USD,USDC")
    return StateCollector(mock_broker, mock_config)


@pytest.fixture
def mock_product() -> Product:
    """Create a mock product."""
    return Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=20,
    )


# ============================================================
# Test: __init__
# ============================================================


class TestStateCollectorInit:
    """Tests for StateCollector initialization."""

    def test_init_stores_broker(self, mock_broker: MagicMock, mock_config) -> None:
        """Test that broker is stored correctly."""
        collector = StateCollector(mock_broker, mock_config)
        assert collector.broker is mock_broker

    def test_init_resolves_collateral_assets(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test that collateral assets are resolved from environment."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "USDT,DAI,USDC")
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USDT", "DAI", "USDC"}

    def test_init_uses_defaults_when_no_env(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test that default collateral assets are used when env is empty."""
        monkeypatch.delenv("PERPS_COLLATERAL_ASSETS", raising=False)
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USD", "USDC"}

    def test_init_integration_mode_explicit_parameter(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test integration mode can be set via explicit parameter."""
        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        assert collector._integration_mode is True

    def test_init_integration_mode_defaults_false(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test integration_mode defaults to False."""
        collector = StateCollector(mock_broker, mock_config)
        assert collector._integration_mode is False


# ============================================================
# Test: _resolve_collateral_assets
# ============================================================


class TestResolveCollateralAssets:
    """Tests for _resolve_collateral_assets method."""

    def test_parses_comma_separated_values(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test parsing comma-separated collateral assets."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "USD, USDC, USDT")
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USD", "USDC", "USDT"}

    def test_uppercase_normalization(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test that asset names are uppercased."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "usd,usdc")
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USD", "USDC"}

    def test_empty_returns_defaults(self, mock_broker: MagicMock, mock_config, monkeypatch) -> None:
        """Test that empty env returns defaults."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "")
        collector = StateCollector(mock_broker, mock_config)
        assert collector.collateral_assets == {"USD", "USDC"}


# ============================================================
# Test: calculate_equity_from_balances
# ============================================================


class TestCalculateEquityFromBalances:
    """Tests for calculate_equity_from_balances method."""

    def test_sums_collateral_balances(self, collector: StateCollector) -> None:
        """Test that collateral balances are summed correctly."""
        balances = [
            Balance(asset="USD", total=Decimal("100"), available=Decimal("50")),
            Balance(asset="USDC", total=Decimal("200"), available=Decimal("100")),
            Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
        ]

        available, collateral, total = collector.calculate_equity_from_balances(balances)

        assert available == Decimal("150")
        assert total == Decimal("300")
        assert len(collateral) == 2

    def test_falls_back_to_usd_balance(self, collector: StateCollector) -> None:
        """Test fallback to USD balance when no collateral matches."""
        collector.collateral_assets = {"NONEXISTENT"}
        balances = [
            Balance(asset="USD", total=Decimal("1000"), available=Decimal("800")),
            Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
        ]

        available, collateral, total = collector.calculate_equity_from_balances(balances)

        assert available == Decimal("800")
        assert total == Decimal("1000")
        assert len(collateral) == 1
        assert collateral[0].asset == "USD"

    def test_returns_zeros_when_no_balances_match(self, collector: StateCollector) -> None:
        """Test returns zeros when no balances match."""
        collector.collateral_assets = {"NONEXISTENT"}
        balances = [
            Balance(asset="BTC", total=Decimal("1"), available=Decimal("1")),
        ]

        available, collateral, total = collector.calculate_equity_from_balances(balances)

        assert available == Decimal("0")
        assert total == Decimal("0")
        assert len(collateral) == 0

    def test_handles_empty_balances(self, collector: StateCollector) -> None:
        """Test handling of empty balance list."""
        available, collateral, total = collector.calculate_equity_from_balances([])

        assert available == Decimal("0")
        assert total == Decimal("0")
        assert len(collateral) == 0


# ============================================================
# Test: log_collateral_update
# ============================================================


class TestLogCollateralUpdate:
    """Tests for log_collateral_update method."""

    def test_skips_empty_collateral_list(self, collector: StateCollector) -> None:
        """Test that empty collateral list is skipped."""
        collector._production_logger = MagicMock()

        collector.log_collateral_update([], Decimal("100"), Decimal("100"), [])

        collector._production_logger.log_balance_update.assert_not_called()

    def test_sets_initial_collateral_value(self, collector: StateCollector) -> None:
        """Test that initial collateral value is set."""
        collector._production_logger = MagicMock()
        collateral = [Balance(asset="USD", total=Decimal("100"), available=Decimal("50"))]

        collector.log_collateral_update(collateral, Decimal("100"), Decimal("100"), collateral)

        assert collector._last_collateral_available == Decimal("50")

    def test_logs_significant_change(self, collector: StateCollector) -> None:
        """Test that significant changes are logged."""
        collector._production_logger = MagicMock()
        collector._last_collateral_available = Decimal("50")
        collateral = [Balance(asset="USD", total=Decimal("110"), available=Decimal("60"))]

        collector.log_collateral_update(collateral, Decimal("100"), Decimal("110"), collateral)

        collector._production_logger.log_balance_update.assert_called_once()

    def test_handles_logger_exception(self, collector: StateCollector) -> None:
        """Test that logger exceptions are suppressed."""
        collector._production_logger = MagicMock()
        collector._production_logger.log_balance_update.side_effect = RuntimeError("Log error")
        collateral = [Balance(asset="USD", total=Decimal("100"), available=Decimal("50"))]

        # Should not raise
        collector.log_collateral_update(collateral, Decimal("100"), Decimal("100"), collateral)


# ============================================================
# Test: collect_account_state
# ============================================================


class TestCollectAccountState:
    """Tests for collect_account_state method."""

    def test_collects_balances_and_positions(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test collecting balances and positions."""
        mock_broker.list_balances.return_value = [
            Balance(asset="USD", total=Decimal("1000"), available=Decimal("800"))
        ]
        mock_broker.list_positions.return_value = []

        balances, equity, collateral, total, positions = collector.collect_account_state()

        assert len(balances) == 1
        assert equity == Decimal("800")
        assert len(collateral) == 1
        assert total == Decimal("1000")
        assert positions == []

    def test_handles_broker_without_list_balances(
        self, mock_broker: MagicMock, mock_config
    ) -> None:
        """Test handling broker without list_balances method."""
        del mock_broker.list_balances
        mock_broker.list_positions.return_value = []

        collector = StateCollector(mock_broker, mock_config)
        balances, equity, collateral, total, positions = collector.collect_account_state()

        assert balances == []
        assert equity == Decimal("0")

    def test_handles_balance_exception_in_integration_mode(self, mock_broker: MagicMock) -> None:
        """Test that balance exceptions are suppressed in integration mode."""
        mock_broker.list_balances.side_effect = RuntimeError("API error")
        mock_broker.list_positions.return_value = []

        mock_settings = MagicMock()
        mock_settings.raw_env = {}

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        balances, equity, _, _, _ = collector.collect_account_state()

        # Should use default balance in integration mode
        assert len(balances) == 1
        assert balances[0].asset == "USD"

    def test_raises_balance_exception_in_normal_mode(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test that balance exceptions are raised in normal mode."""
        mock_broker.list_balances.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            collector.collect_account_state()

    def test_handles_position_exception_in_integration_mode(self, mock_broker: MagicMock) -> None:
        """Test that position exceptions are suppressed in integration mode."""
        mock_broker.list_balances.return_value = [
            Balance(asset="USD", total=Decimal("1000"), available=Decimal("800"))
        ]
        mock_broker.list_positions.side_effect = RuntimeError("API error")

        mock_settings = MagicMock()
        mock_settings.raw_env = {}

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        _, _, _, _, positions = collector.collect_account_state()

        assert positions == []

    def test_provides_default_balance_in_integration_mode(self, mock_broker: MagicMock) -> None:
        """Test that default balance is provided in integration mode."""
        mock_broker.list_balances.return_value = []
        mock_broker.list_positions.return_value = []

        mock_settings = MagicMock()
        mock_settings.raw_env = {}

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        balances, equity, _, _, _ = collector.collect_account_state()

        assert len(balances) == 1
        assert equity == Decimal("100000")


# ============================================================
# Test: build_positions_dict
# ============================================================


class TestBuildPositionsDict:
    """Tests for build_positions_dict method."""

    def test_builds_dict_from_positions(self, collector: StateCollector) -> None:
        """Test building position dictionary."""
        positions = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("1.5"),
                side="long",
                entry_price=Decimal("50000"),
                mark_price=Decimal("51000"),
            ),
        ]

        result = collector.build_positions_dict(positions)

        assert "BTC-PERP" in result
        assert result["BTC-PERP"]["quantity"] == Decimal("1.5")
        assert result["BTC-PERP"]["side"] == "long"
        assert result["BTC-PERP"]["entry_price"] == Decimal("50000")
        assert result["BTC-PERP"]["mark_price"] == Decimal("51000")

    def test_skips_zero_quantity_positions(self, collector: StateCollector) -> None:
        """Test that zero quantity positions are skipped."""
        positions = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("0"),
                side="long",
                entry_price=Decimal("50000"),
                mark_price=Decimal("51000"),
            ),
        ]

        result = collector.build_positions_dict(positions)

        assert len(result) == 0

    def test_handles_parse_errors(self, collector: StateCollector) -> None:
        """Test that parse errors are handled gracefully."""
        positions = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("1.5"),
                side="long",
                entry_price="invalid",  # Will cause Decimal conversion error
                mark_price=Decimal("51000"),
            ),
        ]

        result = collector.build_positions_dict(positions)

        assert len(result) == 0

    def test_defaults_side_to_long(self, collector: StateCollector) -> None:
        """Test that side defaults to 'long'."""
        positions = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("1.5"),
                # No side attribute
                entry_price=Decimal("50000"),
                mark_price=Decimal("51000"),
            ),
        ]

        result = collector.build_positions_dict(positions)

        assert result["BTC-PERP"]["side"] == "long"


# ============================================================
# Test: resolve_effective_price
# ============================================================


class TestResolveEffectivePrice:
    """Tests for resolve_effective_price method."""

    def test_returns_provided_price(self, collector: StateCollector, mock_product: Product) -> None:
        """Test that provided price is returned directly."""
        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=Decimal("50000"),
            product=mock_product,
        )

        assert result == Decimal("50000")

    def test_uses_mark_price_for_market_orders(
        self, collector: StateCollector, mock_broker: MagicMock, mock_product: Product
    ) -> None:
        """Test that mark price is used for market orders."""
        mock_broker.get_mark_price = MagicMock(return_value=Decimal("51000"))

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=mock_product,
        )

        assert result == Decimal("51000")

    def test_uses_mid_price_fallback(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test fallback to bid/ask mid-price."""
        del mock_broker.get_mark_price

        product = SimpleNamespace(
            symbol="BTC-PERP",
            bid_price=Decimal("49000"),
            ask_price=Decimal("51000"),
            price_increment=Decimal("0.01"),
            quote_increment=Decimal("0.01"),
        )

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=product,
        )

        assert result == Decimal("50000")  # (49000 + 51000) / 2

    def test_uses_broker_quote_fallback(
        self, collector: StateCollector, mock_broker: MagicMock, mock_product: Product
    ) -> None:
        """Test fallback to broker quote."""
        mock_broker.get_mark_price = MagicMock(return_value=None)
        mock_broker.get_quote = MagicMock(return_value=SimpleNamespace(last=Decimal("52000")))

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=mock_product,
        )

        assert result == Decimal("52000")

    def test_uses_product_price_fallback(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test fallback to product price."""
        del mock_broker.get_mark_price
        del mock_broker.get_quote

        product = SimpleNamespace(
            symbol="BTC-PERP",
            price=Decimal("53000"),
            price_increment=Decimal("0.01"),
            quote_increment=Decimal("0.01"),
        )

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=product,
        )

        assert result == Decimal("53000")

    def test_uses_quote_increment_as_last_resort(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test last resort uses quote_increment * 100."""
        del mock_broker.get_mark_price
        del mock_broker.get_quote

        product = SimpleNamespace(
            symbol="BTC-PERP",
            price_increment=Decimal("0.01"),
            quote_increment=Decimal("0.01"),
        )

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=product,
        )

        assert result == Decimal("1")  # 0.01 * 100

    def test_handles_zero_price_as_none(
        self, collector: StateCollector, mock_broker: MagicMock, mock_product: Product
    ) -> None:
        """Test that zero price is treated as None."""
        mock_broker.get_mark_price = MagicMock(return_value=Decimal("51000"))

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=Decimal("0"),
            product=mock_product,
        )

        # Should use mark price instead
        assert result == Decimal("51000")

    def test_handles_mark_price_exception(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test that mark price exceptions are handled."""
        mock_broker.get_mark_price = MagicMock(side_effect=RuntimeError("API error"))
        mock_broker.get_quote = MagicMock(return_value=SimpleNamespace(last=Decimal("52000")))

        product = SimpleNamespace(
            symbol="BTC-PERP",
            price_increment=Decimal("0.01"),
            quote_increment=Decimal("0.01"),
        )

        result = collector.resolve_effective_price(
            symbol="BTC-PERP",
            side="buy",
            price=None,
            product=product,
        )

        assert result == Decimal("52000")


# ============================================================
# Test: require_product
# ============================================================


class TestRequireProduct:
    """Tests for require_product method."""

    def test_returns_provided_product(
        self, collector: StateCollector, mock_product: Product
    ) -> None:
        """Test that provided product is returned directly."""
        result = collector.require_product("BTC-PERP", mock_product)

        assert result is mock_product

    def test_fetches_from_broker_when_none(
        self, collector: StateCollector, mock_broker: MagicMock, mock_product: Product
    ) -> None:
        """Test that product is fetched from broker when None."""
        mock_broker.get_product.return_value = mock_product

        result = collector.require_product("BTC-PERP", None)

        assert result is mock_product
        mock_broker.get_product.assert_called_once_with("BTC-PERP")

    def test_raises_validation_error_when_not_found(
        self, collector: StateCollector, mock_broker: MagicMock
    ) -> None:
        """Test that ValidationError is raised when product not found."""
        mock_broker.get_product.return_value = None

        with pytest.raises(ValidationError, match="Product not found"):
            collector.require_product("UNKNOWN-PERP", None)

    def test_provides_synthetic_product_in_integration_mode(self, mock_broker: MagicMock) -> None:
        """Test that synthetic product is provided in integration mode."""
        mock_broker.get_product.return_value = None

        mock_settings = MagicMock()
        mock_settings.raw_env = {}

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        result = collector.require_product("BTC-PERP", None)

        assert result.symbol == "BTC-PERP"
        assert result.base_asset == "BTC"
        assert result.quote_asset == "PERP"
        assert result.market_type == MarketType.PERPETUAL

    def test_synthetic_product_parses_symbol_without_dash(self, mock_broker: MagicMock) -> None:
        """Test synthetic product parsing when symbol has no dash."""
        mock_broker.get_product.return_value = None

        mock_settings = MagicMock()
        mock_settings.raw_env = {}

        collector = StateCollector(mock_broker, mock_config, integration_mode=True)
        result = collector.require_product("BTCUSD", None)

        assert result.symbol == "BTCUSD"
        assert result.base_asset == "BTCUSD"
        assert result.quote_asset == "USD"


# ============================================================
# Test: Integration scenarios
# ============================================================


class TestStateCollectionIntegration:
    """Integration tests for state collection workflows."""

    def test_full_state_collection_flow(
        self, mock_broker: MagicMock, mock_config, monkeypatch
    ) -> None:
        """Test complete state collection flow."""
        monkeypatch.setenv("PERPS_COLLATERAL_ASSETS", "USD,USDC")
        mock_broker.list_balances.return_value = [
            Balance(asset="USD", total=Decimal("10000"), available=Decimal("8000")),
            Balance(asset="USDC", total=Decimal("5000"), available=Decimal("5000")),
        ]
        mock_broker.list_positions.return_value = [
            SimpleNamespace(
                symbol="BTC-PERP",
                quantity=Decimal("0.5"),
                side="long",
                entry_price=Decimal("50000"),
                mark_price=Decimal("51000"),
            ),
        ]

        collector = StateCollector(mock_broker, mock_config)

        # Collect account state
        balances, equity, collateral, total, positions = collector.collect_account_state()

        # Verify balances
        assert len(balances) == 2
        assert equity == Decimal("13000")  # 8000 + 5000
        assert len(collateral) == 2
        assert total == Decimal("15000")  # 10000 + 5000

        # Build positions dict
        pos_dict = collector.build_positions_dict(positions)

        assert "BTC-PERP" in pos_dict
        assert pos_dict["BTC-PERP"]["quantity"] == Decimal("0.5")
