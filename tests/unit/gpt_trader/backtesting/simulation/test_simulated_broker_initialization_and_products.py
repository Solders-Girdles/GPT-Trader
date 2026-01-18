"""Tests for SimulatedBroker initialization and product management."""

from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.types import FeeTier
from gpt_trader.core import MarketType, Product


class TestSimulatedBrokerInitialization:
    """Test SimulatedBroker initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization parameters."""
        broker = SimulatedBroker()

        assert broker.equity == Decimal("100000")
        assert broker.fee_tier == FeeTier.TIER_2
        assert broker.connected is False
        assert len(broker.products) == 0
        assert len(broker.positions) == 0

    def test_custom_equity_initialization(self) -> None:
        """Test initialization with custom equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("50000"))

        assert broker.equity == Decimal("50000")
        assert broker.get_equity() == Decimal("50000")

    def test_custom_fee_tier_initialization(self) -> None:
        """Test initialization with custom fee tier."""
        broker = SimulatedBroker(fee_tier=FeeTier.TIER_5)

        assert broker.fee_tier == FeeTier.TIER_5

    def test_initial_balance_matches_equity(self) -> None:
        """Test that initial USDC balance matches equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("75000"))

        balances = broker.list_balances()
        assert len(balances) == 1
        assert balances[0].asset == "USDC"
        assert balances[0].total == Decimal("75000")
        assert balances[0].available == Decimal("75000")


class TestSimulatedBrokerProductManagement:
    """Test product registration and retrieval."""

    @pytest.fixture
    def sample_product(self) -> Product:
        """Create a sample product for testing."""
        return Product(
            symbol="BTC-PERP-USDC",
            base_asset="BTC",
            quote_asset="USDC",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.0001"),
            step_size=Decimal("0.0001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=10,
        )

    def test_register_product(self, sample_product: Product) -> None:
        """Test product registration."""
        broker = SimulatedBroker()

        broker.register_product(sample_product)

        assert "BTC-PERP-USDC" in broker.products
        assert broker.products["BTC-PERP-USDC"] == sample_product

    def test_get_product_exists(self, sample_product: Product) -> None:
        """Test retrieving a registered product."""
        broker = SimulatedBroker()
        broker.register_product(sample_product)

        retrieved = broker.get_product("BTC-PERP-USDC")

        assert retrieved is not None
        assert retrieved.symbol == "BTC-PERP-USDC"
        assert retrieved.base_asset == "BTC"

    def test_get_product_not_exists(self) -> None:
        """Test retrieving a non-existent product returns None."""
        broker = SimulatedBroker()

        retrieved = broker.get_product("NONEXISTENT-USD")

        assert retrieved is None

    def test_register_multiple_products(self) -> None:
        """Test registering multiple products."""
        broker = SimulatedBroker()

        products = [
            Product(
                symbol="BTC-PERP-USDC",
                base_asset="BTC",
                quote_asset="USDC",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.0001"),
                step_size=Decimal("0.0001"),
                min_notional=Decimal("10"),
                price_increment=Decimal("0.01"),
                leverage_max=10,
            ),
            Product(
                symbol="ETH-PERP-USDC",
                base_asset="ETH",
                quote_asset="USDC",
                market_type=MarketType.PERPETUAL,
                min_size=Decimal("0.001"),
                step_size=Decimal("0.001"),
                min_notional=Decimal("10"),
                price_increment=Decimal("0.01"),
                leverage_max=10,
            ),
        ]

        for product in products:
            broker.register_product(product)

        assert len(broker.products) == 2
        assert broker.get_product("BTC-PERP-USDC") is not None
        assert broker.get_product("ETH-PERP-USDC") is not None

    def test_overwrite_product(self, sample_product: Product) -> None:
        """Test that registering same symbol overwrites previous product."""
        broker = SimulatedBroker()
        broker.register_product(sample_product)

        # Create updated product with same symbol
        updated_product = Product(
            symbol="BTC-PERP-USDC",
            base_asset="BTC",
            quote_asset="USDC",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),  # Changed
            step_size=Decimal("0.001"),
            min_notional=Decimal("20"),  # Changed
            price_increment=Decimal("0.01"),
            leverage_max=10,
        )
        broker.register_product(updated_product)

        retrieved = broker.get_product("BTC-PERP-USDC")
        assert retrieved.min_size == Decimal("0.001")
        assert retrieved.min_notional == Decimal("20")
