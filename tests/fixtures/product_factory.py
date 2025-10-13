"""Factory for creating test products and brokers from structured fixtures."""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

import yaml

from bot_v2.features.brokerages.core.interfaces import Balance, MarketType, Product

if TYPE_CHECKING:  # pragma: no cover - typing only
    from bot_v2.orchestration.deterministic_broker import DeterministicBroker

logger = logging.getLogger(__name__)


class ProductFactory:
    """Factory for creating test products from YAML fixtures."""

    def __init__(self, fixture_path: Path | None = None) -> None:
        """Initialize the factory with optional custom fixture path.

        Args:
            fixture_path: Path to YAML fixtures file. Defaults to standard location.
        """
        if fixture_path is None:
            fixture_path = Path(__file__).parent / "mock_products.yaml"

        self.fixture_path = fixture_path
        self._fixtures = self._load_fixtures()

    def _load_fixtures(self) -> dict[str, Any]:
        """Load fixtures from YAML file."""
        try:
            with open(self.fixture_path) as f:
                return yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning(f"Failed to load product fixtures from {self.fixture_path}: {exc}")
            return {}

    def create_product(
        self, symbol: str, market_type: MarketType = MarketType.PERPETUAL
    ) -> Product:
        """Create a product from fixture data.

        Args:
            symbol: Product symbol
            market_type: Market type (PERPETUAL or SPOT)

        Returns:
            Product instance
        """
        # Look in the appropriate section
        section_key = (
            "perpetual_products" if market_type == MarketType.PERPETUAL else "spot_products"
        )
        products = self._fixtures.get(section_key, {})

        if symbol not in products:
            logger.warning(f"Product {symbol} not found in fixtures, creating default")
            return self._create_default_product(symbol, market_type)

        product_data = products[symbol]
        return Product(
            symbol=product_data["symbol"],
            base_asset=product_data["base_asset"],
            quote_asset=product_data["quote_asset"],
            market_type=MarketType(product_data["market_type"]),
            min_size=Decimal(product_data["min_size"]),
            step_size=Decimal(product_data["step_size"]),
            min_notional=Decimal(product_data["min_notional"]),
            price_increment=Decimal(product_data["price_increment"]),
            leverage_max=product_data["leverage_max"],
        )

    def _create_default_product(self, symbol: str, market_type: MarketType) -> Product:
        """Create a default product when fixture is not found."""
        base_asset = symbol.split("-")[0] if "-" in symbol else symbol
        quote_asset = symbol.split("-")[-1] if "-" in symbol else "USD"

        return Product(
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            market_type=market_type,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=3,
        )

    def get_default_marks(self) -> dict[str, Decimal]:
        """Get default mark prices from fixtures.

        Returns:
            Dictionary mapping symbols to mark prices
        """
        marks_data = self._fixtures.get("default_marks", {})
        return {symbol: Decimal(price) for symbol, price in marks_data.items()}

    def get_price_scenario(self, scenario: str) -> dict[str, Decimal]:
        """Get mark prices for a specific scenario.

        Args:
            scenario: Scenario name (bull_market, bear_market, sideways)

        Returns:
            Dictionary mapping symbols to mark prices
        """
        scenarios = self._fixtures.get("price_scenarios", {})
        if scenario not in scenarios:
            logger.warning(f"Price scenario {scenario} not found, using defaults")
            return self.get_default_marks()

        marks_data = scenarios[scenario]
        return {symbol: Decimal(price) for symbol, price in marks_data.items()}

    def list_perpetual_symbols(self) -> list[str]:
        """List all perpetual product symbols from fixtures.

        Returns:
            List of perpetual product symbols
        """
        products = self._fixtures.get("perpetual_products", {})
        return list(products.keys())

    def list_spot_symbols(self) -> list[str]:
        """List all spot product symbols from fixtures.

        Returns:
            List of spot product symbols
        """
        products = self._fixtures.get("spot_products", {})
        return list(products.keys())

    def create_edge_case_product(self, symbol: str) -> Product:
        """Create an edge case product for testing.

        Args:
            symbol: Edge case product symbol

        Returns:
            Product instance
        """
        edge_products = self._fixtures.get("edge_case_products", {})
        if symbol not in edge_products:
            raise ValueError(f"Edge case product {symbol} not found in fixtures")

        product_data = edge_products[symbol]
        return Product(
            symbol=product_data["symbol"],
            base_asset=product_data["base_asset"],
            quote_asset=product_data["quote_asset"],
            market_type=MarketType(product_data["market_type"]),
            min_size=Decimal(product_data["min_size"]),
            step_size=Decimal(product_data["step_size"]),
            min_notional=Decimal(product_data["min_notional"]),
            price_increment=Decimal(product_data["price_increment"]),
            leverage_max=product_data["leverage_max"],
        )


class BrokerFactory:
    """Factory for creating test brokers with fixture data."""

    def __init__(self, product_factory: ProductFactory | None = None) -> None:
        """Initialize the broker factory.

        Args:
            product_factory: Product factory instance. Creates default if None.
        """
        self.product_factory = product_factory or ProductFactory()

    def create_deterministic_broker(
        self,
        equity: Decimal = Decimal("100000"),
        symbols: list[str] | None = None,
        price_scenario: str = "default",
    ) -> DeterministicBroker:
        """Create a deterministic broker loaded with fixture data.

        Args:
            equity: Starting equity
            symbols: List of symbols to load. If None, loads default perpetuals
            price_scenario: Price scenario to use for marks

        Returns:
            Configured DeterministicBroker instance
        """
        # Import here to avoid circular imports
        from bot_v2.orchestration.deterministic_broker import DeterministicBroker

        broker = DeterministicBroker(equity=equity)

        # Load products
        if symbols is None:
            symbols = self.product_factory.list_perpetual_symbols()

        # Clear default products and load from fixtures
        broker._products.clear()
        for symbol in symbols:
            product = self.product_factory.create_product(symbol, MarketType.PERPETUAL)
            broker._products[symbol] = product

        # Load mark prices
        if price_scenario == "default":
            marks = self.product_factory.get_default_marks()
        else:
            marks = self.product_factory.get_price_scenario(price_scenario)

        broker.marks.clear()
        for symbol, price in marks.items():
            if symbol in broker._products:  # Only set marks for loaded products
                broker.marks[symbol] = price

        return broker

    def create_balances(self, usd_balance: Decimal = Decimal("100000")) -> list[Balance]:
        """Create test balances.

        Args:
            usd_balance: USD balance amount

        Returns:
            List of Balance objects
        """
        return [Balance(asset="USD", total=usd_balance, available=usd_balance, hold=Decimal("0"))]


# Global factory instances for convenience
default_product_factory = ProductFactory()
default_broker_factory = BrokerFactory(default_product_factory)


def create_test_broker(
    symbols: list[str] | None = None,
    equity: Decimal = Decimal("100000"),
    price_scenario: str = "default",
) -> DeterministicBroker:
    """Convenience function to create a test broker.

    Args:
        symbols: List of symbols to load
        equity: Starting equity
        price_scenario: Price scenario for marks

    Returns:
        Configured DeterministicBroker
    """
    return default_broker_factory.create_deterministic_broker(
        symbols=symbols,
        equity=equity,
        price_scenario=price_scenario,
    )


def create_test_product(symbol: str, market_type: MarketType = MarketType.PERPETUAL) -> Product:
    """Convenience function to create a test product.

    Args:
        symbol: Product symbol
        market_type: Market type

    Returns:
        Product instance
    """
    return default_product_factory.create_product(symbol, market_type)
