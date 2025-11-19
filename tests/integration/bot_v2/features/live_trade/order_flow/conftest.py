"""
Shared fixtures for order flow integration tests.
"""

import pytest
from decimal import Decimal
from bot_v2.features.brokerages.core.interfaces import MarketType, Product


@pytest.fixture
def get_risk_validation_context():
    """Helper to get product and equity for risk validation."""
    def _get_context(order):
        mock_product = Product(
            symbol=order.symbol,
            base_asset="BTC" if "BTC" in order.symbol else "ETH",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10.0"),
            price_increment=Decimal("0.01"),
        )

        mock_equity = Decimal("1000000.0")  # $1,000,000 equity to avoid leverage limits

        return {"product": mock_product, "equity": mock_equity, "current_positions": {}}
    
    return _get_context
