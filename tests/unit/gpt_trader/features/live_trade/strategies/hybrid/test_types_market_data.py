"""Tests for hybrid strategy market data model."""

from decimal import Decimal

from gpt_trader.features.live_trade.strategies.hybrid.types import HybridMarketData


class TestHybridMarketData:
    """Tests for HybridMarketData dataclass."""

    def test_spot_only_data(self):
        """Can create with spot data only."""
        data = HybridMarketData(
            symbol="BTC-USD",
            spot_price=Decimal("50000"),
        )
        assert data.symbol == "BTC-USD"
        assert data.spot_price == Decimal("50000")
        assert data.futures_price is None
        assert data.basis is None
        assert data.basis_percentage is None

    def test_full_market_data(self):
        """Can create with all market data."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
            futures_price=Decimal("50500"),
            spot_bid=Decimal("49990"),
            spot_ask=Decimal("50010"),
            futures_bid=Decimal("50490"),
            futures_ask=Decimal("50510"),
            funding_rate=Decimal("0.0001"),
            mark_price=Decimal("50250"),
        )
        assert data.futures_price == Decimal("50500")
        assert data.spot_bid == Decimal("49990")
        assert data.funding_rate == Decimal("0.0001")

    def test_basis_calculation(self):
        """Basis calculated correctly."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
            futures_price=Decimal("50500"),
        )
        assert data.basis == Decimal("500")

    def test_basis_percentage_premium(self):
        """Basis percentage calculated for premium."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
            futures_price=Decimal("50500"),
        )
        assert data.basis_percentage == Decimal("1")  # 500/50000 * 100 = 1%

    def test_basis_percentage_discount(self):
        """Basis percentage calculated for discount."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
            futures_price=Decimal("49750"),
        )
        assert data.basis_percentage == Decimal("-0.5")  # -250/50000 * 100 = -0.5%

    def test_basis_percentage_no_futures(self):
        """Basis percentage is None without futures."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("50000"),
        )
        assert data.basis_percentage is None

    def test_basis_percentage_zero_spot(self):
        """Basis percentage is None with zero spot price."""
        data = HybridMarketData(
            symbol="BTC",
            spot_price=Decimal("0"),
            futures_price=Decimal("50000"),
        )
        assert data.basis_percentage is None
