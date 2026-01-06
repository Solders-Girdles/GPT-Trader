"""
Stub tests for gpt_trader.core module.

These tests verify core types can be imported and instantiated.
The core module has ZERO feature dependencies - only stdlib.
"""

from gpt_trader.core import (
    AuthError,
    BrokerageError,
    InsufficientFunds,
    MarketType,
    OrderSide,
    OrderType,
    RateLimitError,
)


class TestCoreModuleImport:
    """Test that core module exports are available."""

    def test_all_exports_importable(self) -> None:
        """Verify all __all__ exports are importable."""
        import gpt_trader.core

        for name in gpt_trader.core.__all__:
            assert hasattr(gpt_trader.core, name), f"Missing export: {name}"


class TestTradingEnums:
    """Test trading enum types."""

    def test_order_side_values(self) -> None:
        """Verify OrderSide has expected values."""
        assert OrderSide.BUY is not None
        assert OrderSide.SELL is not None

    def test_order_type_values(self) -> None:
        """Verify OrderType has expected values."""
        assert OrderType.MARKET is not None
        assert OrderType.LIMIT is not None

    def test_market_type_values(self) -> None:
        """Verify MarketType has expected values."""
        assert MarketType.SPOT is not None


class TestCoreErrors:
    """Test core error types."""

    def test_brokerage_error_is_exception(self) -> None:
        """Verify BrokerageError is an exception."""
        assert issubclass(BrokerageError, Exception)

    def test_insufficient_funds_is_exception(self) -> None:
        """Verify InsufficientFunds is an exception."""
        assert issubclass(InsufficientFunds, Exception)

    def test_auth_error_is_exception(self) -> None:
        """Verify AuthError is an exception."""
        assert issubclass(AuthError, Exception)

    def test_rate_limit_error_is_exception(self) -> None:
        """Verify RateLimitError is an exception."""
        assert issubclass(RateLimitError, Exception)
