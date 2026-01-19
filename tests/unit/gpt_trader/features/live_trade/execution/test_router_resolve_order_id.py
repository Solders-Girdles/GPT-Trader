"""Tests for router order id resolution helpers."""

from unittest.mock import Mock


class TestResolveOrderIdFallback:
    """Tests for _resolve_order_id fallback behavior."""

    def test_resolve_order_id_with_id_attribute(self) -> None:
        """Test that 'id' attribute is used first."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        order = Mock()
        order.id = "primary-id-123"
        order.order_id = "fallback-id-456"

        result = _resolve_order_id(order)
        assert result == "primary-id-123"

    def test_resolve_order_id_fallback_to_order_id(self) -> None:
        """Test fallback to 'order_id' when 'id' is None."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        order = Mock()
        order.id = None
        order.order_id = "fallback-id-456"

        result = _resolve_order_id(order)
        assert result == "fallback-id-456"

    def test_resolve_order_id_fallback_when_id_missing(self) -> None:
        """Test fallback to 'order_id' when 'id' attribute doesn't exist."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        order = Mock(spec=["order_id"])
        order.order_id = "fallback-id-789"

        result = _resolve_order_id(order)
        assert result == "fallback-id-789"

    def test_resolve_order_id_returns_none_for_none_order(self) -> None:
        """Test that None order returns None."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        result = _resolve_order_id(None)
        assert result is None

    def test_resolve_order_id_returns_none_when_both_missing(self) -> None:
        """Test that None is returned when both id and order_id are missing/None."""
        from gpt_trader.features.live_trade.execution.router import _resolve_order_id

        order = Mock(spec=[])

        result = _resolve_order_id(order)
        assert result is None
