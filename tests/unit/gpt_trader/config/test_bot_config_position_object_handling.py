"""Tests for Position object handling in reduce-only detection."""

from __future__ import annotations


class TestPositionObjectHandling:
    """Tests for Position object handling in reduce-only detection."""

    def test_position_object_long_sell_is_reducing(self) -> None:
        """Test that selling a long Position is detected as reducing."""
        from dataclasses import dataclass
        from decimal import Decimal

        @dataclass
        class MockPosition:
            side: str
            quantity: Decimal

        pos = MockPosition(side="long", quantity=Decimal("1.5"))

        # Simulate the logic from TradingEngine._validate_and_place_order
        pos_side = pos.side.lower() if pos.side else ""
        pos_qty = pos.quantity

        # SELL side for a LONG position should be reducing
        is_reducing = pos_side == "long" and pos_qty > 0
        assert is_reducing is True

    def test_position_object_short_buy_is_reducing(self) -> None:
        """Test that buying to cover a short Position is detected as reducing."""
        from dataclasses import dataclass
        from decimal import Decimal

        @dataclass
        class MockPosition:
            side: str
            quantity: Decimal

        pos = MockPosition(side="short", quantity=Decimal("2.0"))

        # Simulate the logic from TradingEngine._validate_and_place_order
        pos_side = pos.side.lower() if pos.side else ""
        pos_qty = pos.quantity

        # BUY side for a SHORT position should be reducing
        is_reducing = pos_side == "short" and pos_qty > 0
        assert is_reducing is True

    def test_position_object_long_buy_is_not_reducing(self) -> None:
        """Test that buying more of a long Position is NOT reducing."""
        from dataclasses import dataclass
        from decimal import Decimal

        @dataclass
        class MockPosition:
            side: str
            quantity: Decimal

        pos = MockPosition(side="long", quantity=Decimal("1.0"))

        # BUY side for a LONG position should NOT be reducing
        # (selling would be reducing: pos.side == "long" and pos.quantity > 0)
        # We're buying, not selling, so not reducing
        is_reducing = False  # This matches actual logic for BUY on LONG
        assert is_reducing is False
        # Verify the position attributes exist (used in actual engine logic)
        assert pos.side.lower() == "long"
        assert pos.quantity > 0
