"""Tests for PositionReconciler class initialization and basic functionality."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from bot_v2.orchestration.system_monitor_positions import PositionReconciler


@pytest.mark.asyncio
class TestPositionReconciler:
    """Test PositionReconciler initialization and basic methods."""

    def test_construction_stores_dependencies(self, fake_event_store: MagicMock) -> None:
        """Test that PositionReconciler stores provided dependencies correctly."""
        reconciler = PositionReconciler(event_store=fake_event_store, bot_id="test-bot")

        assert reconciler._event_store == fake_event_store
        assert reconciler._bot_id == "test-bot"

    async def test_fetch_positions_happy_path(self, reconciler: PositionReconciler, fake_bot) -> None:
        """Test _fetch_positions returns broker list on success."""
        sample_positions = [
            MagicMock(symbol="BTC-PERP", quantity=Decimal("0.5"), side="long"),
            MagicMock(symbol="ETH-PERP", quantity=Decimal("1.0"), side="short"),
        ]
        fake_bot.broker.list_positions.return_value = sample_positions

        result = await reconciler._fetch_positions(fake_bot)

        assert result == sample_positions
        fake_bot.broker.list_positions.assert_called_once()

    async def test_fetch_positions_failure_path(self, reconciler: PositionReconciler, fake_bot) -> None:
        """Test _fetch_positions returns empty list on broker exception."""
        fake_bot.broker.list_positions.side_effect = Exception("Broker error")

        result = await reconciler._fetch_positions(fake_bot)

        assert result == []

    def test_normalize_positions_valid_entries(
        self, reconciler: PositionReconciler, sample_positions
    ) -> None:
        """Test _normalize_positions processes valid positions correctly."""
        result = reconciler._normalize_positions(sample_positions)

        expected = {
            "BTC-PERP": {"quantity": "0.5", "side": "long"},
            "ETH-PERP": {"quantity": "1.0", "side": "short"},
        }
        assert result == expected

    def test_normalize_positions_missing_symbol_skipped(
        self, reconciler: PositionReconciler, caplog
    ) -> None:
        """Test _normalize_positions skips entries without symbol."""
        # Create position without symbol
        position_no_symbol = MagicMock()
        position_no_symbol.symbol = None
        position_no_symbol.quantity = Decimal("0.5")
        position_no_symbol.side = "long"

        result = reconciler._normalize_positions([position_no_symbol])

        assert result == {}

    def test_normalize_positions_exception_logs_and_continues(
        self, reconciler: PositionReconciler, caplog
    ) -> None:
        """Test _normalize_positions logs exceptions but continues processing."""
        # Create positions: one valid, one that raises during quantity extraction
        valid_pos = MagicMock()
        valid_pos.symbol = "BTC-PERP"
        valid_pos.quantity = Decimal("0.5")
        valid_pos.side = "long"

        bad_pos = MagicMock()
        bad_pos.symbol = "ETH-PERP"
        # quantity_from will raise an exception for this position

        # Mock quantity_from to raise for bad position
        with patch("bot_v2.orchestration.system_monitor_positions.quantity_from") as mock_quantity:
            def quantity_side_effect(pos):
                if pos == bad_pos:
                    raise ValueError("Invalid quantity")
                return Decimal("0.5")

            mock_quantity.side_effect = quantity_side_effect

            result = reconciler._normalize_positions([valid_pos, bad_pos])

            # Should contain only the valid position
            expected = {"BTC-PERP": {"quantity": "0.5", "side": "long"}}
            assert result == expected

            # Should have logged the exception
            assert "Failed to normalize position" in caplog.text
            assert "ETH-PERP" in caplog.text

    def test_normalize_positions_ensures_string_output(
        self, reconciler: PositionReconciler
    ) -> None:
        """Test _normalize_positions converts quantities and sides to strings."""
        pos = MagicMock()
        pos.symbol = "BTC-PERP"
        pos.quantity = Decimal("0.5")
        pos.side = "long"

        result = reconciler._normalize_positions([pos])

        # Verify all values are strings
        btc_data = result["BTC-PERP"]
        assert isinstance(btc_data["quantity"], str)
        assert isinstance(btc_data["side"], str)
        assert btc_data["quantity"] == "0.5"
        assert btc_data["side"] == "long"

    def test_normalize_positions_empty_list(self, reconciler: PositionReconciler) -> None:
        """Test _normalize_positions handles empty list gracefully."""
        result = reconciler._normalize_positions([])
        assert result == {}

    def test_normalize_positions_none_list(self, reconciler: PositionReconciler) -> None:
        """Test _normalize_positions handles None list gracefully."""
        result = reconciler._normalize_positions(None)
        assert result == {}
