"""Tests for position data processing and diff calculation logic."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

from bot_v2.orchestration.system_monitor_positions import PositionReconciler


class TestPositionProcessing:
    """Test position normalization and diff calculation."""

    def test_normalize_positions_valid_position_decimal_handling(
        self, reconciler: PositionReconciler
    ) -> None:
        """Test _normalize_positions handles Decimal quantities correctly."""
        position = SimpleNamespace()
        position.symbol = "BTC-PERP"
        position.quantity = Decimal("0.12345678")
        position.side = "long"

        result = reconciler._normalize_positions([position])

        assert result == {"BTC-PERP": {"quantity": "0.12345678", "side": "long"}}

    def test_normalize_positions_missing_symbol_skipped(
        self, reconciler: PositionReconciler
    ) -> None:
        """Test _normalize_positions skips entries with missing symbol."""
        position_no_symbol = SimpleNamespace()
        position_no_symbol.symbol = None
        position_no_symbol.quantity = Decimal("0.5")
        position_no_symbol.side = "long"

        position_empty_symbol = SimpleNamespace()
        position_empty_symbol.symbol = ""
        position_empty_symbol.quantity = Decimal("1.0")
        position_empty_symbol.side = "short"

        result = reconciler._normalize_positions([position_no_symbol, position_empty_symbol])

        assert result == {}

    def test_normalize_positions_quantity_from_exception(
        self, reconciler: PositionReconciler, caplog
    ) -> None:
        """Test _normalize_positions handles quantity_from exceptions gracefully."""
        valid_pos = SimpleNamespace()
        valid_pos.symbol = "BTC-PERP"
        valid_pos.quantity = Decimal("0.5")
        valid_pos.side = "long"

        bad_pos = SimpleNamespace()
        bad_pos.symbol = "ETH-PERP"
        bad_pos.quantity = "invalid_quantity"
        bad_pos.side = "short"

        # Mock quantity_from to raise for bad position
        with patch("bot_v2.orchestration.system_monitor_positions.quantity_from") as mock_quantity:

            def quantity_side_effect(pos):
                if pos == bad_pos:
                    raise ValueError("Cannot convert quantity")
                return Decimal("0.5")

            mock_quantity.side_effect = quantity_side_effect

            result = reconciler._normalize_positions([valid_pos, bad_pos])

            # Should contain only valid position
            expected = {"BTC-PERP": {"quantity": "0.5", "side": "long"}}
            assert result == expected

            # Should log the exception for bad position
            assert "Failed to normalize position" in caplog.text
            assert "ETH-PERP" in caplog.text
            assert "Cannot convert quantity" in caplog.text

    def test_normalize_positions_ensures_string_conversion(
        self, reconciler: PositionReconciler
    ) -> None:
        """Test _normalize_positions ensures string output for all values."""
        position = SimpleNamespace()
        position.symbol = "BTC-PERP"
        position.quantity = Decimal("0")
        position.side = "short"

        result = reconciler._normalize_positions([position])

        btc_data = result["BTC-PERP"]
        assert isinstance(btc_data["quantity"], str)
        assert isinstance(btc_data["side"], str)
        assert btc_data["quantity"] == "0"
        assert btc_data["side"] == "short"

    def test_calculate_diff_new_symbol(self, reconciler: PositionReconciler) -> None:
        """Test _calculate_diff detects new symbols (old empty)."""
        previous = {}
        current = {"BTC-PERP": {"quantity": "0.5", "side": "long"}}

        result = reconciler._calculate_diff(previous, current)

        expected = {"BTC-PERP": {"old": {}, "new": {"quantity": "0.5", "side": "long"}}}
        assert result == expected

    def test_calculate_diff_quantity_change(self, reconciler: PositionReconciler) -> None:
        """Test _calculate_diff detects quantity changes."""
        previous = {"BTC-PERP": {"quantity": "0.3", "side": "long"}}
        current = {"BTC-PERP": {"quantity": "0.5", "side": "long"}}

        result = reconciler._calculate_diff(previous, current)

        expected = {
            "BTC-PERP": {
                "old": {"quantity": "0.3", "side": "long"},
                "new": {"quantity": "0.5", "side": "long"},
            }
        }
        assert result == expected

    def test_calculate_diff_side_change(self, reconciler: PositionReconciler) -> None:
        """Test _calculate_diff detects side changes."""
        previous = {"BTC-PERP": {"quantity": "0.5", "side": "long"}}
        current = {"BTC-PERP": {"quantity": "0.5", "side": "short"}}

        result = reconciler._calculate_diff(previous, current)

        expected = {
            "BTC-PERP": {
                "old": {"quantity": "0.5", "side": "long"},
                "new": {"quantity": "0.5", "side": "short"},
            }
        }
        assert result == expected

    def test_calculate_diff_removal_case(self, reconciler: PositionReconciler) -> None:
        """Test _calculate_diff detects removed symbols (new empty)."""
        previous = {"BTC-PERP": {"quantity": "0.5", "side": "long"}}
        current = {}

        result = reconciler._calculate_diff(previous, current)

        expected = {"BTC-PERP": {"old": {"quantity": "0.5", "side": "long"}, "new": {}}}
        assert result == expected

    def test_calculate_diff_identical_snapshot_empty_result(
        self, reconciler: PositionReconciler
    ) -> None:
        """Test _calculate_diff returns empty dict for identical snapshots."""
        snapshot = {
            "BTC-PERP": {"quantity": "0.5", "side": "long"},
            "ETH-PERP": {"quantity": "1.0", "side": "short"},
        }

        result = reconciler._calculate_diff(snapshot, snapshot)

        assert result == {}

    def test_calculate_diff_complex_multiple_changes(self, reconciler: PositionReconciler) -> None:
        """Test _calculate_diff handles multiple changes simultaneously."""
        previous = {
            "BTC-PERP": {"quantity": "0.3", "side": "long"},
            "ETH-PERP": {"quantity": "1.0", "side": "short"},
            "SOL-PERP": {"quantity": "100", "side": "long"},
        }
        current = {
            "BTC-PERP": {"quantity": "0.5", "side": "long"},  # Quantity change
            "ETH-PERP": {"quantity": "1.0", "side": "long"},  # Side change
            # SOL-PERP removed
            "ADA-PERP": {"quantity": "500", "side": "short"},  # New symbol
        }

        result = reconciler._calculate_diff(previous, current)

        # Should have changes for all affected symbols
        assert "BTC-PERP" in result
        assert "ETH-PERP" in result
        assert "SOL-PERP" in result
        assert "ADA-PERP" in result

        # Verify specific changes
        assert result["BTC-PERP"]["old"]["quantity"] == "0.3"
        assert result["BTC-PERP"]["new"]["quantity"] == "0.5"
        assert result["ETH-PERP"]["old"]["side"] == "short"
        assert result["ETH-PERP"]["new"]["side"] == "long"
        assert result["SOL-PERP"]["new"] == {}
        assert result["ADA-PERP"]["old"] == {}

    def test_calculate_diff_empty_inputs(self, reconciler: PositionReconciler) -> None:
        """Test _calculate_diff handles empty inputs gracefully."""
        # Both empty
        result = reconciler._calculate_diff({}, {})
        assert result == {}

        # Previous empty, current empty
        result = reconciler._calculate_diff({}, {})
        assert result == {}
