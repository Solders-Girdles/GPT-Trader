"""Tests for `StateCollector.build_positions_dict`."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from gpt_trader.features.live_trade.execution.state_collection import StateCollector


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
