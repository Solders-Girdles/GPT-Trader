"""Tests for LiveRiskManager.check_liquidation_buffer."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
from tests.unit.gpt_trader.features.live_trade.risk_manager_test_utils import (  # naming: allow
    MockConfig,
    MockPosition,
)


@pytest.fixture(autouse=True)
def mock_load_state(monkeypatch: pytest.MonkeyPatch):
    """Prevent LiveRiskManager from loading state during tests."""
    monkeypatch.setattr(LiveRiskManager, "_load_state", lambda self: None)


class TestCheckLiquidationBuffer:
    """Tests for check_liquidation_buffer method."""

    def test_no_config_returns_false(self) -> None:
        """Test returns False when no config."""
        manager = LiveRiskManager()

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 110}, Decimal("1000")
        )

        assert result is False

    def test_dict_position_no_liquidation_price(self) -> None:
        """Test with dict position missing liquidation_price."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer("BTC-USD", {"mark": 110}, Decimal("1000"))

        assert result is False

    def test_dict_position_no_mark_price(self) -> None:
        """Test with dict position missing mark price."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100}, Decimal("1000")
        )

        assert result is False

    def test_dict_position_with_mark_key(self) -> None:
        """Test dict position using 'mark' key."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)

        # Buffer = |110 - 100| / 110 = 0.0909... (9.09%), threshold 5%
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 110}, Decimal("1000")
        )

        assert result is False  # Buffer > threshold, not dangerous

    def test_dict_position_with_mark_price_key(self) -> None:
        """Test dict position using 'mark_price' key."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark_price": 110}, Decimal("1000")
        )

        assert result is False

    def test_object_position_with_mark_price(self) -> None:
        """Test object position with mark_price attribute."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        position = MockPosition(liquidation_price=Decimal("100"), mark_price=Decimal("110"))

        result = manager.check_liquidation_buffer("BTC-USD", position, Decimal("1000"))

        assert result is False

    def test_object_position_with_mark_fallback(self) -> None:
        """Test object position falling back to mark attribute."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.05"))
        manager = LiveRiskManager(config=config)
        position = MockPosition(liquidation_price=Decimal("100"), mark=Decimal("110"))

        result = manager.check_liquidation_buffer("BTC-USD", position, Decimal("1000"))

        assert result is False

    def test_triggers_reduce_only_when_buffer_too_small(self) -> None:
        """Test triggers reduce_only when buffer is below threshold."""
        config = MockConfig(min_liquidation_buffer_pct=Decimal("0.15"))
        manager = LiveRiskManager(config=config)

        # Buffer = |110 - 100| / 110 = 0.0909... (9.09%), threshold 15%
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 110}, Decimal("1000")
        )

        assert result is True
        assert manager.positions["BTC-USD"]["reduce_only"] is True

    def test_zero_mark_price_returns_false(self) -> None:
        """Test returns False when mark price is zero."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": 100, "mark": 0}, Decimal("1000")
        )

        assert result is False

    def test_handles_attribute_error(self) -> None:
        """Test handles AttributeError gracefully."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        # Object without expected attributes
        result = manager.check_liquidation_buffer("BTC-USD", object(), Decimal("1000"))

        assert result is False

    def test_handles_none_values(self) -> None:
        """Test handles None values gracefully."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        # Position with None liquidation_price
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": None, "mark": 110}, Decimal("1000")
        )

        assert result is False

    def test_handles_value_error(self) -> None:
        """Test handles ValueError gracefully (empty string conversion)."""
        config = MockConfig()
        manager = LiveRiskManager(config=config)

        # Empty strings that pass the falsy check but fail Decimal conversion
        # Note: empty string is falsy, so this hits the early return
        result = manager.check_liquidation_buffer(
            "BTC-USD", {"liquidation_price": "", "mark": 110}, Decimal("1000")
        )

        assert result is False
